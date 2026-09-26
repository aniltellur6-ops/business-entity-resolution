import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

def get_top_k_sparse(s1_vecs, s_other_vecs, k=20, threshold=0.4, chunk_size=5000):
    """
    Compute pairwise cosine similarity using sparse matrix multiplication in chunks.
    Avoids scikit-learn's NearestNeighbors bottleneck on massive datasets.
    """
    # Transpose target vectors for fast dot product
    s_other_vecs_T = s_other_vecs.T.tocsc()
    
    row_indices = []
    col_indices = []
    
    num_queries = s1_vecs.shape[0]
    
    logger.info(f"Computing sparse dot products in chunks of {chunk_size}...")
    for start_idx in tqdm(range(0, num_queries, chunk_size), desc="TF-IDF Blocking"):
        end_idx = min(start_idx + chunk_size, num_queries)
        chunk = s1_vecs[start_idx:end_idx]
        
        # Multiply chunk with targets. Result is sparse (chunk_size, num_targets)
        sim_chunk = chunk.dot(s_other_vecs_T)
        
        for row_i in range(sim_chunk.shape[0]):
            row_start = sim_chunk.indptr[row_i]
            row_end = sim_chunk.indptr[row_i+1]
            
            if row_start == row_end:
                continue
                
            cols = sim_chunk.indices[row_start:row_end]
            data = sim_chunk.data[row_start:row_end]
            
            # Filter by threshold (cosine similarity > 0.4 equals cosine distance < 0.6)
            valid = data >= threshold
            cols = cols[valid]
            data = data[valid]
            
            if len(data) == 0:
                continue
                
            # If we have more than K, take top K
            if len(data) > k:
                top_k_idx = np.argpartition(-data, k - 1)[:k]
                cols = cols[top_k_idx]
                
            global_row_i = start_idx + row_i
            row_indices.extend([global_row_i] * len(cols))
            col_indices.extend(cols)
            
    return row_indices, col_indices

class TfidfBlocker:
    """Uses character n-gram TF-IDF and highly optimized sparse matrix multiplication."""
    
    def __init__(self, ngram_range=(2, 4), n_neighbors=20, threshold=0.4):
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, min_df=2)
        self.n_neighbors = n_neighbors
        self.threshold = threshold
        
    def block(self, s1_df: pd.DataFrame, s_other_df: pd.DataFrame, s1_id_col='id', other_id_col='id') -> pd.DataFrame:
        """
        Finds candidates for S1 in S_other.
        Returns a DataFrame of (s1_id, candidate_id)
        """
        logger.info(f"TF-IDF Blocking: {len(s1_df)} S1 records vs {len(s_other_df)} candidates.")
        
        # We need a unified space. We fit on both to get the full vocabulary.
        all_names = pd.concat([s1_df['normalized_name'], s_other_df['normalized_name']]).fillna('')
        self.vectorizer.fit(all_names)
        
        s1_vecs = self.vectorizer.transform(s1_df['normalized_name'].fillna(''))
        s_other_vecs = self.vectorizer.transform(s_other_df['normalized_name'].fillna(''))
        
        # Adjust n_neighbors if the target set is smaller than n_neighbors
        k = min(self.n_neighbors, len(s_other_df))
        
        row_indices, col_indices = get_top_k_sparse(s1_vecs, s_other_vecs, k=k, threshold=self.threshold)
        
        # Build candidate pairs
        s1_ids_array = s1_df[s1_id_col].values
        other_ids_array = s_other_df[other_id_col].values
        
        matched_s1_ids = s1_ids_array[row_indices]
        matched_candidate_ids = other_ids_array[col_indices]
        
        candidates_df = pd.DataFrame({
            's1_id': matched_s1_ids,
            'candidate_id': matched_candidate_ids,
            'method': 'tfidf'
        })
                    
        return candidates_df
