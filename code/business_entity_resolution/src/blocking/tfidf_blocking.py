import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import logging

logger = logging.getLogger(__name__)

class TfidfBlocker:
    """Uses character n-gram TF-IDF and Approximate Nearest Neighbors to find candidates."""
    
    def __init__(self, ngram_range=(2, 4), n_neighbors=20):
        self.vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=ngram_range, min_df=2)
        self.nn = NearestNeighbors(n_neighbors=n_neighbors, metric='cosine', n_jobs=-1)
        
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
        
        self.nn.fit(s_other_vecs)
        
        # Adjust n_neighbors if the target set is smaller than n_neighbors
        k = min(self.nn.n_neighbors, len(s_other_df))
        
        distances, indices = self.nn.kneighbors(s1_vecs, n_neighbors=k)
        
        # Build candidate pairs
        candidates = []
        s1_ids = s1_df[s1_id_col].values
        other_ids = s_other_df[other_id_col].values
        
        for i, s1_id in enumerate(s1_ids):
            for j in range(k):
                # distance threshold could be applied here to reduce candidate bloat
                if distances[i, j] < 0.6: # example threshold (cosine distance)
                    candidates.append({
                        's1_id': s1_id,
                        'candidate_id': other_ids[indices[i, j]],
                        'method': 'tfidf'
                    })
                    
        return pd.DataFrame(candidates)
