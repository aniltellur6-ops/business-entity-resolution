import pandas as pd
import logging
from .tfidf_blocking import TfidfBlocker
from .token_blocking import TokenBlocker
from .postal_blocking import PostalBlocker

logger = logging.getLogger(__name__)

class CandidateGenerator:
    """Orchestrates multiple blocking strategies and unifies the candidates."""
    
    def __init__(self):
        self.tfidf_blocker = TfidfBlocker()
        self.token_blocker = TokenBlocker()
        self.postal_blocker = PostalBlocker()
        
    def _combine_candidates(self, c_tfidf, c_token, c_postal, source_name):
        c_tfidf = c_tfidf.copy()
        c_token = c_token.copy()
        c_postal = c_postal.copy()
        
        c_tfidf['found_by_tfidf'] = 1
        c_token['found_by_token'] = 1
        c_postal['found_by_postal'] = 1
        
        # Concat all
        all_cands = pd.concat([c_tfidf, c_token, c_postal], ignore_index=True)
        if all_cands.empty:
            return pd.DataFrame(columns=['s1_id', 'candidate_id', 'source', 'found_by_tfidf', 'found_by_token', 'found_by_postal'])
            
        all_cands['source'] = source_name
        
        # Ensure columns exist even if one of the methods returned empty df
        for col in ['found_by_tfidf', 'found_by_token', 'found_by_postal']:
            if col not in all_cands.columns:
                all_cands[col] = 0
                
        # Fill NA for missing flags before groupby
        all_cands[['found_by_tfidf', 'found_by_token', 'found_by_postal']] = all_cands[['found_by_tfidf', 'found_by_token', 'found_by_postal']].fillna(0)
        
        # Group by pair and take max of flags
        dedup = all_cands.groupby(['s1_id', 'candidate_id', 'source'], as_index=False).agg({
            'found_by_tfidf': 'max',
            'found_by_token': 'max',
            'found_by_postal': 'max'
        })
        return dedup

    def generate_candidates(self, s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates candidates for S1 from S2 and S3, taking the union of all blocking methods.
        """
        # Block against S2
        logger.info("Generating candidates against S2...")
        c_tfidf_s2 = self.tfidf_blocker.block(s1_df, s2_df)
        c_token_s2 = self.token_blocker.block(s1_df, s2_df)
        c_postal_s2 = self.postal_blocker.block(s1_df, s2_df)
        s2_candidates = self._combine_candidates(c_tfidf_s2, c_token_s2, c_postal_s2, 's2')
        
        # Block against S3
        logger.info("Generating candidates against S3...")
        c_tfidf_s3 = self.tfidf_blocker.block(s1_df, s3_df)
        c_token_s3 = self.token_blocker.block(s1_df, s3_df)
        c_postal_s3 = self.postal_blocker.block(s1_df, s3_df)
        s3_candidates = self._combine_candidates(c_tfidf_s3, c_token_s3, c_postal_s3, 's3')
        
        # Combine and deduplicate
        final_candidates = pd.concat([s2_candidates, s3_candidates], ignore_index=True)
        logger.info(f"Total unique candidate pairs: {len(final_candidates)}")
        
        return final_candidates
