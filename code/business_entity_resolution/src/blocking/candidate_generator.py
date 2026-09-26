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
        
    def generate_candidates(self, s1_df: pd.DataFrame, s2_df: pd.DataFrame, s3_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generates candidates for S1 from S2 and S3, taking the union of all blocking methods.
        """
        all_candidates = []
        
        # Block against S2
        logger.info("Generating candidates against S2...")
        c_tfidf_s2 = self.tfidf_blocker.block(s1_df, s2_df)
        c_token_s2 = self.token_blocker.block(s1_df, s2_df)
        c_postal_s2 = self.postal_blocker.block(s1_df, s2_df)
        
        s2_candidates = pd.concat([c_tfidf_s2, c_token_s2, c_postal_s2], ignore_index=True)
        s2_candidates['source'] = 's2'
        all_candidates.append(s2_candidates)
        
        # Block against S3
        logger.info("Generating candidates against S3...")
        c_tfidf_s3 = self.tfidf_blocker.block(s1_df, s3_df)
        c_token_s3 = self.token_blocker.block(s1_df, s3_df)
        c_postal_s3 = self.postal_blocker.block(s1_df, s3_df)
        
        s3_candidates = pd.concat([c_tfidf_s3, c_token_s3, c_postal_s3], ignore_index=True)
        s3_candidates['source'] = 's3'
        all_candidates.append(s3_candidates)
        
        # Combine and deduplicate
        final_candidates = pd.concat(all_candidates, ignore_index=True)
        logger.info(f"Total candidates before deduplication: {len(final_candidates)}")
        
        # Deduplicate, keeping track of methods that found it (optional, here we just keep the first method)
        final_candidates = final_candidates.drop_duplicates(subset=['s1_id', 'candidate_id'])
        logger.info(f"Total unique candidate pairs: {len(final_candidates)}")
        
        return final_candidates
