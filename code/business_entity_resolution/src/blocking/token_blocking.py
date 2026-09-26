import pandas as pd
import logging

logger = logging.getLogger(__name__)

class TokenBlocker:
    """Uses sorted token signatures to find candidates regardless of word order."""
    
    def block(self, s1_df: pd.DataFrame, s_other_df: pd.DataFrame, s1_id_col='id', other_id_col='id') -> pd.DataFrame:
        """
        Finds candidates by matching sorted token strings.
        """
        logger.info(f"Token Blocking: {len(s1_df)} S1 records vs {len(s_other_df)} candidates.")
        
        def get_signature(text):
            if not isinstance(text, str) or not text:
                return ""
            tokens = text.split()
            return " ".join(sorted(tokens))
            
        s1_sigs = s1_df['normalized_name'].apply(get_signature)
        other_sigs = s_other_df['normalized_name'].apply(get_signature)
        
        # Create mapping dataframes
        s1_map = pd.DataFrame({'s1_id': s1_df[s1_id_col], 'sig': s1_sigs})
        other_map = pd.DataFrame({'candidate_id': s_other_df[other_id_col], 'sig': other_sigs})
        
        # Drop empty signatures
        s1_map = s1_map[s1_map['sig'] != ""]
        other_map = other_map[other_map['sig'] != ""]
        
        # Merge on signature
        candidates_df = pd.merge(s1_map, other_map, on='sig', how='inner')
        candidates_df['method'] = 'token'
        
        return candidates_df[['s1_id', 'candidate_id', 'method']]
