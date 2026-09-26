import pandas as pd
import logging

logger = logging.getLogger(__name__)

class PostalBlocker:
    """Uses exact match on extracted postal code + first letter of name as a fallback blocker."""
    
    def block(self, s1_df: pd.DataFrame, s_other_df: pd.DataFrame, s1_id_col='id', other_id_col='id') -> pd.DataFrame:
        """
        Finds candidates matching postal code and at least the same first letter of name.
        This prevents associating every business in a zip code.
        """
        logger.info(f"Postal Blocking: {len(s1_df)} S1 records vs {len(s_other_df)} candidates.")
        
        def get_postal_sig(row):
            postal = str(row.get('postal_code', ''))
            name = str(row.get('normalized_name', ''))
            if not postal or postal == 'None' or not name:
                return ""
            return f"{postal}_{name[0]}"
            
        s1_sigs = s1_df.apply(get_postal_sig, axis=1)
        other_sigs = s_other_df.apply(get_postal_sig, axis=1)
        
        s1_map = pd.DataFrame({'s1_id': s1_df[s1_id_col], 'sig': s1_sigs})
        other_map = pd.DataFrame({'candidate_id': s_other_df[other_id_col], 'sig': other_sigs})
        
        s1_map = s1_map[s1_map['sig'] != ""]
        other_map = other_map[other_map['sig'] != ""]
        
        candidates_df = pd.merge(s1_map, other_map, on='sig', how='inner')
        candidates_df['method'] = 'postal'
        
        return candidates_df[['s1_id', 'candidate_id', 'method']]
