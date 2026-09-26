import pandas as pd
import logging

logger = logging.getLogger(__name__)

def evaluate_blocking_recall(candidates_df: pd.DataFrame, ground_truth_df: pd.DataFrame) -> dict:
    """
    Measures how many true matches from ground truth survived the blocking stage.
    """
    if ground_truth_df.empty:
        logger.warning("Ground truth is empty. Cannot evaluate recall.")
        return {}
        
    # Assume ground truth has columns: source1_entity_id, source2_entity_ids, source3_entity_ids
    # We need to melt it into pairwise matches
    true_pairs = []
    
    for _, row in ground_truth_df.iterrows():
        s1 = row.get('source1_entity_id')
        
        # S2
        s2_ids = row.get('source2_entity_ids', '')
        if pd.notna(s2_ids) and s2_ids:
            for s2 in s2_ids.split(','):
                if s2.strip():
                    true_pairs.append({'s1_id': s1, 'candidate_id': s2.strip()})
                    
        # S3
        s3_ids = row.get('source3_entity_ids', '')
        if pd.notna(s3_ids) and s3_ids:
            for s3 in s3_ids.split(','):
                if s3.strip():
                    true_pairs.append({'s1_id': s1, 'candidate_id': s3.strip()})
                    
    true_pairs_df = pd.DataFrame(true_pairs)
    if true_pairs_df.empty:
        return {"recall": 0, "found": 0, "total_true": 0}
        
    total_true = len(true_pairs_df)
    
    # Merge candidates with true pairs
    # candidates_df should have s1_id and candidate_id
    found_df = pd.merge(true_pairs_df, candidates_df[['s1_id', 'candidate_id']], 
                        on=['s1_id', 'candidate_id'], how='inner')
                        
    found = len(found_df)
    recall = found / total_true if total_true > 0 else 0
    
    logger.info(f"Blocking Recall: {recall:.4f} ({found}/{total_true} true pairs found)")
    
    return {
        "recall": recall,
        "found": found,
        "total_true": total_true
    }
