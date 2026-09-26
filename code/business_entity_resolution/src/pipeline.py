import logging
import pandas as pd
from pathlib import Path

from data.loader import DataLoader
from normalization.name_normalizer import normalize_business_name
from normalization.address_normalizer import normalize_address
from normalization.structured_fields import extract_postal_code, extract_house_number, extract_city
from blocking.candidate_generator import CandidateGenerator
from blocking.blocking_recall import evaluate_blocking_recall
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all normalization and extraction to a dataframe."""
    if df is None or df.empty:
        return df
        
    logger.info(f"Normalizing {len(df)} records...")
    
    result_df = df.copy()
    
    if 'business_name' in result_df.columns:
        norm_names = result_df['business_name'].apply(
            lambda x: normalize_business_name(str(x)) if pd.notna(x) else {"normalized_name": "", "legal_suffix": ""}
        )
        result_df['normalized_name'] = norm_names.apply(lambda x: x.get('normalized_name', ''))
        result_df['legal_suffix'] = norm_names.apply(lambda x: x.get('legal_suffix', ''))
    
    if 'address' in result_df.columns:
        norm_addrs = result_df['address'].apply(
            lambda x: normalize_address(str(x)) if pd.notna(x) else {"normalized_address": ""}
        )
        result_df['normalized_address'] = norm_addrs.apply(lambda x: x.get('normalized_address', ''))
        
        # Extract structured fields
        result_df['postal_code'] = result_df['address'].apply(
            lambda x: extract_postal_code(str(x)) if pd.notna(x) else None
        )
        result_df['house_number'] = result_df['address'].apply(
            lambda x: extract_house_number(str(x)) if pd.notna(x) else None
        )
        result_df['city_extracted'] = result_df['address'].apply(
            lambda x: extract_city(str(x)) if pd.notna(x) else None
        )
        
    return result_df

def create_true_pairs_set(ground_truth_df: pd.DataFrame) -> set:
    """Creates a set of true matching pairs (s1_id, candidate_id)."""
    if ground_truth_df is None or ground_truth_df.empty:
        return set()
    
    true_pairs = set()
    for _, row in ground_truth_df.iterrows():
        s1 = row.get('source1_entity_id')
        s2_ids = row.get('source2_entity_ids', '')
        if pd.notna(s2_ids) and s2_ids:
            for s2 in s2_ids.split(','):
                if s2.strip(): true_pairs.add((s1, s2.strip()))
        
        s3_ids = row.get('source3_entity_ids', '')
        if pd.notna(s3_ids) and s3_ids:
            for s3 in s3_ids.split(','):
                if s3.strip(): true_pairs.add((s1, s3.strip()))
                
    return true_pairs

def main():
    logger.info("Starting Business Entity Resolution Pipeline (Shriyash's part)")
    
    # 1. Load Data
    loader = DataLoader(config.DATA_DIR)
    
    # Assuming we are running training mode right now to pass to Despo's model
    datasets = loader.load_all_train() 
    
    s1_df = datasets.get("source1")
    s2_df = datasets.get("source2")
    s3_df = datasets.get("source3")
    gt_df = datasets.get("ground_truth")
    
    if s1_df.empty or s2_df.empty or s3_df.empty:
        logger.error("One or more datasets could not be loaded. Please ensure data is in data/ directory.")
        return
        
    # 2. Normalize
    s1_norm = apply_normalization(s1_df)
    s2_norm = apply_normalization(s2_df)
    s3_norm = apply_normalization(s3_df)
    
    # 3. Generate Candidates
    generator = CandidateGenerator()
    candidates = generator.generate_candidates(s1_norm, s2_norm, s3_norm)
    
    # 4. Enrich Candidates for Despo's Feature Engineering
    logger.info("Enriching candidates with raw text for ML feature engineering...")
    s1_names = dict(zip(s1_norm['id'], s1_norm['normalized_name']))
    s2_names = dict(zip(s2_norm['id'], s2_norm['normalized_name']))
    s3_names = dict(zip(s3_norm['id'], s3_norm['normalized_name']))
    
    candidates['name_1'] = candidates['s1_id'].map(s1_names)
    
    def get_other_name(row):
        if row['source'] == 's2': return s2_names.get(row['candidate_id'], '')
        elif row['source'] == 's3': return s3_names.get(row['candidate_id'], '')
        return ''
        
    candidates['name_2'] = candidates.apply(get_other_name, axis=1)
    
    # Add labels if we have ground truth
    if gt_df is not None and not gt_df.empty:
        logger.info("Adding ground truth labels to candidate pairs...")
        true_pairs = create_true_pairs_set(gt_df)
        candidates['label'] = candidates.apply(
            lambda x: 1 if (x['s1_id'], x['candidate_id']) in true_pairs else 0, axis=1
        )
        
        # Log recall
        evaluate_blocking_recall(candidates, gt_df)
    
    # 5. Save candidate pairs
    output_file = config.OUTPUT_DIR / "candidate_pairs.tsv"
    candidates.to_csv(output_file, sep="\t", index=False)
    logger.info(f"Saved {len(candidates)} candidate pairs to {output_file}")
    
    # 6. Model Training (Despo's Pipeline)
    try:
        from model.train_model import main as train_model_main
        if 'label' in candidates.columns:
            logger.info("Triggering Model Training pipeline...")
            train_model_main()
    except ImportError:
        logger.warning("Despo's model training pipeline not found or not importable.")

if __name__ == "__main__":
    main()
