import logging
import pandas as pd
from pathlib import Path

from data.loader import DataLoader
from normalization.name_normalizer import normalize_business_name
from normalization.address_normalizer import normalize_address
from normalization.structured_fields import extract_postal_code, extract_house_number, extract_city
from blocking.candidate_generator import CandidateGenerator
import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def apply_normalization(df: pd.DataFrame) -> pd.DataFrame:
    """Applies all normalization and extraction to a dataframe."""
    if df.empty:
        return df
        
    logger.info(f"Normalizing {len(df)} records...")
    
    # We create a new dataframe to hold the normalized fields to avoid modifying original
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

def main():
    logger.info("Starting Business Entity Resolution Pipeline (Shriyash's part)")
    
    # 1. Load Data
    loader = DataLoader(config.DATA_DIR)
    # Using test sets as example of what would run for inference
    # In practice, you might run train for blocking recall experiments
    datasets = loader.load_all_test() 
    
    s1_df = datasets.get("source1")
    s2_df = datasets.get("source2")
    s3_df = datasets.get("source3")
    
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
    
    # 4. Save candidate pairs
    output_file = config.OUTPUT_DIR / "candidate_pairs.tsv"
    candidates.to_csv(output_file, sep="\t", index=False)
    logger.info(f"Saved {len(candidates)} candidate pairs to {output_file}")
    
    # Note: Feature engineering, model inference, and aggregation are Despo's part.
    # The pipeline would continue here by loading the candidates and generating matching_results.tsv.
    
    logger.info("Shriyash's part of the pipeline completed.")

if __name__ == "__main__":
    main()
