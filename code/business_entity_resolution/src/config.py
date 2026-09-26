from pathlib import Path

# Paths (Assuming execution from code/business_entity_resolution)
BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure output directory exists
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Blocking settings
TFIDF_NGRAM_RANGE = (2, 4)
TFIDF_N_NEIGHBORS = 20

# Model settings
THRESHOLD = 0.5 # To be tuned against F0.5
