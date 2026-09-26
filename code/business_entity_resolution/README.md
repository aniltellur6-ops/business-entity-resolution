# Business Entity Resolution Pipeline

This repository contains the solution for the Business Entity Resolution project.

## Overview
The system takes a clean list of businesses (Source 1) and intelligently finds which noisy records across two other datasets (Source 2, Source 3) represent the same real-world businesses, while minimizing false merges and remaining robust to unseen countries and formatting patterns.

## Reproduction Steps

1. **Install requirements:**
   ```bash
   pip install -r code/business_entity_resolution/requirements.txt
   ```

2. **Prepare data:**
   Place the provided dataset files in the `data/` directory at the root of the repository.

3. **Run pipeline:**
   Execute the main pipeline script.
   ```bash
   python code/business_entity_resolution/src/pipeline.py
   ```
   This will:
   - Load and normalize the data
   - Generate candidates
   - Run feature engineering
   - Execute model inference
   - Generate `candidate_pairs.tsv` and `matching_results.tsv` in the `output/` directory.

4. **Run validator:**
   ```bash
   python utils/validate_submission.py
   ```
