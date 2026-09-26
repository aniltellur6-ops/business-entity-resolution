# Business Entity Resolution Pipeline (Shriyash's Part)

This repository contains Shriyash's part of the solution for the Business Entity Resolution project, which focuses on Data Exploration, Normalization, Blocking, and Candidate Generation.

## Overview
The system takes a clean list of businesses (Source 1) and intelligently finds which noisy records across two other datasets (Source 2, Source 3) represent the same real-world businesses. This part of the pipeline handles parsing the data, normalizing it to be country-agnostic, and generating a high-recall candidate set.

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
   - Generate candidate pairs using multiple blocking strategies (TF-IDF, Token, Postal)
   - Generate `candidate_pairs.tsv` in the `output/` directory.

   *(Note: Feature engineering, model inference, and `matching_results.tsv` generation will be handled downstream in the model pipeline)*
