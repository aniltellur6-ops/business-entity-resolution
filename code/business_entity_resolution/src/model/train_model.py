import sys
from pathlib import Path
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

# Add src to path so we can import config and features
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import OUTPUT_DIR
from src.features.feature_engineering import create_features
from src.model.threshold import find_best_threshold
from src.evaluation.f05_evaluator import evaluate_predictions

INPUT_FILE = OUTPUT_DIR / "candidate_pairs.tsv"
MODEL_FILE = OUTPUT_DIR / "entity_resolution_model.pkl"

TARGET_COLUMN_CANDIDATES = ["label", "target", "is_match", "match", "matched", "y"]
TEST_SIZE = 0.20
RANDOM_STATE = 42

def find_target_column(df):
    for column in TARGET_COLUMN_CANDIDATES:
        if column in df.columns:
            return column
    raise ValueError(f"Target column not found. Available columns: {list(df.columns)}")

def prepare_features(df, target_column):
    # Separate the target, the rest will be generated via feature engineering
    y = df[target_column]
    
    print("Extracting numerical features from text pairs...")
    X_features = create_features(df, "_1", "_2")
    
    # Include provenance flags if they exist
    provenance_cols = ['found_by_tfidf', 'found_by_token', 'found_by_postal']
    for col in provenance_cols:
        if col in df.columns:
            X_features[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            
    return X_features, y

def train_model(X_train, y_train):
    model = LGBMClassifier(
        objective="binary",
        n_estimators=300,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        random_state=RANDOM_STATE,
        class_weight="balanced",
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model

def main():
    print("Business Entity Resolution - Model Training")
    print("------------------------------------------------")

    if not INPUT_FILE.exists():
        print(f"\ncandidate_pairs.tsv not found at {INPUT_FILE}")
        return

    print("Loading candidate pairs...")
    df = pd.read_csv(INPUT_FILE, sep="\t", dtype=str)
    
    # Convert target to numeric
    target_column = find_target_column(df)
    df[target_column] = pd.to_numeric(df[target_column], errors='coerce').fillna(0).astype(int)
    
    print(f"Target column: {target_column}")

    # Prepare features
    X, y = prepare_features(df, target_column)
    print(f"\nNumber of features: {X.shape[1]}")
    
    # Keep s1_id for evaluation
    X['s1_id'] = df['s1_id']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    s1_ids_test = X_test['s1_id'].values
    
    X_train = X_train.drop(columns=['s1_id'])
    X_test = X_test.drop(columns=['s1_id'])

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")

    print("\nTraining LightGBM model...")
    model = train_model(X_train, y_train)
    print("Training completed.")

    # Get predictions and find best threshold using Despo's threshold.py logic
    probabilities = model.predict_proba(X_test)[:, 1]
    
    print("\nOptimizing Threshold against per-entity Macro F0.5...")
    best_threshold, best_f05 = find_best_threshold(y_test.values, probabilities, s1_ids_test)
    print(f"Optimized Threshold : {best_threshold:.2f}")
    
    predictions = (probabilities >= best_threshold).astype(int)
    eval_results = evaluate_predictions(y_test.values, predictions, s1_ids_test)

    print("\nModel Evaluation (Optimized)")
    print("-----------------")
    print(f"Precision : {eval_results['precision']:.4f}")
    print(f"Recall    : {eval_results['recall']:.4f}")
    print(f"Macro F0.5: {eval_results['f0.5']:.4f}")

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": list(X_train.columns),
        "target_column": target_column,
        "best_threshold": best_threshold
    }, MODEL_FILE)

    print(f"\nModel saved to: {MODEL_FILE}")

if __name__ == "__main__":
    main()