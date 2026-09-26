import sys
from pathlib import Path
import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score
from sklearn.model_selection import train_test_split

# Add src to path so we can import config and features
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(PROJECT_ROOT))

from src.config import OUTPUT_DIR
from src.features.feature_engineering import create_features
from src.model.threshold import find_best_threshold

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
    # Separate the text inputs and the target, the rest will be generated via feature engineering
    if 'name_1' not in df.columns or 'name_2' not in df.columns:
        raise ValueError("name_1 and name_2 columns must be present in the candidate pairs!")
        
    X_raw = df[['name_1', 'name_2']]
    y = df[target_column]
    
    print("Extracting numerical features from text pairs...")
    X_features = create_features(X_raw, "name_1", "name_2")
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

def evaluate_model(model, X_test, y_test, threshold=0.5):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)
    
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f05 = fbeta_score(y_test, predictions, beta=0.5, zero_division=0)
    
    return precision, recall, f05, probabilities

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

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")

    print("\nTraining LightGBM model...")
    model = train_model(X_train, y_train)
    print("Training completed.")

    # Get predictions and find best threshold using Despo's threshold.py logic
    probabilities = model.predict_proba(X_test)[:, 1]
    best_threshold, best_f05 = find_best_threshold(y_test, probabilities)
    
    print(f"\nOptimized Threshold : {best_threshold:.2f}")
    
    precision, recall, f05, _ = evaluate_model(model, X_test, y_test, threshold=best_threshold)

    print("\nModel Evaluation (Optimized)")
    print("-----------------")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F0.5      : {f05:.4f}")

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": model,
        "features": list(X.columns),
        "target_column": target_column,
        "best_threshold": best_threshold
    }, MODEL_FILE)

    print(f"\nModel saved to: {MODEL_FILE}")

if __name__ == "__main__":
    main()