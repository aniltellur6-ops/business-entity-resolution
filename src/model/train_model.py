from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import precision_score, recall_score, fbeta_score
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "src" / "output" / "candidate_pairs.tsv"
MODEL_FILE = PROJECT_ROOT / "src" / "output" / "entity_resolution_model.pkl"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

TARGET_COLUMN_CANDIDATES = [
    "label",
    "target",
    "is_match",
    "match",
    "matched",
    "y",
]


TEST_SIZE = 0.20
RANDOM_STATE = 42


# ---------------------------------------------------------
# Find target column
# ---------------------------------------------------------

def find_target_column(df):
    """
    Automatically find the target/label column.
    """

    for column in TARGET_COLUMN_CANDIDATES:
        if column in df.columns:
            return column

    raise ValueError(
        "Target column not found.\n"
        f"Available columns: {list(df.columns)}\n"
        f"Expected one of: {TARGET_COLUMN_CANDIDATES}"
    )


# ---------------------------------------------------------
# Prepare features
# ---------------------------------------------------------

def prepare_features(df, target_column):
    """
    Select numeric feature columns for model training.

    The target column is excluded automatically.
    """

    feature_df = df.drop(columns=[target_column])

    numeric_features = feature_df.select_dtypes(
        include=["number", "bool"]
    )

    if numeric_features.empty:
        raise ValueError(
            "No numeric features found for model training."
        )

    return numeric_features


# ---------------------------------------------------------
# Train model
# ---------------------------------------------------------

def train_model(X_train, y_train):
    """
    Train LightGBM binary classification model.
    """

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


# ---------------------------------------------------------
# Evaluate model
# ---------------------------------------------------------

def evaluate_model(model, X_test, y_test, threshold=0.5):
    """
    Evaluate predictions using precision, recall and F0.5.
    """

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f05 = fbeta_score(
        y_test,
        predictions,
        beta=0.5,
        zero_division=0,
    )

    return precision, recall, f05


# ---------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------

def main():

    print("Business Entity Resolution - Model Training")
    print("------------------------------------------------")

    # Check input file
    if not INPUT_FILE.exists():

        print()
        print("candidate_pairs.tsv not found.")
        print()
        print(f"Expected location:")
        print(INPUT_FILE)
        print()
        print(
            "Waiting for candidate_pairs.tsv from the data pipeline."
        )

        return

    # Load dataset
    print("Loading candidate pairs...")

    df = pd.read_csv(
        INPUT_FILE,
        sep="\t",
    )

    print(f"Dataset shape: {df.shape}")

    print()
    print("Columns:")
    print(list(df.columns))

    # Find target
    target_column = find_target_column(df)

    print()
    print(f"Target column: {target_column}")

    # Separate target
    y = df[target_column]

    # Prepare features
    X = prepare_features(
        df,
        target_column,
    )

    print()
    print("Features used:")
    print(list(X.columns))

    print()
    print(f"Number of features: {X.shape[1]}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print()
    print(f"Training rows: {len(X_train)}")
    print(f"Testing rows : {len(X_test)}")

    # Train
    print()
    print("Training LightGBM model...")

    model = train_model(
        X_train,
        y_train,
    )

    print("Training completed.")

    # Evaluate
    precision, recall, f05 = evaluate_model(
        model,
        X_test,
        y_test,
        threshold=0.5,
    )

    print()
    print("Model Evaluation")
    print("-----------------")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F0.5      : {f05:.4f}")

    # Save model
    MODEL_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        {
            "model": model,
            "features": list(X.columns),
            "target_column": target_column,
        },
        MODEL_FILE,
    )

    print()
    print(f"Model saved to:")
    print(MODEL_FILE)


# ---------------------------------------------------------
# Run
# ---------------------------------------------------------

if __name__ == "__main__":
    main()