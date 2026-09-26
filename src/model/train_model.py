import os
import pickle

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    fbeta_score,
    confusion_matrix,
    classification_report
)

from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)


SOURCE2_FILE = os.path.join(
    OUTPUT_DIR,
    "pseudo_labels_source2_15000.tsv"
)

SOURCE3_FILE = os.path.join(
    OUTPUT_DIR,
    "pseudo_labels_source3_15000.tsv"
)

MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "random_forest_model.pkl"
)


FEATURE_COLUMNS = [
    "name_exact_match",
    "name_similarity",
    "name_length_difference",
    "name_length_ratio",
    "name_token_jaccard",
    "name_token_overlap",
    "address_exact_match",
    "address_similarity",
    "address_length_difference",
    "address_length_ratio",
    "address_token_jaccard",
    "address_token_overlap",
    "country_match"
]


def load_data():

    print("=" * 70)
    print("LOADING PSEUDO-LABELED DATA")
    print("=" * 70)

    source2 = pd.read_csv(
        SOURCE2_FILE,
        sep="\t"
    )

    source3 = pd.read_csv(
        SOURCE3_FILE,
        sep="\t"
    )

    print()
    print(
        f"Source2 rows: {len(source2)}"
    )

    print(
        f"Source3 rows: {len(source3)}"
    )

    df = pd.concat(
        [source2, source3],
        ignore_index=True
    )

    print(
        f"Combined rows: {len(df)}"
    )

    return df


def prepare_data(df):

    print()
    print("=" * 70)
    print("PREPARING FEATURES")
    print("=" * 70)

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:

        print()
        print("ERROR: Missing feature columns:")

        for column in missing_columns:
            print(
                f"  - {column}"
            )

        raise ValueError(
            "Required feature columns are missing."
        )

    X = df[
        FEATURE_COLUMNS
    ].copy()

    y = df[
        "label"
    ].astype(int)

    X = X.fillna(0)

    print()
    print(
        f"Number of features: {X.shape[1]}"
    )

    print(
        f"Number of samples: {len(X)}"
    )

    print()
    print("Class distribution:")

    print(
        f"Non-match (0): {(y == 0).sum()}"
    )

    print(
        f"Match (1): {(y == 1).sum()}"
    )

    return X, y


def train_model(
    X_train,
    y_train
):

    print()
    print("=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    print()
    print("Training complete.")

    return model


def evaluate_model(
    model,
    X_test,
    y_test
):

    print()
    print("=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0
    )

    f05 = fbeta_score(
        y_test,
        predictions,
        beta=0.5,
        zero_division=0
    )

    print()
    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F0.5     : {f05:.4f}"
    )

    print()
    print("Confusion Matrix:")

    matrix = confusion_matrix(
        y_test,
        predictions
    )

    print(matrix)

    print()
    print("Classification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )


def show_feature_importance(
    model
):

    print()
    print("=" * 70)
    print("FEATURE IMPORTANCE")
    print("=" * 70)

    importance = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance":
                model.feature_importances_
        }
    )

    importance = importance.sort_values(
        "importance",
        ascending=False
    )

    for _, row in importance.iterrows():

        print(
            f"{row['feature']:<30} "
            f"{row['importance']:.4f}"
        )


def save_model(
    model
):

    with open(
        MODEL_FILE,
        "wb"
    ) as file:

        pickle.dump(
            model,
            file
        )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        MODEL_FILE
    )


def main():

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION")
    print("RANDOM FOREST MODEL")
    print("=" * 70)

    df = load_data()

    X, y = prepare_data(
        df
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print()
    print("=" * 70)
    print("TRAIN / TEST SPLIT")
    print("=" * 70)

    print(
        f"Training samples: {len(X_train)}"
    )

    print(
        f"Testing samples : {len(X_test)}"
    )

    model = train_model(
        X_train,
        y_train
    )

    evaluate_model(
        model,
        X_test,
        y_test
    )

    show_feature_importance(
        model
    )

    save_model(
        model
    )

    print()
    print("=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

