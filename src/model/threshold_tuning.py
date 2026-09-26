import os
import pickle

import pandas as pd

from sklearn.metrics import (
    precision_score,
    recall_score,
    fbeta_score
)


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


MODEL_FILE = os.path.join(
    OUTPUT_DIR,
    "random_forest_model.pkl"
)

SOURCE2_FILE = os.path.join(
    OUTPUT_DIR,
    "pseudo_labels_source2_15000.tsv"
)

SOURCE3_FILE = os.path.join(
    OUTPUT_DIR,
    "pseudo_labels_source3_15000.tsv"
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


def load_model():

    print("=" * 70)
    print("LOADING RANDOM FOREST MODEL")
    print("=" * 70)

    with open(
        MODEL_FILE,
        "rb"
    ) as file:

        model = pickle.load(
            file
        )

    print()
    print("Model loaded successfully.")

    return model


def load_data():

    print()
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

    df = pd.concat(
        [source2, source3],
        ignore_index=True
    )

    X = df[
        FEATURE_COLUMNS
    ].copy()

    X = X.fillna(0)

    y = df[
        "label"
    ].astype(int)

    print()
    print(
        f"Total samples: {len(df)}"
    )

    print(
        f"Non-match: {(y == 0).sum()}"
    )

    print(
        f"Match: {(y == 1).sum()}"
    )

    return X, y


def tune_threshold(
    model,
    X,
    y
):

    print()
    print("=" * 70)
    print("F0.5 THRESHOLD TUNING")
    print("=" * 70)

    probabilities = model.predict_proba(
        X
    )[:, 1]

    results = []

    thresholds = [
        value / 100
        for value in range(
            10,
            100,
            5
        )
    ]

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y,
            predictions,
            zero_division=0
        )

        recall = recall_score(
            y,
            predictions,
            zero_division=0
        )

        f05 = fbeta_score(
            y,
            predictions,
            beta=0.5,
            zero_division=0
        )

        results.append(
            {
                "threshold": threshold,
                "precision": precision,
                "recall": recall,
                "f0.5": f05
            }
        )

    results_df = pd.DataFrame(
        results
    )

    results_df = results_df.sort_values(
        "f0.5",
        ascending=False
    )

    print()
    print(
        f"{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F0.5':<12}"
    )

    print("-" * 48)

    for _, row in results_df.iterrows():

        print(
            f"{row['threshold']:<12.2f}"
            f"{row['precision']:<12.4f}"
            f"{row['recall']:<12.4f}"
            f"{row['f0.5']:<12.4f}"
        )

    best_row = results_df.iloc[0]

    print()
    print("=" * 70)
    print("BEST THRESHOLD ON PSEUDO-LABELS")
    print("=" * 70)

    print(
        f"Threshold : {best_row['threshold']:.2f}"
    )

    print(
        f"Precision : {best_row['precision']:.4f}"
    )

    print(
        f"Recall    : {best_row['recall']:.4f}"
    )

    print(
        f"F0.5      : {best_row['f0.5']:.4f}"
    )

    result_file = os.path.join(
        OUTPUT_DIR,
        "threshold_results.tsv"
    )

    results_df.to_csv(
        result_file,
        sep="\t",
        index=False
    )

    print()
    print("Threshold results saved:")
    print(result_file)


def main():

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION")
    print("THRESHOLD TUNING")
    print("=" * 70)

    model = load_model()

    X, y = load_data()

    tune_threshold(
        model,
        X,
        y
    )

    print()
    print("=" * 70)
    print("THRESHOLD TUNING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()

