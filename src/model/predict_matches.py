import os
import pickle

import pandas as pd


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
    "features_source2_15000.tsv"
)

SOURCE3_FILE = os.path.join(
    OUTPUT_DIR,
    "features_source3_15000.tsv"
)

SOURCE2_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "predictions_source2_15000.tsv"
)

SOURCE3_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "predictions_source3_15000.tsv"
)


THRESHOLD = 0.25


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


def predict_file(
    model,
    input_file,
    output_file,
    source_name
):

    print()
    print("=" * 70)
    print(
        f"GENERATING PREDICTIONS: {source_name}"
    )
    print("=" * 70)

    df = pd.read_csv(
        input_file,
        sep="\t"
    )

    print()
    print(
        f"Candidate rows: {len(df)}"
    )

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

    X = X.fillna(0)

    probabilities = model.predict_proba(
        X
    )[:, 1]

    predictions = (
        probabilities >= THRESHOLD
    ).astype(int)

    df["match_probability"] = probabilities

    df["predicted_match"] = predictions

    df["prediction"] = df[
        "predicted_match"
    ].map(
        {
            1: "MATCH",
            0: "NON_MATCH"
        }
    )

    df = df.sort_values(
        "match_probability",
        ascending=False
    )

    df.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    match_count = (
        predictions == 1
    ).sum()

    non_match_count = (
        predictions == 0
    ).sum()

    print()
    print(
        f"Threshold: {THRESHOLD}"
    )

    print(
        f"Predicted MATCH: {match_count}"
    )

    print(
        f"Predicted NON_MATCH: {non_match_count}"
    )

    print()
    print("Top 10 predictions:")

    display_columns = [
        "source1_entity_id",
        "target_source",
        "target_entity_id",
        "source1_business_name",
        "target_business_name",
        "match_probability",
        "prediction"
    ]

    available_columns = [
        column
        for column in display_columns
        if column in df.columns
    ]

    print(
        df[
            available_columns
        ].head(10).to_string(
            index=False
        )
    )

    print()
    print("Saved:")
    print(output_file)


def main():

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION")
    print("FINAL MATCH PREDICTIONS")
    print("=" * 70)

    model = load_model()

    predict_file(
        model,
        SOURCE2_FILE,
        SOURCE2_OUTPUT,
        "SOURCE1 -> SOURCE2"
    )

    predict_file(
        model,
        SOURCE3_FILE,
        SOURCE3_OUTPUT,
        "SOURCE1 -> SOURCE3"
    )

    print()
    print("=" * 70)
    print("PREDICTION COMPLETE")
    print("=" * 70)

    print()
    print("Output files:")

    print(
        SOURCE2_OUTPUT
    )

    print(
        SOURCE3_OUTPUT
    )


if __name__ == "__main__":
    main()
