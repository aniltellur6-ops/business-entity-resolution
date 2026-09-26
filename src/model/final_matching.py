import os

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

SOURCE2_FILE = os.path.join(
    OUTPUT_DIR,
    "predictions_source2_15000.tsv"
)

SOURCE3_FILE = os.path.join(
    OUTPUT_DIR,
    "predictions_source3_15000.tsv"
)

FINAL_FILE = os.path.join(
    OUTPUT_DIR,
    "final_matches_15000.tsv"
)


def load_predictions(file_path):

    return pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )


def prepare_predictions(df):

    df["match_probability"] = pd.to_numeric(
        df["match_probability"],
        errors="coerce"
    )

    df["prediction"] = df[
        "prediction"
    ].astype(str)

    return df


def select_best_matches(df):

    matches = df[
        df["prediction"] == "MATCH"
    ].copy()

    if matches.empty:
        return matches

    matches = matches.sort_values(
        [
            "source1_entity_id",
            "match_probability"
        ],
        ascending=[
            True,
            False
        ]
    )

    best = matches.drop_duplicates(
        subset=[
            "source1_entity_id"
        ],
        keep="first"
    )

    best["rank"] = 1

    return best


def add_confidence(df):

    def confidence(probability):

        if probability >= 0.90:
            return "HIGH"

        if probability >= 0.70:
            return "MEDIUM"

        return "LOW"

    df["confidence"] = df[
        "match_probability"
    ].apply(confidence)

    return df


def main():

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION")
    print("FINAL BEST MATCH SELECTION")
    print("=" * 70)

    print()
    print("Loading prediction files...")

    source2 = load_predictions(
        SOURCE2_FILE
    )

    source3 = load_predictions(
        SOURCE3_FILE
    )

    print(
        f"Source2 prediction rows: "
        f"{len(source2)}"
    )

    print(
        f"Source3 prediction rows: "
        f"{len(source3)}"
    )

    source2 = prepare_predictions(
        source2
    )

    source3 = prepare_predictions(
        source3
    )

    print()
    print("Selecting best Source1 -> Source2 matches...")

    best_source2 = select_best_matches(
        source2
    )

    print(
        f"Best Source2 matches: "
        f"{len(best_source2)}"
    )

    print()
    print("Selecting best Source1 -> Source3 matches...")

    best_source3 = select_best_matches(
        source3
    )

    print(
        f"Best Source3 matches: "
        f"{len(best_source3)}"
    )

    best_source2 = add_confidence(
        best_source2
    )

    best_source3 = add_confidence(
        best_source3
    )

    columns = [
        "source1_entity_id",
        "target_source",
        "target_entity_id",
        "source1_business_name",
        "target_business_name",
        "source1_business_address",
        "target_business_address",
        "source1_country",
        "target_country",
        "match_probability",
        "prediction",
        "confidence",
        "rank"
    ]

    best_source2 = best_source2[
        columns
    ]

    best_source3 = best_source3[
        columns
    ]

    final_matches = pd.concat(
        [
            best_source2,
            best_source3
        ],
        ignore_index=True
    )

    final_matches = final_matches.sort_values(
        [
            "source1_entity_id",
            "target_source"
        ]
    )

    final_matches.to_csv(
        FINAL_FILE,
        sep="\t",
        index=False
    )

    print()
    print("=" * 70)
    print("FINAL MATCHING COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Source2 best matches: "
        f"{len(best_source2)}"
    )

    print(
        f"Source3 best matches: "
        f"{len(best_source3)}"
    )

    print(
        f"Total final matches: "
        f"{len(final_matches)}"
    )

    print()
    print("Confidence distribution:")

    print(
        final_matches[
            "confidence"
        ].value_counts()
    )

    print()
    print("Top final matches:")

    display_columns = [
        "source1_entity_id",
        "target_source",
        "target_entity_id",
        "source1_business_name",
        "target_business_name",
        "match_probability",
        "confidence"
    ]

    print(
        final_matches[
            display_columns
        ].head(20).to_string(
            index=False
        )
    )

    print()
    print("Output file:")

    print(FINAL_FILE)


if __name__ == "__main__":
    main()
