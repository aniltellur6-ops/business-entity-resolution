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

INPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "final_matches_15000.tsv"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "quality_controlled_matches_15000.tsv"
)


def load_final_matches():

    print("Loading final matches...")

    df = pd.read_csv(
        INPUT_FILE,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print(
        f"Rows loaded: {len(df)}"
    )

    return df


def assign_quality(probability):

    if probability >= 0.90:
        return "HIGH"

    if probability >= 0.70:
        return "MEDIUM"

    return "REVIEW"


def calculate_quality(df):

    df["match_probability"] = pd.to_numeric(
        df["match_probability"],
        errors="coerce"
    )

    df["quality"] = df[
        "match_probability"
    ].apply(
        assign_quality
    )

    return df


def main():

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION")
    print("QUALITY CONTROL")
    print("=" * 70)

    df = load_final_matches()

    print()
    print("Assigning match quality...")

    df = calculate_quality(
        df
    )

    df = df.sort_values(
        [
            "quality",
            "match_probability"
        ],
        ascending=[
            True,
            False
        ]
    )

    df.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False
    )

    high_count = (
        df["quality"] == "HIGH"
    ).sum()

    medium_count = (
        df["quality"] == "MEDIUM"
    ).sum()

    review_count = (
        df["quality"] == "REVIEW"
    ).sum()

    source2_count = (
        df["target_source"] == "source2"
    ).sum()

    source3_count = (
        df["target_source"] == "source3"
    ).sum()

    print()
    print("=" * 70)
    print("QUALITY CONTROL COMPLETE")
    print("=" * 70)

    print()
    print("Total candidate matches:")
    print(len(df))

    print()
    print("Quality distribution:")

    print(
        f"HIGH    : {high_count}"
    )

    print(
        f"MEDIUM  : {medium_count}"
    )

    print(
        f"REVIEW  : {review_count}"
    )

    print()
    print("Target source distribution:")

    print(
        f"Source2 : {source2_count}"
    )

    print(
        f"Source3 : {source3_count}"
    )

    print()
    print("HIGH confidence matches:")

    high_matches = df[
        df["quality"] == "HIGH"
    ]

    high_columns = [
        "source1_entity_id",
        "target_source",
        "target_entity_id",
        "source1_business_name",
        "target_business_name",
        "match_probability",
        "quality"
    ]

    if len(high_matches) > 0:

        print(
            high_matches[
                high_columns
            ].head(20).to_string(
                index=False
            )
        )

    else:

        print(
            "No HIGH confidence matches found."
        )

    print()
    print("MEDIUM confidence matches:")

    medium_matches = df[
        df["quality"] == "MEDIUM"
    ]

    if len(medium_matches) > 0:

        print(
            medium_matches[
                high_columns
            ].head(20).to_string(
                index=False
            )
        )

    else:

        print(
            "No MEDIUM confidence matches found."
        )

    print()
    print("REVIEW examples:")

    review_matches = df[
        df["quality"] == "REVIEW"
    ]

    if len(review_matches) > 0:

        print(
            review_matches[
                high_columns
            ].head(20).to_string(
                index=False
            )
        )

    else:

        print(
            "No REVIEW matches found."
        )

    print()
    print("Output file:")

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()
