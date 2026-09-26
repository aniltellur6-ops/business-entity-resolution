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

FINAL_FILE = os.path.join(
    OUTPUT_DIR,
    "final_submission_15000.tsv"
)

REVIEW_FILE = os.path.join(
    OUTPUT_DIR,
    "review_matches_15000.tsv"
)


def load_file(file_path):

    if not os.path.exists(file_path):

        print(
            f"File not found: {file_path}"
        )

        return None

    return pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )


def main():

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION")
    print("FINAL MATCH REPORT")
    print("=" * 70)

    print()
    print("Loading final submission file...")

    final_df = load_file(
        FINAL_FILE
    )

    if final_df is None:
        return

    print(
        f"Final rows: {len(final_df)}"
    )

    print()
    print("=" * 70)
    print("FINAL OUTPUT SUMMARY")
    print("=" * 70)

    print(
        f"Unique Source1 entities: "
        f"{final_df['source1_entity_id'].nunique()}"
    )

    source2_count = (
        final_df["target_source"] == "source2"
    ).sum()

    source3_count = (
        final_df["target_source"] == "source3"
    ).sum()

    print(
        f"Source2 matches: {source2_count}"
    )

    print(
        f"Source3 matches: {source3_count}"
    )

    print()
    print("Target source distribution:")

    print(
        final_df[
            "target_source"
        ].value_counts()
    )

    if "validation_status" in final_df.columns:

        print()
        print("Validation status:")

        print(
            final_df[
                "validation_status"
            ].value_counts()
        )

    final_df[
        "match_probability"
    ] = pd.to_numeric(
        final_df[
            "match_probability"
        ],
        errors="coerce"
    )

    print()
    print("Match probability statistics:")

    print(
        final_df[
            "match_probability"
        ].describe()
    )

    high = final_df[
        final_df[
            "match_probability"
        ] >= 0.90
    ]

    medium = final_df[
        (
            final_df[
                "match_probability"
            ] >= 0.70
        )
        &
        (
            final_df[
                "match_probability"
            ] < 0.90
        )
    ]

    print()
    print(
        f"HIGH probability matches: "
        f"{len(high)}"
    )

    print(
        f"MEDIUM probability matches: "
        f"{len(medium)}"
    )

    print()
    print("=" * 70)
    print("SAMPLE HIGH MATCHES")
    print("=" * 70)

    display_columns = [
        "source1_entity_id",
        "target_source",
        "target_entity_id",
        "source1_business_name",
        "target_business_name",
        "match_probability"
    ]

    if len(high) > 0:

        print(
            high[
                display_columns
            ].head(15).to_string(
                index=False
            )
        )

    else:

        print(
            "No HIGH matches found."
        )

    print()
    print("=" * 70)
    print("SAMPLE MEDIUM MATCHES")
    print("=" * 70)

    if len(medium) > 0:

        print(
            medium[
                display_columns
            ].head(15).to_string(
                index=False
            )
        )

    else:

        print(
            "No MEDIUM matches found."
        )

    print()
    print("=" * 70)
    print("REVIEW FILE")
    print("=" * 70)

    review_df = load_file(
        REVIEW_FILE
    )

    if review_df is not None:

        print(
            f"Rows requiring review: "
            f"{len(review_df)}"
        )

        if (
            len(review_df) > 0
            and "validation_status"
            in review_df.columns
        ):

            print()
            print(
                review_df[
                    "validation_status"
                ].value_counts()
            )

    print()
    print("=" * 70)
    print("FINAL REPORT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
