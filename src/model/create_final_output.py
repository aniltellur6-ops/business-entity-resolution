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
    "validated_matches_15000.tsv"
)

ACCEPTED_FILE = os.path.join(
    OUTPUT_DIR,
    "accepted_matches_15000.tsv"
)

REVIEW_FILE = os.path.join(
    OUTPUT_DIR,
    "review_matches_15000.tsv"
)

FINAL_FILE = os.path.join(
    OUTPUT_DIR,
    "final_submission_15000.tsv"
)


def load_matches():

    print("Loading validated matches...")

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


def main():

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION")
    print("FINAL OUTPUT GENERATION")
    print("=" * 70)

    matches = load_matches()

    accepted_statuses = [
        "VALIDATED_HIGH",
        "VALIDATED_MEDIUM"
    ]

    review_statuses = [
        "HIGH_REVIEW",
        "MEDIUM_REVIEW",
        "REVIEW"
    ]

    accepted = matches[
        matches[
            "validation_status"
        ].isin(
            accepted_statuses
        )
    ].copy()

    review = matches[
        matches[
            "validation_status"
        ].isin(
            review_statuses
        )
    ].copy()

    accepted = accepted.sort_values(
        [
            "source1_entity_id",
            "match_probability"
        ],
        ascending=[
            True,
            False
        ]
    )

    review = review.sort_values(
        [
            "source1_entity_id",
            "match_probability"
        ],
        ascending=[
            True,
            False
        ]
    )

    final_columns = [
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
        "quality",
        "validation_status"
    ]

    accepted = accepted[
        [
            column
            for column in final_columns
            if column in accepted.columns
        ]
    ]

    review = review[
        [
            column
            for column in final_columns
            if column in review.columns
        ]
    ]

    accepted.to_csv(
        ACCEPTED_FILE,
        sep="\t",
        index=False
    )

    review.to_csv(
        REVIEW_FILE,
        sep="\t",
        index=False
    )

    final_output = accepted.copy()

    final_output.to_csv(
        FINAL_FILE,
        sep="\t",
        index=False
    )

    print()
    print("=" * 70)
    print("FINAL OUTPUT COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Total validated rows : "
        f"{len(matches)}"
    )

    print(
        f"Accepted matches     : "
        f"{len(accepted)}"
    )

    print(
        f"Rows requiring review: "
        f"{len(review)}"
    )

    print()
    print("Accepted match distribution:")

    if len(accepted) > 0:

        print(
            accepted[
                "validation_status"
            ].value_counts()
        )

    else:

        print(
            "No accepted matches."
        )

    print()
    print("Accepted Source2 matches:")

    if len(accepted) > 0:

        print(
            (
                accepted[
                    "target_source"
                ] == "source2"
            ).sum()
        )

    else:

        print(0)

    print()
    print("Accepted Source3 matches:")

    if len(accepted) > 0:

        print(
            (
                accepted[
                    "target_source"
                ] == "source3"
            ).sum()
        )

    else:

        print(0)

    print()
    print("Sample accepted matches:")

    if len(accepted) > 0:

        display_columns = [
            "source1_entity_id",
            "target_source",
            "target_entity_id",
            "source1_business_name",
            "target_business_name",
            "match_probability",
            "quality",
            "validation_status"
        ]

        print(
            accepted[
                display_columns
            ].head(20).to_string(
                index=False
            )
        )

    else:

        print(
            "No accepted matches to display."
        )

    print()
    print("Files created:")

    print(ACCEPTED_FILE)
    print(REVIEW_FILE)
    print(FINAL_FILE)


if __name__ == "__main__":
    main()
