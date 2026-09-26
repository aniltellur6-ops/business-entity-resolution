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
    "quality_controlled_matches_15000.tsv"
)

SOURCE2_FEATURES = os.path.join(
    OUTPUT_DIR,
    "features_source2_15000.tsv"
)

SOURCE3_FEATURES = os.path.join(
    OUTPUT_DIR,
    "features_source3_15000.tsv"
)

OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "validated_matches_15000.tsv"
)


def load_file(file_path):

    return pd.read_csv(
        file_path,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )


def convert_numeric_columns(df):

    numeric_columns = [
        "match_probability",
        "name_similarity",
        "name_token_jaccard",
        "name_token_overlap",
        "address_similarity",
        "address_token_jaccard",
        "address_token_overlap"
    ]

    for column in numeric_columns:

        if column in df.columns:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    return df


def main():

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION")
    print("MATCH VALIDATION")
    print("=" * 70)

    print()
    print("Loading quality-controlled matches...")

    matches = load_file(
        INPUT_FILE
    )

    print(
        f"Quality-controlled rows: "
        f"{len(matches)}"
    )

    print()
    print("Loading feature tables...")

    source2 = load_file(
        SOURCE2_FEATURES
    )

    source3 = load_file(
        SOURCE3_FEATURES
    )

    source2["target_source"] = "source2"
    source3["target_source"] = "source3"

    features = pd.concat(
        [
            source2,
            source3
        ],
        ignore_index=True
    )

    print(
        f"Feature rows loaded: "
        f"{len(features)}"
    )

    feature_columns = [
        "source1_entity_id",
        "target_source",
        "target_entity_id",
        "name_similarity",
        "name_token_jaccard",
        "name_token_overlap",
        "address_similarity",
        "address_token_jaccard",
        "address_token_overlap",
        "country_match"
    ]

    features = features[
        feature_columns
    ]

    matches = matches.merge(
        features,
        on=[
            "source1_entity_id",
            "target_source",
            "target_entity_id"
        ],
        how="left"
    )

    matches = convert_numeric_columns(
        matches
    )

    print()
    print("Applying validation rules...")

    def validate_row(row):

        probability = row[
            "match_probability"
        ]

        name_similarity = row[
            "name_similarity"
        ]

        name_jaccard = row[
            "name_token_jaccard"
        ]

        address_similarity = row[
            "address_similarity"
        ]

        address_jaccard = row[
            "address_token_jaccard"
        ]

        country_match = row[
            "country_match"
        ]

        if probability >= 0.90:

            if (
                name_similarity >= 0.90
                and (
                    address_similarity >= 0.50
                    or name_jaccard >= 0.80
                )
            ):

                return "VALIDATED_HIGH"

            if (
                name_similarity >= 0.95
                and name_jaccard >= 0.80
            ):

                return "VALIDATED_HIGH"

            return "HIGH_REVIEW"

        if probability >= 0.70:

            if (
                name_similarity >= 0.85
                and (
                    address_similarity >= 0.40
                    or name_jaccard >= 0.70
                )
            ):

                return "VALIDATED_MEDIUM"

            return "MEDIUM_REVIEW"

        return "REVIEW"

    matches["validation_status"] = matches.apply(
        validate_row,
        axis=1
    )

    matches.to_csv(
        OUTPUT_FILE,
        sep="\t",
        index=False
    )

    print()
    print("=" * 70)
    print("VALIDATION COMPLETE")
    print("=" * 70)

    print()
    print("Validation distribution:")

    print(
        matches[
            "validation_status"
        ].value_counts()
    )

    print()
    print("Validated HIGH examples:")

    validated_high = matches[
        matches[
            "validation_status"
        ] == "VALIDATED_HIGH"
    ]

    display_columns = [
        "source1_entity_id",
        "target_source",
        "target_entity_id",
        "source1_business_name",
        "target_business_name",
        "match_probability",
        "name_similarity",
        "name_token_jaccard",
        "address_similarity",
        "country_match",
        "validation_status"
    ]

    if len(validated_high) > 0:

        print(
            validated_high[
                display_columns
            ].head(20).to_string(
                index=False
            )
        )

    else:

        print(
            "No validated HIGH matches found."
        )

    print()
    print("Potential review cases:")

    review = matches[
        matches[
            "validation_status"
        ].isin(
            [
                "HIGH_REVIEW",
                "MEDIUM_REVIEW"
            ]
        )
    ]

    if len(review) > 0:

        print(
            review[
                display_columns
            ].head(20).to_string(
                index=False
            )
        )

    else:

        print(
            "No additional review cases found."
        )

    print()
    print("Output file:")

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()