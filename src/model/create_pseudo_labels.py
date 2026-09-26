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


INPUT_FILES = [
    os.path.join(
        OUTPUT_DIR,
        "features_source2_15000.tsv"
    ),
    os.path.join(
        OUTPUT_DIR,
        "features_source3_15000.tsv"
    )
]

OUTPUT_FILES = [
    os.path.join(
        OUTPUT_DIR,
        "pseudo_labels_source2_15000.tsv"
    ),
    os.path.join(
        OUTPUT_DIR,
        "pseudo_labels_source3_15000.tsv"
    )
]


def create_pseudo_labels(input_file, output_file):

    print()
    print("=" * 70)
    print("PROCESSING")
    print(input_file)
    print("=" * 70)

    df = pd.read_csv(
        input_file,
        sep="\t"
    )

    print(
        f"Candidate rows: {len(df)}"
    )

    required_columns = [
        "name_similarity",
        "name_token_jaccard",
        "address_similarity",
        "address_token_jaccard",
        "country_match"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        print()
        print("ERROR: Missing columns:")

        for column in missing_columns:
            print(column)

        return

    df["label"] = -1

    positive_condition = (
        (df["country_match"] == 1)
        &
        (
            (
                (df["name_similarity"] >= 0.95)
                &
                (df["address_similarity"] >= 0.80)
            )
            |
            (
                (df["name_similarity"] >= 0.98)
                &
                (df["name_token_jaccard"] >= 0.90)
            )
            |
            (
                (df["name_similarity"] >= 0.95)
                &
                (df["address_token_jaccard"] >= 0.80)
            )
        )
    )

    negative_condition = (
        (
            (df["name_similarity"] < 0.45)
            &
            (df["address_similarity"] < 0.30)
        )
        |
        (
            (df["name_similarity"] < 0.35)
        )
    )

    df.loc[
        positive_condition,
        "label"
    ] = 1

    df.loc[
        negative_condition,
        "label"
    ] = 0

    labelled_df = df[
        df["label"] != -1
    ].copy()

    print()
    print("Pseudo-label results:")
    print(
        f"Positive matches: "
        f"{(labelled_df['label'] == 1).sum()}"
    )
    print(
        f"Negative matches: "
        f"{(labelled_df['label'] == 0).sum()}"
    )
    print(
        f"Unlabelled: "
        f"{(df['label'] == -1).sum()}"
    )

    print()
    print("Keeping only high-confidence pseudo-labels.")

    labelled_df.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    print()
    print("Saved:")
    print(output_file)

    print(
        f"Final labelled rows: "
        f"{len(labelled_df)}"
    )


def main():

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION")
    print("PSEUDO-LABEL GENERATION")
    print("=" * 70)

    for input_file, output_file in zip(
        INPUT_FILES,
        OUTPUT_FILES
    ):

        create_pseudo_labels(
            input_file,
            output_file
        )

    print()
    print("=" * 70)
    print("PSEUDO-LABEL GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()