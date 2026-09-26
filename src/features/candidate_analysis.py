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

FILES = {
    "source2": os.path.join(
        OUTPUT_DIR,
        "candidate_pairs_source2_15000.tsv"
    ),
    "source3": os.path.join(
        OUTPUT_DIR,
        "candidate_pairs_source3_15000.tsv"
    )
}


def analyze_file(
    target_source,
    filepath
):

    print()
    print("=" * 70)
    print(
        "CANDIDATE ANALYSIS:",
        target_source
    )
    print("=" * 70)

    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print()
    print(
        "Candidate rows:",
        len(df)
    )

    if len(df) == 0:
        print("No candidates found.")
        return

    unique_source1 = (
        df["source1_entity_id"]
        .nunique()
    )

    print(
        "Unique Source1 entities:",
        unique_source1
    )

    candidate_counts = (
        df.groupby(
            "source1_entity_id"
        )
        .size()
    )

    print()
    print("Candidates per Source1 entity:")

    print(
        "Minimum:",
        candidate_counts.min()
    )

    print(
        "Maximum:",
        candidate_counts.max()
    )

    print(
        "Mean:",
        round(
            candidate_counts.mean(),
            3
        )
    )

    print(
        "Median:",
        candidate_counts.median()
    )

    print()
    print("Candidate count distribution:")

    print(
        candidate_counts
        .value_counts()
        .sort_index()
        .head(25)
        .to_string()
    )

    print()
    print(
        "Entities with exactly 1 candidate:",
        (
            candidate_counts == 1
        ).sum()
    )

    print(
        "Entities with exactly 20 candidates:",
        (
            candidate_counts == 20
        ).sum()
    )

    print(
        "Entities with more than 10 candidates:",
        (
            candidate_counts > 10
        ).sum()
    )

    country_matches = (
        df["source1_country_normalized"]
        ==
        df["target_country_normalized"]
    )

    print()
    print(
        "Country match rate:",
        round(
            country_matches.mean() * 100,
            2
        ),
        "%"
    )

    source1_missing_names = (
        df[
            "source1_business_name_normalized"
        ]
        .eq("")
        .sum()
    )

    target_missing_names = (
        df[
            "target_business_name_normalized"
        ]
        .eq("")
        .sum()
    )

    print()
    print(
        "Source1 missing normalized names:",
        source1_missing_names
    )

    print(
        "Target missing normalized names:",
        target_missing_names
    )


def main():

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION")
    print("15K CANDIDATE QUALITY ANALYSIS")
    print("=" * 70)

    for target_source, filepath in FILES.items():

        analyze_file(
            target_source,
            filepath
        )

    print()
    print("=" * 70)
    print("CANDIDATE ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()