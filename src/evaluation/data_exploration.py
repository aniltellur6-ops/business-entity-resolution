import os
import pandas as pd


BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data"
)

FILES = [
    "test_source1_15000.tsv",
    "test_source2_15000.tsv",
    "test_source3_15000.tsv"
]


def explore_file(filename):

    filepath = os.path.join(
        DATA_DIR,
        filename
    )

    print()
    print("=" * 70)
    print("FILE:", filename)
    print("=" * 70)

    df = pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print()
    print("Rows:", len(df))
    print("Columns:", len(df.columns))

    print()
    print("Column names:")
    for column in df.columns:
        print("  -", column)

    print()
    print("Missing values:")

    missing = df.isna().sum()

    for column in df.columns:
        empty_values = (
            df[column]
            .astype(str)
            .str.strip()
            .eq("")
            .sum()
        )

        print(
            f"  {column}: "
            f"{empty_values}"
        )

    print()
    print("Duplicate complete rows:")
    print(df.duplicated().sum())

    print()
    print("Unique entity IDs:")
    print(
        df["entity_id"].nunique()
    )

    print()
    print("Unique business names:")
    print(
        df["business_name"].nunique()
    )

    print()
    print("Unique business addresses:")
    print(
        df["business_address"].nunique()
    )

    print()
    print("Country distribution:")

    print(
        df["country"]
        .value_counts()
        .to_string()
    )

    print()
    print("First 5 records:")

    print(
        df.head(5).to_string(
            index=False
        )
    )


def main():

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION")
    print("15K DATA EXPLORATION")
    print("=" * 70)

    for filename in FILES:

        explore_file(filename)

    print()
    print("=" * 70)
    print("DATA EXPLORATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()