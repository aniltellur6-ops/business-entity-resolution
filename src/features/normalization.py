import os
import re
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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

FILES = [
    "test_source1_15000.tsv",
    "test_source2_15000.tsv",
    "test_source3_15000.tsv"
]


def normalize_text(text):

    if pd.isna(text):
        return ""

    text = str(text).lower()

    text = re.sub(
        r"[^a-z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize_country(text):

    text = normalize_text(text)

    country_map = {
        "india": "india",
        "ind": "india",

        "usa": "united states",
        "us": "united states",
        "u s": "united states",
        "united states": "united states",

        "france": "france",
        "fra": "france"
    }

    return country_map.get(
        text,
        text
    )


def normalize_file(filename):

    input_file = os.path.join(
        DATA_DIR,
        filename
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        filename.replace(
            ".tsv",
            "_normalized.tsv"
        )
    )

    print()
    print("=" * 70)
    print("Processing:", filename)
    print("=" * 70)

    df = pd.read_csv(
        input_file,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    df["business_name_normalized"] = (
        df["business_name"]
        .apply(normalize_text)
    )

    df["business_address_normalized"] = (
        df["business_address"]
        .apply(normalize_text)
    )

    df["country_normalized"] = (
        df["country"]
        .apply(normalize_country)
    )

    df.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    print(
        "Rows processed:",
        len(df)
    )

    print(
        "Output:",
        output_file
    )

    print()
    print("Original vs normalized examples:")

    examples = df[
        [
            "business_name",
            "business_name_normalized",
            "business_address",
            "business_address_normalized"
        ]
    ].head(5)

    print(
        examples.to_string(
            index=False
        )
    )


def main():

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION")
    print("15K DATA NORMALIZATION")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    for filename in FILES:

        normalize_file(filename)

    print()
    print("=" * 70)
    print("NORMALIZATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()