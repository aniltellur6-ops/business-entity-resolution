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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "output"
)

SOURCE1_FILE = os.path.join(
    OUTPUT_DIR,
    "test_source1_15000_normalized.tsv"
)

TARGET_FILES = {
    "source2": os.path.join(
        OUTPUT_DIR,
        "test_source2_15000_normalized.tsv"
    ),
    "source3": os.path.join(
        OUTPUT_DIR,
        "test_source3_15000_normalized.tsv"
    )
}

STOP_WORDS = {
    "the",
    "and",
    "pvt",
    "private",
    "limited",
    "ltd",
    "llp",
    "inc",
    "incorporated",
    "company",
    "co",
    "corp",
    "corporation",
    "sa",
    "sas",
    "sarl",
    "plc",
    "gmbh",
    "group",
    "holdings",
    "international"
}

MAX_CANDIDATES = 20


def make_block_key(name, country):

    name = str(name).strip()
    country = str(country).strip()

    tokens = re.findall(
        r"[a-z0-9]+",
        name
    )

    useful_tokens = []

    for token in tokens:

        if len(token) < 3:
            continue

        if token in STOP_WORDS:
            continue

        if token not in useful_tokens:
            useful_tokens.append(token)

    useful_tokens.sort(
        key=lambda token: (
            -len(token),
            token
        )
    )

    selected = useful_tokens[:2]

    if not selected:
        return country

    return country + "|" + "|".join(
        selected
    )


def load_file(filepath):

    return pd.read_csv(
        filepath,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )


def build_block_index(target_df):

    target_df = target_df.copy()

    target_df["block_key"] = [
        make_block_key(
            name,
            country
        )
        for name, country in zip(
            target_df[
                "business_name_normalized"
            ],
            target_df[
                "country_normalized"
            ]
        )
    ]

    return target_df


def generate_candidates(
    source1_df,
    target_df,
    target_source
):

    print()
    print(
        "Generating candidates for:",
        target_source
    )

    target_blocks = {}

    for index, row in target_df.iterrows():

        block_key = row["block_key"]

        if block_key not in target_blocks:
            target_blocks[block_key] = []

        target_blocks[block_key].append(
            index
        )

    candidate_rows = []

    total = len(source1_df)

    for position, (_, source1_row) in enumerate(
        source1_df.iterrows(),
        start=1
    ):

        block_key = make_block_key(
            source1_row[
                "business_name_normalized"
            ],
            source1_row[
                "country_normalized"
            ]
        )

        matching_indices = target_blocks.get(
            block_key,
            []
        )

        matching_indices = matching_indices[
            :MAX_CANDIDATES
        ]

        for target_index in matching_indices:

            target_row = target_df.loc[
                target_index
            ]

            candidate_rows.append(
                {
                    "source1_entity_id":
                        source1_row[
                            "entity_id"
                        ],

                    "target_source":
                        target_source,

                    "target_entity_id":
                        target_row[
                            "entity_id"
                        ],

                    "source1_business_name":
                        source1_row[
                            "business_name"
                        ],

                    "target_business_name":
                        target_row[
                            "business_name"
                        ],

                    "source1_business_address":
                        source1_row[
                            "business_address"
                        ],

                    "target_business_address":
                        target_row[
                            "business_address"
                        ],

                    "source1_country":
                        source1_row[
                            "country"
                        ],

                    "target_country":
                        target_row[
                            "country"
                        ],

                    "source1_business_name_normalized":
                        source1_row[
                            "business_name_normalized"
                        ],

                    "target_business_name_normalized":
                        target_row[
                            "business_name_normalized"
                        ],

                    "source1_business_address_normalized":
                        source1_row[
                            "business_address_normalized"
                        ],

                    "target_business_address_normalized":
                        target_row[
                            "business_address_normalized"
                        ],

                    "source1_country_normalized":
                        source1_row[
                            "country_normalized"
                        ],

                    "target_country_normalized":
                        target_row[
                            "country_normalized"
                        ],

                    "block_key":
                        block_key
                }
            )

        if (
            position % 1000 == 0
            or position == total
        ):

            print(
                f"Processed "
                f"{position}/{total}"
            )

    return pd.DataFrame(
        candidate_rows
    )


def main():

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION")
    print("15K CANDIDATE GENERATION")
    print("=" * 70)

    source1_df = load_file(
        SOURCE1_FILE
    )

    print()
    print(
        "Source1 rows:",
        len(source1_df)
    )

    for target_source, filepath in TARGET_FILES.items():

        target_df = load_file(
            filepath
        )

        print()
        print(
            target_source,
            "rows:",
            len(target_df)
        )

        target_df = build_block_index(
            target_df
        )

        candidates = generate_candidates(
            source1_df,
            target_df,
            target_source
        )

        output_file = os.path.join(
            OUTPUT_DIR,
            "candidate_pairs_" +
            target_source +
            "_15000.tsv"
        )

        candidates.to_csv(
            output_file,
            sep="\t",
            index=False
        )

        print()
        print(
            target_source,
            "candidate rows:",
            len(candidates)
        )

        print(
            "Output:",
            output_file
        )

    print()
    print("=" * 70)
    print("CANDIDATE GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()