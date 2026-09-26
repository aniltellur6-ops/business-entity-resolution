import re
import os
import pandas as pd
from difflib import SequenceMatcher


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

INPUT_FILES = {
    "source2": os.path.join(
        OUTPUT_DIR,
        "candidate_pairs_source2_15000.tsv"
    ),
    "source3": os.path.join(
        OUTPUT_DIR,
        "candidate_pairs_source3_15000.tsv"
    )
}


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


def tokenize(text):

    text = normalize_text(text)

    if not text:
        return set()

    return set(
        text.split()
    )


def exact_match(text1, text2):

    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    if not text1 or not text2:
        return 0

    return int(
        text1 == text2
    )


def similarity(text1, text2):

    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    if not text1 or not text2:
        return 0.0

    return SequenceMatcher(
        None,
        text1,
        text2
    ).ratio()


def length_difference(text1, text2):

    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    return abs(
        len(text1) - len(text2)
    )


def length_ratio(text1, text2):

    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    if not text1 or not text2:
        return 0.0

    longer = max(
        len(text1),
        len(text2)
    )

    shorter = min(
        len(text1),
        len(text2)
    )

    if longer == 0:
        return 0.0

    return shorter / longer


def token_jaccard(text1, text2):

    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(
        tokens1.intersection(tokens2)
    )

    union = len(
        tokens1.union(tokens2)
    )

    if union == 0:
        return 0.0

    return intersection / union


def token_overlap(text1, text2):

    tokens1 = tokenize(text1)
    tokens2 = tokenize(text2)

    if not tokens1 or not tokens2:
        return 0.0

    intersection = len(
        tokens1.intersection(tokens2)
    )

    denominator = min(
        len(tokens1),
        len(tokens2)
    )

    if denominator == 0:
        return 0.0

    return intersection / denominator


def create_features(
    df,
    left_name,
    right_name
):

    features = pd.DataFrame(
        index=df.index
    )

    features[
        "name_exact_match"
    ] = [
        exact_match(a, b)
        for a, b in zip(
            df[left_name],
            df[right_name]
        )
    ]

    features[
        "name_similarity"
    ] = [
        similarity(a, b)
        for a, b in zip(
            df[left_name],
            df[right_name]
        )
    ]

    features[
        "name_length_difference"
    ] = [
        length_difference(a, b)
        for a, b in zip(
            df[left_name],
            df[right_name]
        )
    ]

    features[
        "name_length_ratio"
    ] = [
        length_ratio(a, b)
        for a, b in zip(
            df[left_name],
            df[right_name]
        )
    ]

    features[
        "name_token_jaccard"
    ] = [
        token_jaccard(a, b)
        for a, b in zip(
            df[left_name],
            df[right_name]
        )
    ]

    features[
        "name_token_overlap"
    ] = [
        token_overlap(a, b)
        for a, b in zip(
            df[left_name],
            df[right_name]
        )
    ]

    return features


def create_pair_features(df):

    features = pd.DataFrame(
        index=df.index
    )

    name_features = create_features(
        df,
        "source1_business_name",
        "target_business_name"
    )

    features = pd.concat(
        [
            features,
            name_features
        ],
        axis=1
    )

    address_features = create_features(
        df,
        "source1_business_address",
        "target_business_address"
    )

    features[
        "address_exact_match"
    ] = (
        address_features[
            "name_exact_match"
        ]
    )

    features[
        "address_similarity"
    ] = (
        address_features[
            "name_similarity"
        ]
    )

    features[
        "address_length_difference"
    ] = (
        address_features[
            "name_length_difference"
        ]
    )

    features[
        "address_length_ratio"
    ] = (
        address_features[
            "name_length_ratio"
        ]
    )

    features[
        "address_token_jaccard"
    ] = (
        address_features[
            "name_token_jaccard"
        ]
    )

    features[
        "address_token_overlap"
    ] = (
        address_features[
            "name_token_overlap"
        ]
    )

    features[
        "country_match"
    ] = [
        int(
            normalize_text(a)
            ==
            normalize_text(b)
        )
        if normalize_text(a)
        and normalize_text(b)
        else 0
        for a, b in zip(
            df["source1_country"],
            df["target_country"]
        )
    ]

    return features


def process_file(
    target_source,
    input_file
):

    print()
    print("=" * 70)
    print(
        "FEATURE ENGINEERING:",
        target_source
    )
    print("=" * 70)

    df = pd.read_csv(
        input_file,
        sep="\t",
        dtype=str,
        keep_default_na=False
    )

    print(
        "Candidate rows:",
        len(df)
    )

    features = create_pair_features(
        df
    )

    result_columns = [
        "source1_entity_id",
        "target_source",
        "target_entity_id",
        "source1_business_name",
        "target_business_name",
        "source1_business_address",
        "target_business_address",
        "source1_country",
        "target_country"
    ]

    result = df[
        result_columns
    ].copy()

    result = pd.concat(
        [
            result,
            features
        ],
        axis=1
    )

    output_file = os.path.join(
        OUTPUT_DIR,
        "features_" +
        target_source +
        "_15000.tsv"
    )

    result.to_csv(
        output_file,
        sep="\t",
        index=False
    )

    print(
        "Feature rows:",
        len(result)
    )

    print(
        "Feature columns:",
        len(features.columns)
    )

    print()
    print("Feature names:")

    for column in features.columns:
        print(
            "  -",
            column
        )

    print()
    print(
        "Output:",
        output_file
    )


def main():

    print("=" * 70)
    print("BUSINESS ENTITY RESOLUTION")
    print("15K FEATURE ENGINEERING")
    print("=" * 70)

    for target_source, input_file in INPUT_FILES.items():

        process_file(
            target_source,
            input_file
        )

    print()
    print("=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()