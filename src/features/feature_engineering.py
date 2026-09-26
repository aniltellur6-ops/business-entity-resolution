import re
import pandas as pd


def normalize_text(text):
    """
    Convert text into a normalized form.
    """
    if pd.isna(text):
        return ""

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def exact_match(text1, text2):
    """
    Check whether two text values are exactly the same
    after normalization.
    """
    return int(normalize_text(text1) == normalize_text(text2))


def text_length_difference(text1, text2):
    """
    Calculate the absolute difference in text lengths.
    """
    text1 = normalize_text(text1)
    text2 = normalize_text(text2)

    return abs(len(text1) - len(text2))


def create_features(df, left_column, right_column):
    """
    Create basic features for two entity-name columns.
    """

    features = pd.DataFrame(index=df.index)

    features["exact_match"] = df.apply(
        lambda row: exact_match(
            row[left_column],
            row[right_column]
        ),
        axis=1
    )

    features["length_difference"] = df.apply(
        lambda row: text_length_difference(
            row[left_column],
            row[right_column]
        ),
        axis=1
    )

    return features


if __name__ == "__main__":
    # Small test dataset
    data = {
        "name_1": [
            "ABC Technologies",
            "Shree Medical Store",
            "Google India"
        ],
        "name_2": [
            "ABC Technologies",
            "Shree Medical",
            "Google"
        ]
    }

    df = pd.DataFrame(data)

    features = create_features(
        df,
        "name_1",
        "name_2"
    )

    print("Generated Features")
    print("------------------")
    print(features)