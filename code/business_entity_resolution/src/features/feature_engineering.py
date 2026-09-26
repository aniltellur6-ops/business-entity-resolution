import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
from rapidfuzz import distance, fuzz

def safe_str(val):
    if pd.isna(val) or val is None:
        return ""
    return str(val)

def exact_match(text1, text2):
    return int(safe_str(text1) == safe_str(text2))

def length_difference(text1, text2):
    return abs(len(safe_str(text1)) - len(safe_str(text2)))

def levenshtein_sim(text1, text2):
    t1, t2 = safe_str(text1), safe_str(text2)
    if not t1 and not t2: return 1.0
    if not t1 or not t2: return 0.0
    return fuzz.ratio(t1, t2) / 100.0

def jaro_winkler_sim(text1, text2):
    t1, t2 = safe_str(text1), safe_str(text2)
    if not t1 and not t2: return 1.0
    if not t1 or not t2: return 0.0
    return distance.JaroWinkler.similarity(t1, t2)

def token_jaccard(text1, text2):
    t1, t2 = safe_str(text1).split(), safe_str(text2).split()
    set1, set2 = set(t1), set(t2)
    if not set1 and not set2: return 1.0
    if not set1 or not set2: return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def token_sort_ratio(text1, text2):
    t1, t2 = safe_str(text1), safe_str(text2)
    if not t1 and not t2: return 1.0
    if not t1 or not t2: return 0.0
    return fuzz.token_sort_ratio(t1, t2) / 100.0

def partial_ratio(text1, text2):
    t1, t2 = safe_str(text1), safe_str(text2)
    if not t1 and not t2: return 1.0
    if not t1 or not t2: return 0.0
    return fuzz.partial_ratio(t1, t2) / 100.0

def create_features(df: pd.DataFrame, left_column: str = "name_1", right_column: str = "name_2") -> pd.DataFrame:
    """
    Creates advanced fuzzy matching features for ML training and inference.
    """
    print(f"Generating features for {len(df)} pairs...")
    features = pd.DataFrame(index=df.index)
    
    # Exact and Length features
    features["exact_match"] = df.apply(lambda r: exact_match(r[left_column], r[right_column]), axis=1)
    features["length_difference"] = df.apply(lambda r: length_difference(r[left_column], r[right_column]), axis=1)
    
    # Fuzzy String Matching
    features["levenshtein_sim"] = df.apply(lambda r: levenshtein_sim(r[left_column], r[right_column]), axis=1)
    features["jaro_winkler_sim"] = df.apply(lambda r: jaro_winkler_sim(r[left_column], r[right_column]), axis=1)
    features["token_sort_ratio"] = df.apply(lambda r: token_sort_ratio(r[left_column], r[right_column]), axis=1)
    features["partial_ratio"] = df.apply(lambda r: partial_ratio(r[left_column], r[right_column]), axis=1)
    
    # Set-based Matching
    features["token_jaccard"] = df.apply(lambda r: token_jaccard(r[left_column], r[right_column]), axis=1)
    
    return features

if __name__ == "__main__":
    # Small test dataset
    data = {
        "name_1": ["ABC Technologies", "Shree Medical Store", "Google India", "Apple"],
        "name_2": ["ABC Technologies", "Medical Shree", "Google", ""]
    }
    test_df = pd.DataFrame(data)
    f = create_features(test_df)
    print(f)
