import pandas as pd
import numpy as np
from rapidfuzz import distance, fuzz
from sklearn.feature_extraction.text import TfidfVectorizer

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

def evaluate_presence_match(v1, v2):
    v1, v2 = safe_str(v1).strip(), safe_str(v2).strip()
    if not v1 and not v2: return 0 # both missing
    if not v1 or not v2: return 1  # one missing
    if v1 == v2: return 2          # both present and equal
    return 3                       # both present and different

def compute_tfidf_cosine_sim(names1, names2, analyzer='char_wb', ngram_range=(2,4)):
    all_names = pd.concat([names1, names2]).fillna('')
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram_range, min_df=2)
    try:
        vec.fit(all_names)
        v1 = vec.transform(names1.fillna(''))
        v2 = vec.transform(names2.fillna(''))
        sim = v1.multiply(v2).sum(axis=1)
        return np.array(sim).flatten()
    except ValueError:
        return np.zeros(len(names1))

def create_features(df: pd.DataFrame, prefix1: str = "_1", prefix2: str = "_2") -> pd.DataFrame:
    """
    Creates advanced fuzzy matching features for ML training and inference.
    """
    print(f"Generating features for {len(df)} pairs...")
    features = pd.DataFrame(index=df.index)
    
    # 1. Name Features
    name1 = "name" + prefix1
    name2 = "name" + prefix2
    if name1 in df.columns and name2 in df.columns:
        features["name_exact_match"] = df.apply(lambda r: exact_match(r[name1], r[name2]), axis=1)
        features["name_len_diff"] = df.apply(lambda r: length_difference(r[name1], r[name2]), axis=1)
        features["name_levenshtein"] = df.apply(lambda r: levenshtein_sim(r[name1], r[name2]), axis=1)
        features["name_jaro_winkler"] = df.apply(lambda r: jaro_winkler_sim(r[name1], r[name2]), axis=1)
        features["name_token_sort"] = df.apply(lambda r: token_sort_ratio(r[name1], r[name2]), axis=1)
        features["name_partial_ratio"] = df.apply(lambda r: partial_ratio(r[name1], r[name2]), axis=1)
        features["name_token_jaccard"] = df.apply(lambda r: token_jaccard(r[name1], r[name2]), axis=1)
        
        print("Computing TF-IDF character similarities for names...")
        features["name_tfidf_char_sim"] = compute_tfidf_cosine_sim(df[name1], df[name2], analyzer='char_wb', ngram_range=(2,4))
        
        print("Computing TF-IDF word (rarity-weighted token) similarities for names...")
        features["name_tfidf_word_sim"] = compute_tfidf_cosine_sim(df[name1], df[name2], analyzer='word', ngram_range=(1,1))
    
    # 2. Address Features
    addr1 = "address" + prefix1
    addr2 = "address" + prefix2
    if addr1 in df.columns and addr2 in df.columns:
        features["addr_exact_match"] = df.apply(lambda r: exact_match(r[addr1], r[addr2]), axis=1)
        features["addr_len_diff"] = df.apply(lambda r: length_difference(r[addr1], r[addr2]), axis=1)
        features["addr_levenshtein"] = df.apply(lambda r: levenshtein_sim(r[addr1], r[addr2]), axis=1)
        features["addr_jaro_winkler"] = df.apply(lambda r: jaro_winkler_sim(r[addr1], r[addr2]), axis=1)
        features["addr_token_sort"] = df.apply(lambda r: token_sort_ratio(r[addr1], r[addr2]), axis=1)
        features["addr_partial_ratio"] = df.apply(lambda r: partial_ratio(r[addr1], r[addr2]), axis=1)
        features["addr_token_jaccard"] = df.apply(lambda r: token_jaccard(r[addr1], r[addr2]), axis=1)
        
        print("Computing TF-IDF character similarities for addresses...")
        features["addr_tfidf_char_sim"] = compute_tfidf_cosine_sim(df[addr1], df[addr2], analyzer='char_wb', ngram_range=(2,4))
        
    # 3. Country Exact Match
    country1 = "country" + prefix1
    country2 = "country" + prefix2
    if country1 in df.columns and country2 in df.columns:
        features["country_exact_match"] = df.apply(lambda r: exact_match(r[country1], r[country2]), axis=1)
        
    # 4. Postal Code and House Number logic
    postal1 = "postal" + prefix1
    postal2 = "postal" + prefix2
    if postal1 in df.columns and postal2 in df.columns:
        features["postal_match_status"] = df.apply(lambda r: evaluate_presence_match(r[postal1], r[postal2]), axis=1)
        
    house1 = "house_num" + prefix1
    house2 = "house_num" + prefix2
    if house1 in df.columns and house2 in df.columns:
        features["house_match_status"] = df.apply(lambda r: evaluate_presence_match(r[house1], r[house2]), axis=1)

    # 5. Legal Suffix Match
    suf1 = "legal_suffix" + prefix1
    suf2 = "legal_suffix" + prefix2
    if suf1 in df.columns and suf2 in df.columns:
        features["suffix_match_status"] = df.apply(lambda r: evaluate_presence_match(r[suf1], r[suf2]), axis=1)
        
    return features
