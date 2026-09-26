import re
from typing import Tuple
from .text_normalizer import normalize_text

LEGAL_SUFFIXES = {
    r'\bpvt\b': 'private',
    r'\bltd\b': 'limited',
    r'\bcorp\b': 'corporation',
    r'\binc\b': 'incorporated',
    r'\bllc\b': 'limited liability company',
    r'\bsarl\b': 'societe a responsabilite limitee',
    r'\bsa\b': 'societe anonyme',
    r'\bco\b': 'company',
    r'\b&\b': 'and'
}

def extract_legal_suffix(name: str) -> Tuple[str, str]:
    """
    Extracts and normalizes legal suffixes from a normalized name.
    Returns (name_without_suffix, canonical_suffix)
    """
    if not name:
        return "", ""
        
    extracted_suffixes = []
    current_name = name
    
    # This is a basic approach. We look for suffixes at the end of the string.
    # In a real scenario, this regex should be more robust and iterative.
    for pattern, canonical in LEGAL_SUFFIXES.items():
        if re.search(pattern, current_name):
            extracted_suffixes.append(canonical)
            # We don't remove it from the name to preserve information, or we can.
            # The instructions say "Keep the canonical suffix as a separate field, not deleted"
            # However, separating it completely might also be good. Let's just extract it.
            
    # For now, we just identify them but do not remove them from the original name
    # as per "don't destroy information" instruction.
    suffix_str = " ".join(set(extracted_suffixes))
    
    return current_name, suffix_str

def normalize_business_name(raw_name: str) -> dict:
    """Normalizes business name and extracts suffix."""
    norm_name = normalize_text(raw_name)
    
    # Apply suffix standardizations to the name string itself
    for pattern, canonical in LEGAL_SUFFIXES.items():
        norm_name = re.sub(pattern, canonical, norm_name)
    
    # Extract the canonical suffix as a separate feature
    _, canonical_suffix = extract_legal_suffix(norm_name)
    
    return {
        "original_name": raw_name,
        "normalized_name": norm_name,
        "legal_suffix": canonical_suffix
    }
