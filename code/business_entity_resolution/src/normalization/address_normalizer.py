import re
from .text_normalizer import normalize_text

ADDRESS_ABBREVIATIONS = {
    r'\brd\b': 'road',
    r'\bst\b': 'street',
    r'\bapt\b': 'apartment',
    r'\bave\b': 'avenue',
    r'\bblvd\b': 'boulevard',
    r'\bdr\b': 'drive',
    r'\bln\b': 'lane',
    r'\bsq\b': 'square',
    r'\bpkwy\b': 'parkway',
    r'\bste\b': 'suite',
    r'\bbldg\b': 'building',
    r'\bfl\b': 'floor',
    r'\brm\b': 'room'
}

def normalize_address_string(raw_address: str) -> str:
    """Normalizes the address text by standardizing abbreviations."""
    norm_address = normalize_text(raw_address)
    
    for pattern, canonical in ADDRESS_ABBREVIATIONS.items():
        norm_address = re.sub(pattern, canonical, norm_address)
        
    return norm_address

def normalize_address(raw_address: str) -> dict:
    """Returns a dictionary with original and normalized address."""
    return {
        "original_address": raw_address,
        "normalized_address": normalize_address_string(raw_address)
    }
