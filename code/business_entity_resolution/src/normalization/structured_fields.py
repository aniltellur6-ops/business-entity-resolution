import re
from typing import Optional

def extract_postal_code(address: str) -> Optional[str]:
    """
    Attempts to extract a postal code or PIN code from an address string.
    This is generic and intentionally not country-specific.
    Matches typical 5-6 digit postal codes, potentially with spaces/hyphens (e.g. 75001, 123 456, 12345-6789).
    """
    if not address:
        return None
        
    # Look for 5 to 6 digits, possibly with a space or hyphen in the middle
    match = re.search(r'\b\d{2,3}[\s\-]?\d{3}\b', address)
    if match:
        # Return cleaned version
        return re.sub(r'[\s\-]', '', match.group(0))
        
    return None

def extract_house_number(address: str) -> Optional[str]:
    """
    Attempts to extract a house or building number from the beginning of an address.
    """
    if not address:
        return None
        
    # Match number at the start of the string or preceded by "no " / "num "
    match = re.search(r'(?:^|\b(?:no|num|number)\s+)(\d+[a-z]?(?:-\d+[a-z]?)?)\b', address)
    if match:
        return match.group(1)
        
    return None

def extract_city(address: str) -> Optional[str]:
    """
    A placeholder for generic city extraction. 
    Without external dictionaries, this is hard to do perfectly generically.
    Could be expanded to look for words before postal codes.
    """
    # Simple heuristic: word(s) immediately before a comma and a postal code
    # e.g., "12 mg road, pune 411001" -> pune
    if not address:
        return None
        
    match = re.search(r'([a-z\s]+),\s*(?:[a-z]{2}\s*)?\d{5,6}\b', address)
    if match:
        city_candidate = match.group(1).strip()
        # Ensure it's not the whole address
        if len(city_candidate.split()) < 4: 
            return city_candidate
            
    return None
