import re
import unicodedata

def normalize_unicode(text: str) -> str:
    """Convert unicode characters to ASCII equivalent, e.g., 'é' -> 'e'."""
    if not isinstance(text, str):
        return ""
    # Normalize using NFKD and encode to ascii ignoring non-ascii, then decode back
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')

def normalize_text(text: str) -> str:
    """Basic text normalization: lowercase, unicode, punctuation, and whitespace."""
    if not isinstance(text, str) or not text.strip():
        return ""
    
    # Lowercase
    text = text.lower()
    
    # Unicode normalize
    text = normalize_unicode(text)
    
    # Remove punctuation, replace with space
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text
