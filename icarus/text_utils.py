"""Shared low-level string primitives. No CSV, no state."""
import re, unicodedata


def _fold(s):
    """ASCII-fold: é→e, ç→c, ï→i. For matching reads OCR may strip accents from."""
    return "".join(c for c in unicodedata.normalize("NFKD", s or "")
                   if not unicodedata.combining(c))

def _norm(s):
    """Lowercase, keep letters/spaces only."""
    return re.sub(r'[^A-Za-zÀ-ÿ\s]', '', s or '').strip().lower()


def _one_off(a, b):
    """Equal, or differ by exactly one same-position substitution."""
    if a == b:
        return True
    if len(a) != len(b):
        return False
    return sum(x != y for x, y in zip(a, b)) == 1
