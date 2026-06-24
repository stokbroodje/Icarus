"""Tesseract user-words generation from the ADDR index. Emits one token per line, UPPER:
street tokens (+ accent-folded variants), postcode prefixes and letter-pairs, and a fixed
set of order-flow keywords. Used as a Tesseract user-words file to bias OCR toward the
vocabulary that actually appears in the dataset.

Consumes the ADDR dict built by addr_index.load_addr (reads `streets` and
`postcode_centroid` only).
"""
from ...text_utils import _fold

_KEYWORDS = ("PIECE", "ONLINE", "NEW", "DELIVERY", "CARRYOUT", "DINEIN", "TIMED", "EXPEDITED")

def write_wordlist(addr, out_path):
    """Tesseract user-words from ADDR: street tokens (+ accent-folded) + postcode
    prefixes + letter-pairs + fixed keywords. One token per line, UPPER."""
    toks = set(_KEYWORDS)
    for street in addr["streets"]:                       # _norm'd already
        for t in street.split():
            if len(t) > 1:
                toks.add(t.upper())
                toks.add(_fold(t).upper())
    for pc in addr["postcode_centroid"]:                 # canon: spaceless upper
        if len(pc) >= 6 and pc[:4].isdigit():
            toks.add(pc[:4]); toks.add(pc[4:6])
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(toks)) + "\n")
    return out_path
