"""Dataset loader: build the in-memory ADDR index the Validator/Resolver consume from the
franchise BAG export (adressen.csv). Pure I/O + grouping — the scoring/resolution policy
lives in the rest of resolve/. ADDR shape:

    streets:           [normalized street name, ...]      # fuzzy-match candidates
    rows_by_street:    {norm street -> [row dict, ...]}   # the street's CSV rows
    street_centroid:   {norm street -> (lat, lon)}        # mean of the street's rows
    postcode_centroid: {canon postcode -> (lat, lon)}     # mean of the postcode's rows

Keys are normalized to match the resolve layer: `streets` / `rows_by_street` / `street_centroid`
are keyed by _norm(straatnaam) (what street_winner returns); `postcode_centroid` by _canon
(spaceless upper). Each row keeps only the fields the resolver reads, with the original-case
straatnaam/woonplaatsnaam preserved for display.
"""
import csv
from collections import defaultdict

from ..text_utils import _norm, _fold
from .scoring import _canon

_KEEP = ("nummeraanduiding_gid", "postcode", "huisnummer", "straatnaam",
         "woonplaatsnaam", "lat", "lon")
_KEYWORDS = ("PIECE", "ONLINE", "NEW", "DELIVERY", "CARRYOUT", "DINEIN", "TIMED", "EXPEDITED")

def load_addr(csv_path, woonplaats=None):
    """Build ADDR from a BAG-style CSV. `woonplaats` (str or iterable) restricts to those
    places (case-insensitive); None loads everything."""
    if isinstance(woonplaats, str):
        woonplaats = (woonplaats,)
    keep_wp = {w.strip().lower() for w in woonplaats} if woonplaats else None

    rows_by_street = defaultdict(list)
    st_acc = defaultdict(lambda: [0.0, 0.0, 0])      # norm street -> [sum_lat, sum_lon, n]
    pc_acc = defaultdict(lambda: [0.0, 0.0, 0])      # canon pc     -> [sum_lat, sum_lon, n]

    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if keep_wp is not None and r["woonplaatsnaam"].strip().lower() not in keep_wp:
                continue
            skey = _norm(r["straatnaam"])
            if not skey:
                continue
            rows_by_street[skey].append({k: r[k] for k in _KEEP})
            try:
                lat, lon = float(r["lat"]), float(r["lon"])
            except (TypeError, ValueError):
                continue
            a = st_acc[skey]; a[0] += lat; a[1] += lon; a[2] += 1
            p = pc_acc[_canon(r["postcode"])]; p[0] += lat; p[1] += lon; p[2] += 1

    return {
        "streets": sorted(rows_by_street),
        "rows_by_street": dict(rows_by_street),
        "street_centroid": {s: (a[0] / a[2], a[1] / a[2]) for s, a in st_acc.items() if a[2]},
        "postcode_centroid": {p: (v[0] / v[2], v[1] / v[2]) for p, v in pc_acc.items() if v[2]},
    }

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