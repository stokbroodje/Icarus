"""ADDR index loader: build the in-memory ADDR index the Validator/Resolver consume from the
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

from ...text_utils import _norm
from ..scoring import _canon

_KEEP = ("nummeraanduiding_gid", "postcode", "huisnummer", "straatnaam",
         "woonplaatsnaam", "lat", "lon")

def load_addr(csv_path):
    """Build ADDR from a BAG-style CSV. Loads everything — deliveries cross town lines."""
    rows_by_street = defaultdict(list)
    st_acc = defaultdict(lambda: [0.0, 0.0, 0])      # norm street -> [sum_lat, sum_lon, n]
    pc_acc = defaultdict(lambda: [0.0, 0.0, 0])      # canon pc     -> [sum_lat, sum_lon, n]

    with open(csv_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
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
