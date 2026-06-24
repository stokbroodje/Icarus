"""Shared CSV scoring: street fuzzy-match aggregation, postcode-under-street, and
house-in-range. Pure functions over the address dataset; no state. Used by both the
Validator (latch decisions) and the Resolver (resolution confidence)."""
import difflib

from ..text_utils import _norm, _one_off


def _canon(pc):
    return (pc or "").replace(" ", "").upper()


MIN_STREET = 3   # the BAG has streets named 'D'/'E'/'Ra'; an OCR split-token matching one
                 # at ratio 1.0 outscores the real street, so short canons don't compete


def _best_span(raw, streets, cutoff=0.8, max_span=4):
    """Best fuzzy street match over 1..n token spans of a noisy read."""
    toks = _norm(raw).split()
    if not toks:
        return None, 0.0
    best = (None, 0.0)
    for n in range(1, min(max_span, len(toks)) + 1):
        for s in range(0, len(toks) - n + 1):
            span = " ".join(toks[s:s + n])
            if len(span) < MIN_STREET:
                continue
            m = difflib.get_close_matches(span, streets, n=1, cutoff=cutoff)
            if m and len(m[0]) >= MIN_STREET:
                r = difflib.SequenceMatcher(None, span, m[0]).ratio()
                if r > best[1]:
                    best = (m[0], r)
    return best


def street_winner(votes, streets, cutoff):
    """Aggregate raw reads by fuzzy-matched canonical street. (canon, occ, conf)."""
    agg = {}
    for raw in votes:
        m, r = _best_span(raw, streets)
        if not m:
            continue
        a = agg.setdefault(m, [0, 0.0])
        a[0] += 1
        a[1] = max(a[1], r)
    if not agg:
        return None, 0, 0.0
    canon = max(agg, key=lambda k: agg[k][0])
    occ, ratio = agg[canon]
    conf = round(ratio if ratio > cutoff else 0.0, 3)
    return (canon if conf else None), occ, conf


def postcode_score(value, street, addr):
    """Exact under street -> 1.0; one char off -> 0.5; otherwise 0."""
    pcs = {_canon(r["postcode"]) for r in addr["rows_by_street"].get(street, [])}
    if not pcs:
        return 0.0
    if value in pcs:
        return 1.0
    return 0.5 if any(_one_off(value, p) for p in pcs) else 0.0


def house_rows(street, postcode, addr):
    """Rows under the street, narrowed to a postcode block when one is given."""
    rows = addr["rows_by_street"].get(street, [])
    if postcode:
        rows = [r for r in rows if _canon(r["postcode"]) == postcode]
    return rows


def house_score(value, street, postcode, addr):
    """Exact under the street -> 1.0 (the postcode block is NOT a gate here: a misread
    postcode letter must not block a 30x-consistent house). The block only narrows the
    distance decay for non-exact values."""
    sval = str(value)
    if sval in {str(r["huisnummer"]) for r in addr["rows_by_street"].get(street, [])}:
        return 1.0
    nums = {str(r["huisnummer"]) for r in house_rows(street, postcode, addr)}
    ints = [int(x) for x in nums if x.isdigit()]
    if sval.isdigit() and ints:
        d = min(abs(int(sval) - i) for i in ints)
        return round(max(0.0, 1.0 - d / 50.0), 3)
    return 0.0


def snap_postcode(value, street, addr):
    """Canonical postcode for a vote: exact CSV pc under the street, or the UNIQUE
    one-off (8737RH -> 9737RH). Ambiguous or unmatchable -> None."""
    pcs = {_canon(r["postcode"]) for r in addr["rows_by_street"].get(street, [])}
    if value in pcs:
        return value
    near = [p for p in pcs if _one_off(value, p)]
    return near[0] if len(near) == 1 else None
