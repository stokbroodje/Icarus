"""Validator: scores threshold-passed fields against the CSV, enforces the chain
(type -> straat -> postcode/huisnummer), latches on occurrence + confidence, and runs
the prune cascade. `resolve` produces the address record the pipeline ships out."""
import difflib
from collections import Counter

from ..constants import TYPES, NO_ADDRESS_TYPES
from .scoring import street_winner, postcode_score, house_score, snap_postcode, _canon
from .resolver import Resolver

CHAIN = ("type", "straat", "postcode", "huisnummer")
PREREQ = {"type": None, "straat": "type", "postcode": "straat", "huisnummer": "straat"}


def _snap_type(raw):
    """Nearest canonical type + its match ratio (raw is already alpha-lower)."""
    canon = difflib.get_close_matches(raw, TYPES, n=1, cutoff=0.0)[0]
    return canon, difflib.SequenceMatcher(None, raw, canon).ratio()


class Validator:
    def __init__(self, thresholds, addr):
        self.th = thresholds
        self.addr = addr
        self.resolver = Resolver(addr)

    # --- latch decisions --------------------------------------------------
    def validate(self, data, ordernr, passed):
        latches = data.latches[ordernr]
        for ft in CHAIN:                       # chain order, so a same-frame straat latch
            if ft not in passed or latches[ft]:  # unlocks postcode/huisnummer below it
                continue
            prq = PREREQ[ft]
            if prq and not latches[prq]:
                continue
            conf, value, occ = self._score(data, ordernr, ft)
            data.confidences[ordernr][ft] = conf
            if value is not None and occ >= self.th[ft] and conf >= self.th[f"{ft}_cutoff"]:
                data.values[ordernr][ft] = value
                latches[ft] = True
                if ft == "type" and value in NO_ADDRESS_TYPES:
                    self._exclude_address(data, ordernr)
                else:
                    self._cascade(data, ordernr, ft)

    def _score(self, data, ordernr, ft):
        """(confidence, committed_value, occurrence) for the leading candidate."""
        votes = data.buckets[ordernr].get(ft)
        if not votes:
            return 0.0, None, 0
        if ft == "type":
            return self._score_type(votes)
        if ft == "straat":
            canon, occ, conf = street_winner(votes, self.addr["streets"], self.th["straat_cutoff"])
            return conf, canon, occ
        street = data.values[ordernr]["straat"]
        if ft == "postcode":
            # canonical aggregation, like type/straat: snap each vote to its CSV postcode
            # under the street so 8/9 misreads count toward the real value
            agg = Counter(c for raw in votes if (c := snap_postcode(str(raw), street, self.addr)))
            if not agg:
                val, occ = Counter(votes).most_common(1)[0]
                return postcode_score(str(val), street, self.addr), val, occ
            val, occ = agg.most_common(1)[0]
            return postcode_score(str(val), street, self.addr), val, occ
        # huisnummer: pooled, block-aware winner. Glued phantom digits make 112 arrive
        # as 1121/11241, so each vote also contributes its leading- and trailing-trimmed
        # variants to the pool (1121 -> 121, 112); identity votes outrank trim-only
        # candidates at equal CSV standing. Winner order: exact under the latched pc
        # block, exact under the street, untrimmed, pooled count. The committed SCORE is
        # street-wide so a pc misread can't gate the latch.
        pc = data.values[ordernr]["postcode"] if data.latches[ordernr]["postcode"] else None
        # candidate ranks: identity 3 > trailing-trim 2 (appended phantoms are the
        # common artifact) > prepend-1 1 (eaten leading digit) > leading-trim 0
        pool = {}                                  # value -> [pooled_count, best_rank]
        for v, c in Counter(votes).items():
            s = str(v)
            cands = [(v, 3), (int("1" + s), 1)]
            if len(s) >= 2:
                cands += [(int(s[:-1]), 2), (int(s[1:]), 0)]
            for cand, rank in cands:
                e = pool.setdefault(cand, [0, -1])
                e[0] += c
                e[1] = max(e[1], rank)
        val, (occ, _) = max(pool.items(), key=lambda kv: (
            house_score(kv[0], street, pc, self.addr) if pc else 0.0,
            house_score(kv[0], street, None, self.addr),
            kv[1][1],
            kv[1][0]))
        return house_score(val, street, None, self.addr), val, occ

    def _score_type(self, votes):
        agg = {}
        for raw in votes:
            canon, ratio = _snap_type(raw)
            a = agg.setdefault(canon, [0, 0.0])
            a[0] += 1
            a[1] = max(a[1], ratio)
        canon = max(agg, key=lambda k: agg[k][0])
        occ, ratio = agg[canon]
        conf = round(ratio if ratio >= self.th["type_cutoff"] else 0.0, 3)
        return conf, (canon if conf else None), occ

    # --- address-excluded types -------------------------------------------
    def _exclude_address(self, data, ordernr):
        """Carryout/Dinein carry no delivery address: latch the whole address tail terminal
        so it's excluded from address OCR from here on and the order counts as resolved."""
        for ft in ("straat", "postcode", "huisnummer"):
            data.latches[ordernr][ft] = True

    # --- prune cascade ----------------------------------------------------
    def _cascade(self, data, ordernr, ft):
        """Street commit narrows postcode AND house to the street's rows. The pc commit
        does NOT further prune house: a one-letter pc misread (ST/SE) would evict the
        true house from the bucket forever — the street prune is constraint enough."""
        if ft != "straat":
            return
        b = data.buckets[ordernr]
        rows = self.addr["rows_by_street"].get(data.values[ordernr]["straat"], [])
        b.prune("postcode", {_canon(r["postcode"]) for r in rows})
        # huisnummer is NOT pruned: glued reads (247 for 24, 1121 for 112) must stay in
        # the bucket so the pooled trim/prepend candidates can recover them at scoring

    # --- resolution -------------------------------------------------------
    def resolve(self, data, ordernr):
        """(computed_address_record | None, addr_latched). Recomputed each frame from the
        current vote state — postcode/house keep voting after the street latches."""
        latches = data.latches[ordernr]
        street = data.values[ordernr]["straat"]
        if not latches["straat"] or street is None:   # unlatched, or address-excluded type
            return None, False
        b = data.buckets[ordernr]
        pc_val = self._top(b.get("postcode"))
        house_val = self._top(b.get("huisnummer"))
        pconf = postcode_score(str(pc_val), street, self.addr) if pc_val is not None else 0.0
        hpc = data.values[ordernr]["postcode"] if latches["postcode"] else None
        hconf = house_score(house_val, street, hpc, self.addr) if house_val is not None else 0.0
        return self.resolver.resolve(street, latches, pc_val, house_val, pconf, hconf)

    @staticmethod
    def _top(votes):
        return Counter(votes).most_common(1)[0][0] if votes else None
