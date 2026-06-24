"""Resolver: the address-resolution policy. Given the latched street and the postcode/house
vote state, it picks a full address — a specific CSV row (lat/lon/gid) or a centroid
fallback. Computes both a house-driven and a postcode-driven candidate, never hard-gates on
postcode. The Validator owns one of these and feeds it the current vote state; swap this
class to change the policy."""
from .scoring import _canon
from ..text_utils import _norm


class Resolver:
    def __init__(self, addr):
        self.addr = addr

    def resolve(self, street, latches, pc_val, house_val, pconf, hconf):
        """(chosen_record, addr_latched). street+house IS the row (unique within a street),
        so a latched house drives resolution; the postcode is confirmation, not a gate —
        agreement raises confidence, disagreement caps it. Order of preference:
        house-final, then pc-block-final, then pc centroid, then street centroid."""
        rows = self.addr["rows_by_street"].get(street, [])
        house_driven = self._house_driven(rows, house_val, hconf, street)
        postcode_driven = self._postcode_driven(rows, pc_val, house_val, pconf, hconf, street)

        addr_latched = latches["straat"] and (latches["huisnummer"] or latches["postcode"])

        if latches["huisnummer"] and house_driven["resolve"] == "final":
            chosen = dict(house_driven)
            if pc_val is not None:                          # pc as confirmation signal
                if _canon(chosen.get("postcode") or "") == pc_val:
                    chosen["confidence"] = round(
                        min(1.0, chosen["confidence"] + 0.2 * max(pconf, 0.5)), 3)
                else:
                    chosen["confidence"] = round(min(chosen["confidence"], 0.75), 3)
        elif latches["postcode"] and postcode_driven["resolve"] == "final":
            chosen = postcode_driven
        elif postcode_driven["lat"] is not None:
            chosen = postcode_driven
        else:
            chosen = house_driven
        return chosen, addr_latched

    # --- candidates -------------------------------------------------------
    def _house_driven(self, rows, house_val, hconf, street):
        row, exact = self._pick_house(rows, house_val)
        if exact:
            return self._from_row(row, "final", round(min(1.0, 0.6 + 0.4 * hconf), 3))
        canon = rows[0]["straatnaam"] if rows else street
        return self._centroid(canon, None, "street", 0.3)

    def _postcode_driven(self, rows, pc_val, house_val, pconf, hconf, street):
        canon = rows[0]["straatnaam"] if rows else street
        pcs = {_canon(r["postcode"]) for r in rows}
        if pc_val not in pcs:
            return self._centroid(canon, None, "street", 0.3)
        block = [r for r in rows if _canon(r["postcode"]) == pc_val]
        row, exact = self._pick_house(block, house_val)
        if exact:
            return self._from_row(row, "final", round(min(1.0, 0.6 + 0.4 * hconf), 3))
        return self._centroid(canon, pc_val, "postcode", round(0.6 + 0.2 * pconf, 3))

    # --- record builders --------------------------------------------------
    @staticmethod
    def _pick_house(rows, house_val):
        """Exact row for house_val, else nearest numeric row, else None."""
        by_house = {str(r["huisnummer"]): r for r in rows}
        if house_val is not None and str(house_val) in by_house:
            return by_house[str(house_val)], True
        if house_val is not None and str(house_val).isdigit():
            digits = [r for r in rows if str(r["huisnummer"]).isdigit()]
            if digits:
                return min(digits, key=lambda r: abs(int(r["huisnummer"]) - int(house_val))), False
        return None, False

    @staticmethod
    def _from_row(row, tier, conf):
        return {"straat": row["straatnaam"], "huisnummer": row["huisnummer"],
                "postcode": row["postcode"], "woonplaats": row["woonplaatsnaam"].strip(),
                "lat": float(row["lat"]) if row["lat"] else None,
                "lon": float(row["lon"]) if row["lon"] else None,
                "gid": int(row["nummeraanduiding_gid"]) if row.get("nummeraanduiding_gid") else None,
                "resolve": tier, "confidence": conf}

    def _centroid(self, canon, pc, tier, conf):
        lat = lon = None
        if tier == "postcode" and pc and pc in self.addr.get("postcode_centroid", {}):
            lat, lon = self.addr["postcode_centroid"][pc]
        elif _norm(canon) in self.addr.get("street_centroid", {}):
            lat, lon = self.addr["street_centroid"][_norm(canon)]
        return {"straat": canon.title(), "huisnummer": None, "postcode": pc, "woonplaats": None,
                "lat": lat, "lon": lon, "gid": None, "resolve": tier, "confidence": conf}
