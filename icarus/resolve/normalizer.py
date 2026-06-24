"""Normalizer: the single place every field's raw OCR string is shaped/whitelisted/cast,
plus the Field value shape it emits. Canonical forms:
ordernr->int, type->alpha-lower, straat->letters/spaces-lower, postcode->4d2l upper,
huisnummer->int. Anything that fails its shape becomes None and is dropped."""
import re
from dataclasses import dataclass

from ..text_utils import _norm

_PC = re.compile(r"(\d{4})\s*([A-Z]{2})")    


@dataclass
class Field:
    type: str       # one of FIELDS
    value: object   # normalized str/int


class Normalizer:
    def ordernr(self, raw):
        digits = "".join(c for c in (raw or "") if c.isdigit())
        return int(digits) if len(digits) == 3 else None

    def fields(self, raw):
        """raw {field: str|None} -> [Field(type, value)]. `addr` (letters pass) expands to
        straat/postcode; `addr_digits` (digits pass) yields the huisnummer votes."""
        out = []
        for ft, value in raw.items():
            if ft == "addr":
                for at, av in self._addr(value).items():
                    if av is not None:
                        out.append(Field(at, av))
            elif ft == "addr_digits":
                out.extend(Field("huisnummer", h) for h in self._houses(value))
            else:
                v = self._field(ft, value)
                if v is not None:
                    out.append(Field(ft, v))
        return out

    def _field(self, ft, raw):
        if raw is None:
            return None
        if ft == "type":
            t = "".join(c for c in raw if c.isalpha()).lower()
            return t or None
        if ft == "straat":
            return _norm(raw) or None
        if ft == "postcode":
            s = "".join(c for c in raw if c.isalnum()).upper()
            return s if _PC.fullmatch(s) else None
        if ft == "huisnummer":
            d = "".join(c for c in str(raw) if c.isdigit())
            return int(d) if d else None
        return None
        
    def _addr(self, raw):
        """Letters pass -> straat + postcode ONLY. The house is never taken from this read:
        on screen the house digits sit glued behind the postcode's ')', and under the letter
        whitelist ')1' smears into N/J/T — the leading digit dies and prefix junk (the
        pieces count) gets mistaken for a house. Houses come from _houses(addr_digits)."""
        empty = {"straat": None, "postcode": None}
        if not raw:
            return empty
        out = dict(empty)
        m = _PC.search(raw.upper())
        if m:
            out["postcode"] = m.group(1) + m.group(2)        # 4d2l, spaceless
        st = _norm(raw)                                      # strips digits/punct, lowercases
        # separator header leaking into an adjacent order's row: strip the phrase but
        # KEEP the rest — the street often shares the read ('timed orders barmaheerd')
        st = " ".join(st.replace("timed orders", " ").split())
        out["straat"] = st or None
        return out

    def _houses(self, raw):
        """Second (caps) pass -> house votes. Anchor on the full postcode pattern:
        4 digits, the two letters, at most one junk char (the ')'), then digits until
        the first non-digit. findall so a row bleeding into the next contributes its
        own match as just another vote. Falls back to the ')' / digit-group anchors
        for reads where the pc letters smeared. 5+ digit runs get truncated; the
        validator's trim pool repairs glued phantoms."""
        if not raw:
            return []
        s = raw.upper()
        runs = re.findall(r"\d{4}\s?[A-Z]{2}[^\d]?\s?(\d+)", s)
        if not runs:                                         # pc letters smeared away
            t = re.sub(r"\(\s*\d{1,2}\s*\)", " ", s.split("/", 1)[-1])
            if ")" in t:
                runs = re.findall(r"\d+", t.split(")", 1)[1])
            else:
                m = re.search(r"\d{3,5}", t)
                runs = re.findall(r"\d+", t[m.end():]) if m else []
        seen, out = set(), []
        for r in runs[:4]:
            if len(r) > 5:
                continue
            v = int(r[:4])
            if v not in seen:
                seen.add(v)
                out.append(v)
        return out