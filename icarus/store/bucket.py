"""Bucket: raw normalized votes per field for one order, with a prune gate — after an
upstream latch, prune drops out-of-range votes already held AND gates future ones."""


class Bucket:
    def __init__(self):
        self.type = []
        self.straat = []
        self.postcode = []
        self.huisnummer = []
        self._valid = {}        # field -> allowed value set (str), once pruned

    def add(self, f):
        lst = getattr(self, f.type)
        valid = self._valid.get(f.type)
        if valid is not None and str(f.value) not in valid:
            return len(lst)                      # out of range -> dropped
        lst.append(f.value)
        return len(lst)

    def get(self, field_type):
        return getattr(self, field_type)

    def prune(self, field_type, valid):
        """Keep only in-range votes; gate future votes to `valid` (compared as str)."""
        valid = set(valid)
        self._valid[field_type] = valid
        setattr(self, field_type, [v for v in getattr(self, field_type) if str(v) in valid])
