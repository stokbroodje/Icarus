"""Data: buckets, latches, confidences, committed values. No lifecycle, no confidence
logic — it stores votes and the verdicts the Validator writes back."""
from .bucket import Bucket
from ..constants import FIELDS


class Data:
    def __init__(self, thresholds):
        self.th = thresholds
        self.buckets = {}        # ordernr -> Bucket
        self.latches = {}        # ordernr -> {field: bool}
        self.confidences = {}    # ordernr -> {field: float}
        self.values = {}         # ordernr -> {field: committed value}

    def get_or_create(self, ordernr):
        if ordernr not in self.buckets:
            self.buckets[ordernr] = Bucket()
            self.latches[ordernr] = {f: False for f in FIELDS}
            self.confidences[ordernr] = {f: 0.0 for f in FIELDS}
            self.values[ordernr] = {f: None for f in FIELDS}

    def latched(self, ordernr):
        return self.latches[ordernr]

    def add(self, ordernr, fields):
        """Append votes; return field types whose count crossed the occurrence threshold."""
        b = self.buckets[ordernr]
        passed = []
        for f in fields:
            if self.latches[ordernr][f.type]:
                continue
            n = b.add(f)
            if n >= self.th[f.type]:
                passed.append(f.type)
        return passed
