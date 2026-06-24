"""Track: per-order live/timed countdown + created_at. No OCR data here."""
import time

from ...constants import HOLD


class Track:
    def __init__(self, ordernr, timed):
        self.live = HOLD
        self.timed = HOLD if timed else 0
        self.created_at = time.time()

    def update(self, live, timed):
        """Tick toward expiry; return (dropped_live, dropped_timed)."""
        un_live = un_timed = False
        if live:
            self.live = HOLD
        else:
            self.live -= 1
            un_live = self.live <= 0
        if timed:
            self.timed = HOLD
        else:
            self.timed -= 1
            un_timed = self.timed <= 0
        return un_live, un_timed
