"""State: presence/timed now-state, keyed by ordernr. Emits lifecycle events per frame:
new, present, absent (live), timed, untimed."""
from .track import Track


class State:
    def __init__(self):
        self.tracks = {}     # ordernr -> Track
        self.active = []     # currently live
        self.timed = []      # currently timed
        self.seen = []       # ever seen

    def update(self, ordernrs, timed):
        events = []
        present = set(ordernrs)
        self._tick_missing(present, events)
        self._sighted(ordernrs, timed, events)
        return events

    def _tick_missing(self, present, events):
        """Actives not sighted this frame decay toward absent/untimed."""
        for nr in list(self.active):
            if nr in present:
                continue
            un_live, un_timed = self.tracks[nr].update(live=False, timed=False)
            if un_live:
                self.active.remove(nr)
                events.append(("absent", nr))
            if un_timed and nr in self.timed:
                self.timed.remove(nr)
                events.append(("untimed", nr))

    def _sighted(self, ordernrs, timed, events):
        """Create new tracks, refresh returning ones."""
        for nr, tm in zip(ordernrs, timed):
            if nr not in self.seen:
                self._create(nr, tm, events)
            else:
                self._refresh(nr, tm, events)

    def _create(self, nr, tm, events):
        self.tracks[nr] = Track(nr, tm)
        self.seen.append(nr)
        self.active.append(nr)
        events.append(("new", nr))
        if tm:
            self.timed.append(nr)
            events.append(("timed", nr))

    def _refresh(self, nr, tm, events):
        _, un_timed = self.tracks[nr].update(live=True, timed=tm)
        if nr not in self.active:
            self.active.append(nr)
            events.append(("present", nr))
        if tm and nr not in self.timed:
            self.timed.append(nr)
            events.append(("timed", nr))
        elif un_timed and nr in self.timed:
            self.timed.remove(nr)
            events.append(("untimed", nr))
