"""Pipeline: frame in, diff out — it never talks to its consumer. Five phases:
identify (segment + ordernr OCR, parallel) -> lifecycle (state.update, events) ->
plan (per order, the fields still unlatched; fully-latched orders cost no OCR) ->
ocr (one pool task per planned segment) -> settle (normalize/bucket/validate, serial).
OCR phases run on a thread pool; tesserocr/cv2 release the GIL, so workers use real cores."""
import os
from concurrent.futures import ThreadPoolExecutor

from .constants import FIELDS, DEFAULT_THRESHOLDS
from .resolve.segmenter import Segmenter
from .ocr import Ocr
from .resolve.normalizer import Normalizer
from .resolve.lifecycle.state import State
from .store.data import Data
from .resolve.validator import Validator

# 2 by default: katana has 4 cores; daemon/ffmpeg and the desktop keep the other two.
WORKERS = int(os.environ.get("ICARUS_OCR_WORKERS", "2"))


class Pipeline:
    def __init__(self, addr, words=None, thresholds=None, workers=WORKERS):
        self.th = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
        self.seg = Segmenter()
        self.ocr = Ocr(words=words)
        self.norm = Normalizer()
        self.state = State()
        self.data = Data(self.th)
        self.valid = Validator(self.th, addr)
        self.pool = ThreadPoolExecutor(max_workers=max(1, workers),
                                       thread_name_prefix="ocr")
        self.prev = {}           # last bundle; empty -> first frame diffs fully

    def run(self, frame_bgr):
        """Raw BGR frame in, diff out. The caller (daemon / sim) decides what to do with it."""
        frame = self.seg.prepare(frame_bgr)
        self.seg.segment(frame)

        # --- 1. identify: ordernr per segment, in parallel ------------------------------
        raws = list(self.pool.map(lambda s: self.ocr.ordernr(s, frame), frame.segments))
        normed = [(self.norm.ordernr(raw), raw, s)
                  for raw, s in zip(raws, frame.segments)]
        admitted = [(nr, s) for nr, _, s in normed if nr is not None]

        # --- 2. lifecycle: create/refresh tracks, collect events ------------------------
        events = self.state.update([nr for nr, _ in admitted],
                                   [s.timed for _, s in admitted])
        for nr, _ in admitted:
            self.data.get_or_create(nr)

        # --- 3. plan: per order, the fields its segment still owes ----------------------
        # {ordernr: (segment, open_fields)} — fully-latched orders drop out here and
        # cost zero OCR for the rest of their life on screen.
        work = {nr: (s, open_) for nr, s in admitted
                if (open_ := [f for f in FIELDS if not self.data.latched(nr)[f]])}

        # --- 4. ocr: one pool task per segment -------------------------------------------
        futures = {nr: self.pool.submit(self.ocr.fields, s, frame,
                                        [f for f in FIELDS if f not in open_])
                   for nr, (s, open_) in work.items()}
        raws_by_nr = {nr: fut.result() for nr, fut in futures.items()}

        # --- 5. settle: normalize -> bucket -> threshold -> validate (serial) ------------
        ocr_fields = {}
        for nr, raws_f in raws_by_nr.items():
            fields = self.norm.fields(raws_f)
            ocr_fields[nr] = (raws_f, fields)
            passed = self.data.add(nr, fields)
            if passed:
                self.valid.validate(self.data, nr, passed)

        bundle = self._bundle(normed, ocr_fields)
        diff = self._diff(bundle, events)
        self.prev = bundle
        return diff

    def _bundle(self, normed, ocr_fields):
        """Per active order: lifecycle state, committed field values, per-field confidences,
        the computed address record (lat/lon/gid), and this frame's raw OCR reads. Rejected
        ordernr reads ride along for debugging. The live countdown is omitted on purpose —
        it would churn the diff every decay frame."""
        rejected = [nr_raw for nr, nr_raw, _ in normed if nr is None and nr_raw is not None]
        out = {}
        for nr in self.state.active:
            t = self.state.tracks[nr]
            rec, addr_latched = self.valid.resolve(self.data, nr)
            raws_f, fields = ocr_fields.get(nr, ({}, []))
            nmap = {f.type: f.value for f in fields}
            out[nr] = {
                "ordernr": nr,
                "timed": nr in self.state.timed,
                "created_at": t.created_at,
                "values": dict(self.data.values[nr]),
                "latches": dict(self.data.latches[nr]),
                "confidences": dict(self.data.confidences[nr]),
                "address": rec,
                "addr_latched": addr_latched,
                # raws are keyed type/addr (addr is one joined read); norms are per field
                "ocr": {"raw": dict(raws_f),
                        "norm": {ft: nmap.get(ft) for ft in FIELDS}},
                "rejected": rejected,
            }
        return out

    def _diff(self, bundle, events):
        """Changed/new active orders, plus lifecycle events overlaid as an `event` tag.
        Absent orders have left the bundle, so they enter the diff purely via their event."""
        diff = {nr: dict(e) for nr, e in bundle.items() if self.prev.get(nr) != e}
        for ev, nr in events:
            entry = diff.setdefault(nr, {"ordernr": nr})
            entry["event"] = ev
        return diff
