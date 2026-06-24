"""Daemon: the live process. Owns the loop:

    framegrabber.next_frame()  ->  3% pixel-diff gate  ->  pipeline.run()  ->  board

It pulls frames on demand (the grabber blocks between frames, so an idle screen costs
nothing), spaces pipeline runs on a fibonacci budget that resets whenever the screen
moves (see loop), and
translates the pipeline's diff into the discrete events the Flask board consumes
(live / pin / timed / absent — see frontend/app.py's contract). It also POSTs a JPEG of
each processed frame for the board's /latest.jpg preview.

Ownership note: the pipeline does NOT import this module. run() returns the diff and the
daemon decides what happens to it — same diff feeds the sim harness in replay.
"""
import os
import sys
import json
import time
import datetime as dt

import cv2
import requests

from .framegrabber import FrameGrabber
from .pipeline import Pipeline

BOARD = os.environ.get("ICARUS_BOARD", "http://127.0.0.1:5000")
DIFF_GATE = float(os.environ.get("ICARUS_DIFF_GATE", "0.03"))   # 3% of pixels must move
DIFF_STEP = 25                                                  # ...by >25 gray levels
RESET_HOUR = int(os.environ.get("ICARUS_RESET_HOUR", "3"))      # nightly rollover at 03:00
DUMP_DIR = os.environ.get("ICARUS_DUMP_DIR", "dumps")


def _next_reset(hour=RESET_HOUR):
    """Epoch seconds of the next local `hour`:00."""
    now = dt.datetime.now()
    nxt = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if nxt <= now:
        nxt += dt.timedelta(days=1)
    return nxt.timestamp()


class Daemon:
    def __init__(self, addr, words=None):
        self.addr, self.words = addr, words      # kept to rebuild the pipeline nightly
        self.grab = FrameGrabber()
        self.pipe = Pipeline(addr, words=words)
        self.frame_no = 0
        self.pinned = {}          # ordernr -> location identity of the last pin sent
        self.announced = {}       # ordernr -> last type sent in a live event
        self.daylog = {}          # ordernr -> last full bundle entry seen today
        self.next_reset = _next_reset()

    # --- main loop ----------------------------------------------------------
    def loop(self):
        """Fibonacci-spaced processing, reset on motion: runs land 1,2,3,5,8,13,...
        ticks after the last movement. The fib tail (not a hard gate) still catches
        sub-3% changes, just a few ticks late. Idle tick cost: one blocking read +
        a 160x90 absdiff (~1 ms)."""
        a = b = budget = 1
        prev = None
        while True:
            frame = self.grab.next_frame()        # blocks; zero CPU while waiting
            self.frame_no += 1
            if time.time() >= self.next_reset:
                self._rollover()
            sig = self._sig(frame)
            if prev is not None and (cv2.absdiff(sig, prev) > DIFF_STEP).mean() > DIFF_GATE:
                budget, a, b = 1, 1, 1            # movement -> process this frame, restart fib
            prev = sig
            if b == budget:
                diff = self.pipe.run(frame)
                self._log(diff)
                updates = self._events(diff)
                if updates:
                    self._post(updates)
                self._post_frame(frame)
                a, b = b, a + b                   # next run fib-many ticks out
            budget += 1

    # --- day log + nightly rollover -------------------------------------------
    def _log(self, diff):
        """Accumulate the last full state per order; skeleton event entries only tag."""
        for nr, e in diff.items():
            if "values" in e:
                self.daylog[nr] = e
            elif nr in self.daylog and "event" in e:
                self.daylog[nr]["event"] = e["event"]

    def _rollover(self):
        """Nightly: dump the day's orders, then start fresh — new Pipeline (trackers,
        buckets, lifecycle, prev bundle), cleared board memory, next reset armed."""
        try:
            os.makedirs(DUMP_DIR, exist_ok=True)
            stamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
            path = os.path.join(DUMP_DIR, f"orders_{stamp}.json")
            with open(path, "w") as f:
                json.dump(self.daylog, f, indent=1, default=str)
            print(f"[daemon] rollover: {len(self.daylog)} orders -> {path}", flush=True)
        except OSError as exc:
            print(f"[daemon] rollover dump failed (continuing): {exc}", flush=True)
        self.pipe = Pipeline(self.addr, words=self.words)
        self.daylog = {}
        self.pinned = {}
        self.announced = {}
        self.next_reset = _next_reset()

    @staticmethod
    def _sig(frame):
        """160x90 gray signature: cheap enough to compute on every sampled frame."""
        return cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)

    # --- pipeline diff -> board events ----------------------------------------
    def _events(self, diff):
        """Translate one pipeline diff into the board's event vocabulary. The pipeline
        speaks in order state (values/latches/address + lifecycle tags); the board wants
        discrete live/pin/timed/absent events. This is the only place that mapping lives."""
        out = []
        for nr, e in diff.items():
            ev = e.get("event")
            if ev == "absent":
                out.append({"event": "absent", "number": nr})
                self.pinned.pop(nr, None)
                self.announced.pop(nr, None)
                continue                          # absent entries carry no state
            if ev in ("timed", "untimed"):
                out.append({"event": "timed", "number": nr, "timed": ev == "timed"})

            values = e.get("values", {})
            typ = values.get("type")
            # live on first sight and again when the type latches
            if ev in ("new", "present") or (typ and self.announced.get(nr) != typ):
                self.announced[nr] = typ
                out.append({"event": "live", "number": nr, "type": typ,
                            "timed": e.get("timed", False),
                            "created_at": e.get("created_at")})
            # pin on address latch, re-pin on location upgrade (centroid -> exact row).
            # Identity excludes confidence: it creeps every vote and would re-route each frame.
            rec = e.get("address")
            if e.get("addr_latched") and rec \
                    and (rec.get("lat") is not None or rec.get("gid") is not None):
                ident = (rec.get("gid"), rec.get("lat"), rec.get("lon"), rec.get("resolve"))
                if self.pinned.get(nr) != ident:
                    self.pinned[nr] = ident
                    addr_str = " ".join(str(p) for p in
                                        (rec.get("straat"), rec.get("huisnummer"),
                                         rec.get("postcode")) if p is not None)
                    out.append({"event": "pin", "number": nr, "address": addr_str,
                                "lat": rec.get("lat"), "lon": rec.get("lon"),
                                "gid": rec.get("gid"), "kind": rec.get("resolve"),
                                "confidence": rec.get("confidence")})
        return out

    # --- board posts ------------------------------------------------------------
    def _post(self, updates):
        try:
            requests.post(f"{BOARD}/api/orders",
                          json={"frame": self.frame_no, "updates": updates}, timeout=5)
        except requests.RequestException as exc:
            print(f"[daemon] board post failed: {exc}", flush=True)

    def _post_frame(self, frame):
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ok:
            return
        try:
            requests.post(f"{BOARD}/api/frame", data=jpg.tobytes(),
                          headers={"Content-Type": "image/jpeg"}, timeout=5)
        except requests.RequestException as exc:
            print(f"[daemon] frame post failed: {exc}", flush=True)


def main():
    """Entry point: python -m icarus.daemon"""
    from .resolve.data.addr_index import load_addr
    from .resolve.data.wordlist import write_wordlist

    csv_path = os.environ.get("ICARUS_ADDR_CSV",
                              os.path.join(os.path.dirname(__file__),
                                           "resolve", "data", "adressen.csv"))
    # whole-province ADDR index on purpose: deliveries leave the city (Hoogkerk,
    # Zuidwolde, ...) and a town filter would make those streets unresolvable
    addr = load_addr(csv_path)
    words = write_wordlist(addr, "/tmp/icarus_words.txt")
    Daemon(addr, words=words).loop()


if __name__ == "__main__":
    sys.exit(main())
