"""frontendsim: replay sim.json against the board, standing in for the daemon. Asks for
a start index (1500 -> 001500.png), diffs consecutive bundles into the daemon's event
vocabulary (live/pin/timed/absent), POSTs them, and pushes the frame image when reachable.

    docker compose run --rm frontendsim          # interactive
    python sim/frontendsim.py 1500               # start index as argv

Env: ICARUS_SIM_JSON, ICARUS_BOARD, ICARUS_FRAMES_DIR, ICARUS_REPLAY_DELAY."""
import os
import re
import sys
import json
import time
import pathlib

import requests

SIM_JSON   = os.environ.get("ICARUS_SIM_JSON", "/app/out/sim.json")
BOARD      = os.environ.get("ICARUS_BOARD", "http://app:5000")
FRAMES_DIR = pathlib.Path(os.environ.get("ICARUS_FRAMES_DIR", "/app/sim/frames"))
DELAY      = float(os.environ.get("ICARUS_REPLAY_DELAY", "0.5"))


def _index(name):
    """'001500.png' -> 1500 (first digit run in the filename)."""
    m = re.search(r"\d+", name)
    return int(m.group()) if m else -1


class Replayer:
    """Bundle-to-event translation with the daemon's memory: announced types/timed flags,
    pinned orders, and the previous frame's active set (for absents)."""

    def __init__(self):
        self.prev_active = set()
        self.announced = {}       # nr -> last type sent in a live event
        self.timed = {}           # nr -> last timed flag sent
        self.pinned = {}          # nr -> location identity of the last pin sent

    def events(self, bundle):
        out = []
        active = {int(nr) for nr in bundle}
        for nr in sorted(self.prev_active - active):           # left the board
            out.append({"event": "absent", "number": nr})
            self.announced.pop(nr, None)
            self.timed.pop(nr, None)
            self.pinned.pop(nr, None)
        for nr_s, e in bundle.items():
            nr = int(nr_s)
            typ = (e.get("values") or {}).get("type")
            tm = bool(e.get("timed"))
            if nr not in self.prev_active or self.announced.get(nr, ...) != typ:
                self.announced[nr] = typ                       # first sight / type latched
                self.timed[nr] = tm
                out.append({"event": "live", "number": nr, "type": typ,
                            "timed": tm, "created_at": e.get("created_at")})
            elif self.timed.get(nr) != tm:
                self.timed[nr] = tm
                out.append({"event": "timed", "number": nr, "timed": tm})
            rec = e.get("address")
            if e.get("addr_latched") and rec \
                    and (rec.get("lat") is not None or rec.get("gid") is not None):
                ident = (rec.get("gid"), rec.get("lat"), rec.get("lon"), rec.get("resolve"))
                if self.pinned.get(nr) != ident:    # first pin OR location upgrade
                    self.pinned[nr] = ident
                    addr_str = " ".join(str(p) for p in
                                        (rec.get("straat"), rec.get("huisnummer"),
                                         rec.get("postcode")) if p is not None)
                    out.append({"event": "pin", "number": nr, "address": addr_str,
                                "lat": rec.get("lat"), "lon": rec.get("lon"),
                                "gid": rec.get("gid"), "kind": rec.get("resolve"),
                                "confidence": rec.get("confidence")})
        self.prev_active = active
        return out


def _post(idx, updates):
    requests.post(f"{BOARD}/api/orders",
                  json={"frame": idx, "updates": updates}, timeout=5)


def _post_image(name):
    """Push the frame to /latest.jpg's backing store. PNGs are converted so the board's
    image/jpeg mimetype stays honest; missing files are silently fine."""
    path = FRAMES_DIR / name
    if not path.is_file():
        return
    import cv2
    img = cv2.imread(str(path))
    if img is None:
        return
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    if ok:
        requests.post(f"{BOARD}/api/frame", data=jpg.tobytes(),
                      headers={"Content-Type": "image/jpeg"}, timeout=5)


def main():
    data = json.loads(pathlib.Path(SIM_JSON).read_text())
    keys = sorted(data, key=_index)
    if not keys:
        sys.exit(f"{SIM_JSON} holds no frames")

    if len(sys.argv) > 1:
        start = int(sys.argv[1])
    else:
        first, last = _index(keys[0]), _index(keys[-1])
        start = int(input(f"start frame index ({first}..{last}): ") or first)

    todo = [k for k in keys if _index(k) >= start]
    if not todo:
        sys.exit(f"no frames at index >= {start}")
    print(f"replaying {len(todo)} frames ({todo[0]} .. {todo[-1]}) -> {BOARD}", flush=True)

    rp = Replayer()
    for k in todo:
        updates = rp.events(data[k])
        if updates:
            _post(_index(k), updates)
            print(f"{k}: {[u['event'] + ':' + str(u['number']) for u in updates]}", flush=True)
        _post_image(k)
        time.sleep(DELAY)
    print("replay done", flush=True)


if __name__ == "__main__":
    main()
