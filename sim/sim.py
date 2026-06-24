"""Sim: replay every frame in ICARUS_FRAMES through the pipeline, oldest first; dump
{frame filename: bundle after that frame} to out/. Segment boxes aren't recorded —
the filename key lets a notebook recompute them (load the png, run the Segmenter).

  docker compose up sim
"""
import os
import sys
import json
import glob
import time
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]      # repo root / container /app
sys.path.insert(0, str(ROOT))
from icarus import Pipeline                              # noqa: E402
from icarus.resolve.data.addr_index import load_addr     # noqa: E402
from icarus.resolve.data.wordlist import write_wordlist  # noqa: E402

SIM_OUT  = ROOT / "out"
FRAMES   = os.environ.get("ICARUS_FRAMES",   str(ROOT / "frames" / "*.png"))
ADDR_CSV = os.environ.get("ICARUS_ADDR_CSV",
                          str(ROOT / "icarus" / "resolve" / "data" / "adressen.csv"))
OUT_NAME   = os.environ.get("ICARUS_OUT", "sim.json")


def run(frames_glob=FRAMES, addr_csv=ADDR_CSV, out_name=OUT_NAME):
    import cv2
    paths = sorted(glob.glob(frames_glob))
    print(f"frames: {len(paths)} matched by {frames_glob}", flush=True)
    if not paths:
        # the usual cause in docker: the host side of the /app/frames bind mount doesn't
        # exist (docker silently mounts an empty dir) or the extension doesn't match
        sys.exit(f"no frames — check the frames volume in docker-compose.yml "
                 f"and ICARUS_FRAMES ({frames_glob})")
    t = time.time()
    # whole-province index: deliveries go beyond the city (Hoogkerk, Zuidwolde, ...)
    addr = load_addr(addr_csv)
    SIM_OUT.mkdir(exist_ok=True)
    words = write_wordlist(addr, str(SIM_OUT / "streetwords.txt"))
    print(f"addr: {len(addr['streets'])} streets loaded ({time.time() - t:.1f}s)", flush=True)

    p = Pipeline(addr, words=words)
    frames, t0 = {}, time.time()
    for i, path in enumerate(paths, 1):
        p.run(cv2.imread(path))
        frames[os.path.basename(path)] = p.prev          # the bundle after this frame
        if i % 25 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {i}/{len(paths)} frames "
                  f"({time.time() - t0:.1f}s)", flush=True)

    dst = SIM_OUT / out_name
    dst.write_text(json.dumps(frames, indent=1, default=str))
    print(f"{len(paths)} frames -> {dst}  ({time.time() - t0:.1f}s)", flush=True)
    return frames


if __name__ == "__main__":
    run()
