import os, glob, sim

frames_glob = os.environ.get("ICARUS_FRAMES", "data/frames/*.png")
range_env   = os.environ.get("ICARUS_RANGE", "0:250")   # matches sim-config
out_name    = os.environ.get("ICARUS_OUT", "sim.json")

# apply range slice to the glob
start, end = (int(x) for x in range_env.split(":"))
all_frames  = sorted(glob.glob(frames_glob))
sliced      = all_frames[start:end]

# write a tmp glob-like list — sim.run accepts a glob string, so we patch around it
# by writing a manifest and pointing ICARUS_FRAMES at a known dir subset isn't clean;
# easier to just call run() with the sliced list directly via its internal path list.
# sim.run() uses glob.glob(frames_glob) internally, so override via env re-point:
import tempfile, pathlib, shutil

tmp = pathlib.Path(tempfile.mkdtemp())
for p in sliced:
    shutil.copy(p, tmp / pathlib.Path(p).name)

os.environ["ICARUS_FRAMES"] = str(tmp / "*.png")
sim.run(out_name=out_name)
