"""FrameGrabber: pull-based ffmpeg capture. The fps filter throttles output to
ICARUS_FPS, so next_frame() is a blocking read — zero CPU between frames, and frames
are never reused. Dropping is backpressure, not code: a 6 MB raw frame dwarfs the 64 KB
pipe, so a busy pipeline blocks ffmpeg mid-write and the v4l2 ring discards the rest;
the next read is always current. The drain loop covers multi-frame pipes (small sizes).

Env: ICARUS_DEVICE (/dev/video0), ICARUS_W/H (1920x1080), ICARUS_FPS (sample rate, 1),
ICARUS_DEVICE_FPS (pin a rate the card lists — katana: YUYV 1080p @ 10),
ICARUS_INPUT_FORMAT (use yuyv422; MJPG would decode every input frame)."""
import os
import struct
import fcntl
import termios
import subprocess

import numpy as np

DEVICE = os.environ.get("ICARUS_DEVICE", "/dev/video0")
W = int(os.environ.get("ICARUS_W", "1920"))
H = int(os.environ.get("ICARUS_H", "1080"))
FPS = os.environ.get("ICARUS_FPS", "1")
DEVICE_FPS = os.environ.get("ICARUS_DEVICE_FPS")              # e.g. "10"
INPUT_FORMAT = os.environ.get("ICARUS_INPUT_FORMAT")          # e.g. "yuyv422"


class FrameGrabber:
    def __init__(self, device=DEVICE, w=W, h=H, fps=FPS):
        self.w, self.h = w, h
        self.size = w * h * 3                                  # bgr24
        cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error",
               "-f", "v4l2", "-video_size", f"{w}x{h}"]
        if DEVICE_FPS:
            cmd += ["-framerate", DEVICE_FPS]
        if INPUT_FORMAT:
            cmd += ["-input_format", INPUT_FORMAT]
        cmd += ["-i", device,
                "-vf", f"fps={fps}",
                "-pix_fmt", "bgr24", "-f", "rawvideo", "-"]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=0)
        self.fd = self.proc.stdout.fileno()

    # --- io ----------------------------------------------------------------
    def _read_exact(self):
        """Block until one whole frame has been read off the pipe."""
        chunks, need = [], self.size
        while need:
            b = os.read(self.fd, need)
            if not b:
                raise RuntimeError("ffmpeg stream ended (device unplugged / ffmpeg died)")
            chunks.append(b)
            need -= len(b)
        return b"".join(chunks)

    def _buffered(self):
        """Bytes currently sitting unread in the pipe (FIONREAD)."""
        return struct.unpack("i", fcntl.ioctl(self.fd, termios.FIONREAD, b"\0\0\0\0"))[0]

    # --- public ------------------------------------------------------------
    def next_frame(self):
        """Block until the next sampled frame; if a backlog of whole frames exists,
        skip to the newest. Returns an HxWx3 BGR ndarray."""
        buf = self._read_exact()
        while self._buffered() >= self.size:
            buf = self._read_exact()
        return np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)

    def close(self):
        self.proc.kill()
        self.proc.wait()
