"""OCR readers: mask + OCR per field, raw strings out — normalization/CSV matching live
downstream. Address fields run two reads (letters: street/postcode, digits: house) over
per-pass prepped strips; type_read snaps against
TYPES only to rank rows, returning raw text. Engine is tesserocr in-process, hard
requirement (no CLI fallback); TESSDATA_PREFIX must point at apt tessdata
(Dockerfile sets it; host default /usr/share/tesseract-ocr/5/tessdata)."""
import os
import difflib
import threading
import cv2
import numpy as np

import tesserocr
from PIL import Image

from .resolve import segmenter as seg
from .constants import TYPES

TYPE_SCAN_ROWS = 3
TYPE_STRONG = 0.85

# caps + digits + accented caps + structural punctuation. The punctuation/accents aren't kept —
# they give tesseract a correct bin so it doesn't hallucinate a digit where a '(' or 'é' sits.
# The Normalizer strips all of it. On the lib path this is passed as an API variable.
_ADDR_WL = ("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "ÀÁÂÄÇÈÉÊËÍÏÓÔÖÚÜ '()/=.-")


class _Engine:
    """Resident OCR. Keeps one libtesseract API per (lang, dawg, whitelist, words) and reuses
    it across reads — no per-read process spawn, which was the ~117 ms/call cost that dominated
    the pipeline.

    Thread-safety: the pipeline calls read() from a small thread pool, and a PyTessBaseAPI
    must not be shared between threads — so the API cache is thread-local. Each pool worker
    lazily builds its own API per config the first time it needs it (a few hundred ms, once),
    then reuses it for the life of the process."""

    def __init__(self):
        self._tls = threading.local()
        self._path = os.environ.get("TESSDATA_PREFIX", "")

    def _apis(self):
        apis = getattr(self._tls, "apis", None)
        if apis is None:
            apis = self._tls.apis = {}
        return apis

    def read(self, img, lang="eng", whitelist=None, dawg=True, words=None):
        """One single-line OCR of `img`. `whitelist`/`dawg`/`words` mirror the config flags;
        `words` is a path to a tesseract user-words file biasing recognition toward real tokens."""
        apis = self._apis()
        key = (lang, dawg, whitelist or "", words or "")
        api = apis.get(key)
        if api is None:
            v = {"load_system_dawg": "1" if dawg else "0",
                 "load_freq_dawg":   "1" if dawg else "0",
                 "tessedit_char_whitelist": whitelist or ""}
            if words:
                v["user_words_file"] = words
            api = tesserocr.PyTessBaseAPI(
                path=self._path, lang=lang, psm=tesserocr.PSM.SINGLE_LINE, variables=v)
            apis[key] = api
        api.SetImage(Image.fromarray(img))
        return api.GetUTF8Text()


OCR = _Engine()


# ------------------------------------------------------------------ number box

def number_mask(crop, min_neighbors=2):
    """Binary digit mask for the number box (orange headers inverted first)."""
    orange = seg.header_is_orange(crop)
    src = cv2.bitwise_not(crop) if orange else crop
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    _, b = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if cv2.countNonZero(b) < b.size * 0.5:
        b = cv2.bitwise_not(b)
    return seg.remove_isolated(b, min_neighbors=min_neighbors)


def ordernr_read(order, frame):
    """Raw digit string from the number box, or None. Shape/range judged downstream."""
    if order.number is None:
        return None
    crop = order.number.crop(frame.bgr)
    m = number_mask(crop)
    raw = "".join(c for c in OCR.read(m, lang="eng", whitelist="0123456789") if c.isdigit())
    return raw or None


# ------------------------------------------------------------------ type line

def remove_outlier_pixels(bgr, radius=1, threshold=30, iterations=2):
    """Smooth pixels that differ too much from their neighbourhood average."""
    result = bgr.astype(float)
    for _ in range(iterations):
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.float32)
        kernel /= kernel.sum()
        for c in range(3):
            neighbor_avg = cv2.filter2D(result[..., c], -1, kernel)
            diff = np.abs(result[..., c] - neighbor_avg)
            result[..., c] = np.where(diff > threshold, neighbor_avg, result[..., c])
    return result.astype(np.uint8)


def badge_mask(crop_bgr):
    """Clean white-on-black text mask of a type line."""
    crop = seg.trim_left(crop_bgr)
    crop = seg.trim_right(crop)
    crop = cv2.resize(crop, None, fx=8, fy=8, interpolation=cv2.INTER_CUBIC)
    crop = remove_outlier_pixels(crop, threshold=20, iterations=3)
    m = seg.text_mask(crop)
    m = cv2.erode(m, np.ones((3, 3), np.uint8), iterations=2)
    for _ in range(3):
        m = seg.remove_isolated(m, min_neighbors=3)
    return m


def _snap(text):
    """(best_vocab_word, ratio) for the alpha chars of a read. Row-ranking only — the read
    returned to the pipeline is the raw text, not this match."""
    t = "".join(c for c in text if c.isalpha())
    if not t:
        return None, 0.0
    s = {w: difflib.SequenceMatcher(None, t.lower(), w.lower()).ratio() for w in TYPES}
    best = max(s, key=s.get)
    return best, s[best]


def classify_type(crop_bgr):
    """OCR one type line; return (raw_text, snap_ratio). The ratio ranks rows only."""
    mask = badge_mask(crop_bgr)
    raw = OCR.read(cv2.bitwise_not(mask), lang="eng").strip()
    _, ratio = _snap(raw)
    return raw, round(ratio, 2)


def type_read(order, frame):
    """Raw OCR text of the strongest type row (a timed line can push the type line down)."""
    if not order.rows:
        return None
    best_raw, best_ratio = None, 0.0
    for r in order.rows[:TYPE_SCAN_ROWS]:
        crop = r.crop(frame.bgr)
        if crop.size == 0:
            continue
        raw, ratio = classify_type(crop)
        if ratio > best_ratio:
            best_raw, best_ratio = raw, ratio
        if ratio >= TYPE_STRONG:
            break
    return best_raw


# ------------------------------------------------------------------ address (one joined pass)

def _cell(crop, mode):
    """One address row -> inverted strip cell at NATIVE resolution, ready for _hjoin.
    'mask' keeps the legacy binary (text_mask threshold); 'g4'/'g8' keep grayscale and
    let tesseract binarize the upscaled image itself — a hard threshold here fuses the
    leading '(9' and eats the postcode's first digit."""
    if mode == "mask":
        return 255 - seg.text_mask(crop)
    return 255 - cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)


def _hjoin(imgs, gap=24):
    """Stack row masks side by side on a white field, centred vertically, with a word gap
    between rows so a wrapped postcode/street reads as separate tokens, not one fused word."""
    h = max(i.shape[0] for i in imgs)
    out = []
    for i in imgs:
        d = h - i.shape[0]
        out.append(cv2.copyMakeBorder(i, d // 2, d - d // 2, 0, gap,
                                      cv2.BORDER_CONSTANT, value=255))
    return np.hstack(out)


_PC_WL = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789() "

# Per-pass strip preprocessing, overridable per run via env so you can A/B through
# `docker compose up sim` without editing this file:
#   mask = legacy text_mask threshold + 4x upscale (the old shared prep; the baseline)
#   g4   = grayscale + 4x, tesseract binarizes      (cheaper; house already clean here)
#   g8   = grayscale + 8x                           (sharper; recovers the leading pc digit)
# Letters pass (postcode/street) defaults to g8 — the extra resolution is what stops a
# threshold fusing the '(' into the first digit. Digits/house pass defaults to g4.
LETTERS_MODE = os.environ.get("ICARUS_LETTERS_MODE", "g8")
DIGITS_MODE  = os.environ.get("ICARUS_DIGITS_MODE", "g4")


def _addr_strip(order, frame, mode):
    """The prepared (joined, upscaled, padded) address strip image, or None.
    `mode` is one of 'mask' (legacy: text_mask + 4x), 'g4' (grayscale + 4x) or
    'g8' (grayscale + 8x). Join happens at native res, then a single upscale, so the
    inter-row gap scales with the text and 'mask' reproduces the old prep exactly."""
    crops = [c for r in order.rows[1:] if (c := r.crop(frame.bgr)).size]
    if not crops:
        return None
    fx = 8 if mode == "g8" else 4
    strip = _hjoin([_cell(c, mode) for c in crops])
    strip = cv2.resize(strip, None, fx=fx, fy=fx, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(strip, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)


def addr_read(strip, words=None):
    """Letters pass over the address strip: postcode + street tokens for the Normalizer.
    NOT trusted for the house number — the house digits sit glued behind the postcode's
    ')' and the letter whitelist lets ')1' smear into N/J/T, eating the leading digit."""
    raw = OCR.read(strip, lang="nld", whitelist=_ADDR_WL, words=words, dawg=False).strip()
    return raw or None


def addr_digits_read(strip):
    """Second pass over the SAME strip, caps+digits+parens only — no lowercase, no words
    bias. The pc letters stay letters, so '9736 AE)104' keeps its shape and the Normalizer
    can anchor the house on the full postcode pattern. The Normalizer takes the house
    votes from this read only."""
    raw = OCR.read(strip, lang="eng", whitelist=_PC_WL, dawg=False).strip()
    return raw or None


# ------------------------------------------------------------------ dispatch

class Ocr:
    """Dispatch the readers the pipeline asks for. The address fields (straat/postcode/
    huisnummer) share one `addr_read` pass; `skip` only decides whether that pass runs at all.
    `words` is the tesseract user-words path built from ADDR at pipeline setup."""

    def __init__(self, words=None):
        self.words = words

    def ordernr(self, order, frame):
        return ordernr_read(order, frame)

    def fields(self, order, frame, skip):
        raws = {}
        if "type" not in skip:
            raws["type"] = type_read(order, frame)
        open_addr = {"straat", "postcode", "huisnummer"} - set(skip)
        if not open_addr:
            return raws
        need_letters = bool(open_addr - {"huisnummer"})      # straat or postcode open
        need_digits = "huisnummer" in open_addr              # house open
        if need_letters and need_digits and LETTERS_MODE == DIGITS_MODE:
            strip = _addr_strip(order, frame, LETTERS_MODE)  # identical mode -> one build
            if strip is not None:
                raws["addr"] = addr_read(strip, self.words)
                raws["addr_digits"] = addr_digits_read(strip)
            return raws
        if need_letters:
            strip = _addr_strip(order, frame, LETTERS_MODE)
            if strip is not None:
                raws["addr"] = addr_read(strip, self.words)
        if need_digits:                                      # house comes from digits only
            strip = _addr_strip(order, frame, DIGITS_MODE)
            if strip is not None:
                raws["addr_digits"] = addr_digits_read(strip)
        return raws
