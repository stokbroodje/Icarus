"""Segmenter: rotate the POS frame to screen-space, find orders, box each region.
Module-level functions do the work; the Segmenter class is the pipeline's interface."""
import cv2, numpy as np
from dataclasses import dataclass, field

@dataclass
class Box:
    x0:int; y0:int; x1:int; y1:int
    def crop(self, img): return img[self.y0:self.y1, self.x0:self.x1]
    def as_tuple(self): return (self.x0,self.y0,self.x1,self.y1)

@dataclass
class Segment:
    region: Box                  # whole order band
    number: Box                  # digits only (may be None)
    rows:   list                 # content-line Boxes (type, pieces/addr, street, online/timed)
    timed:  bool

@dataclass
class Frame:
    bgr:      np.ndarray         # rotated processing surface (screen-space)
    content:  np.ndarray         # binary "has content" map used by the band finders
    segments: list = field(default_factory=list)

def prepare(raw_bgr):
    bgr  = cv2.rotate(raw_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return Frame(bgr, (gray>60).astype(np.uint8))

def find_strip(content):
    H,W=content.shape; d=content.sum(1)/W; act=d>0.20; ys=np.where(act)[0]
    if not len(ys): return None
    y0=int(ys.min()); y1=y0; gap=0
    for y in range(y0,H):
        if act[y]: y1,gap=y,0
        else:
            gap+=1
            if gap>0.04*H: break
    return y0,y1

def split_bands(content,y0,y1,mhf=0.012):
    H,W=content.shape; d=content[y0:y1+1].sum(1)/W; gut=d<0.15; bands=[]; s=None; mh=mhf*H
    for i,g in enumerate(gut):
        y=y0+i
        if not g and s is None: s=y
        elif g and s is not None:
            if y-s>mh: bands.append((s,y-1)); s=None
    if s is not None and y1-s>mh: bands.append((s,y1))
    return bands

def is_real_order(b):
    hsv=cv2.cvtColor(b,cv2.COLOR_BGR2HSV)
    return 0.05<((hsv[...,2]>235)&(hsv[...,1]<25)).mean()<0.97

def is_separator(b):
    hsv=cv2.cvtColor(b,cv2.COLOR_BGR2HSV); h,s,v=hsv[...,0],hsv[...,1],hsv[...,2]
    return ((h>=85)&(h<=115)&(s>60)&(v>70)).mean()>0.40

def header_width(b):
    hsv=cv2.cvtColor(b,cv2.COLOR_BGR2HSV); cw=((hsv[...,2]>235)&(hsv[...,1]<25)).mean(0)
    for x in range(b.shape[1]):
        if cw[x]>0.60: return max(x,1)
    return b.shape[1]

def header_is_orange(bgr):
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = (int(np.median(hsv[..., i])) for i in range(3))
    return (15 <= h <= 25) and s > 150 and v > 150

def text_mask(bgr):
    """Colour-aware: white text on green/red/gray, black text on orange. Returns 0/255."""
    g=cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(int); bg=np.median(g)
    if header_is_orange(bgr):
        b,r=bgr[...,0].astype(int),bgr[...,2].astype(int)
        m=(g<bg-45)&(~(b-r>15))                 # black on orange, exclude bluish gutter
    else:
        m=(g>bg+45)                             # white on green/red/gray
    return m.astype(np.uint8)*255

def split_lines(band,hw,max_indent=0.20):
    t=text_mask(band[:,:hw]); t=cv2.morphologyEx(t,cv2.MORPH_OPEN,np.ones((2,2),np.uint8))
    rows=t.sum(1)>(2*255); H=t.shape[0]; lines=[]; s=None
    for y,on in enumerate(rows):
        if on and s is None: s=y
        elif not on and s is not None:
            if y-s>0.03*H: lines.append((s,y-1)); s=None
    if s is not None: lines.append((s,H-1))
    if lines:
        med=np.median([b-a+1 for a,b in lines])
        lines=[(a,b) for a,b in lines if not((a<=1 or b>=H-2) and (b-a+1)<0.5*med)]
    out=[]
    for y0,y1 in lines:
        xs=np.where(t[y0:y1+1].any(0))[0]
        if len(xs) and xs.min()<=max_indent*hw: out.append((y0,y1))
    return out

def number_box(band, hw):
    """Number = leftmost run of digit-shaped components in the header's top-left.
    Shape/gap-filtered so the separator bar, clock and '>>' chevron are excluded."""
    H = band.shape[0]; xw = max(1, int(0.60 * hw))
    t = text_mask(band[:, :xw]); t = cv2.morphologyEx(t, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    t[(t > 0).mean(1) > 0.85, :] = 0                                  # strip full-width border rows
    n, _, stats, _ = cv2.connectedComponentsWithStats((t > 0).astype(np.uint8), 8)
    C = [tuple(int(v) for v in stats[i][:4]) for i in range(1, n)
         if stats[i][4] >= 10 and stats[i][3] <= 0.75 * H]
    if not C: return None
    topy = min(c[1] for c in C)
    dh = max(c[3] for c in C if c[1] <= topy + 8)                     # digit height = tallest top-row comp
    row = [c for c in C if c[1] <= topy + 0.5 * dh and c[3] >= 0.6 * dh and c[2] >= 0.20 * c[3]]
    if not row: return None
    row.sort(key=lambda c: c[0])
    keep = [row[0]]
    for c in row[1:]:
        prev = keep[-1]
        if c[0] - (prev[0] + prev[2]) <= 0.20 * dh and len(keep) < 6:  # digit gaps <=4, chevron gaps >=6
            keep.append(c)
        else:
            break
    x0 = min(c[0] for c in keep); x1 = max(c[0] + c[2] for c in keep)
    y0 = min(c[1] for c in keep); y1 = max(c[1] + c[3] for c in keep)
    return (x0, y0, x1 - 1, y1)

def get_segments(frame, toppad_row=5, botpad_row=2, npad=3):
    st=find_strip(frame.content)
    if not st: return []
    bands=split_bands(frame.content,*st)
    seps=[(r0,r1) for (r0,r1) in bands if is_separator(frame.bgr[r0:r1+1])]
    sep=seps[1] if len(seps)>=2 else None
    segs=[]
    for (r0,r1) in bands:
        band=frame.bgr[r0:r1+1]
        if is_separator(band) or not is_real_order(band): continue
        H=band.shape[0]; hw=header_width(band); lines=split_lines(band,hw); t=text_mask(band[:,:hw])
        nb=number_box(band,hw)
        if nb:
            nx0,ny0,nx1,ny1=nb; p=max(2,int(0.18*(ny1-ny0)))
            number=Box(max(0,nx0-p), max(0,r0+ny0-npad), min(hw,nx1+1), r0+ny1+npad)
        else:
            number=None
        rows=[]
        for (y0,y1) in lines[1:]:
            xs=np.where(t[y0:y1+1].any(0))[0]
            x0=max(0,int(xs.min())-2) if len(xs) else 0
            rows.append(Box(x0, r0+max(0,y0-toppad_row), hw-2, r0+min(H,y1+botpad_row)))
        segs.append(Segment(Box(0,r0,band.shape[1],r1), number, rows,
                            sep is not None and r0>=sep[1]))
    frame.segments=segs
    return segs

def trim_right(bgr, threshold=10, margin=10):
    """Trim right side until a significant column-average change is found."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(float)
    col_means = gray.mean(axis=0)
    diffs = np.abs(np.diff(col_means))
    for x in range(len(diffs) - 1, 0, -1):
        if diffs[x] > threshold:
            cut = min(x + 1 + margin, bgr.shape[1])
            return bgr[:, :cut]
    return bgr

def trim_left(bgr, dark_threshold=50, min_dark_ratio=0.4, margin=0):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    for x in range(gray.shape[1]):
        if (gray[:, x] > dark_threshold).mean() > (1 - min_dark_ratio):
            cut = max(x - margin, 0)
            return bgr[:, cut:]
    return bgr

def remove_isolated(mask, min_neighbors=2):
    binary = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    neighbor_count = cv2.filter2D(binary, -1, kernel)
    neighbor_count = neighbor_count - binary
    return ((binary > 0) & (neighbor_count >= min_neighbors)).astype(np.uint8) * 255


class Segmenter:
    """Frame in, segments out. prepare() builds the screen-space Frame (carries .bgr);
    segment() populates and returns its order Segments (each with .number, .rows, .timed)."""
    def prepare(self, frame_bgr):
        return prepare(frame_bgr)

    def segment(self, frame):
        get_segments(frame)
        return frame.segments
