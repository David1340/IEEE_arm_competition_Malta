"""Step 01: label large red speed-climbing handholds.

Run: python main.py
Two options: Manual / Model-assisted. Works without a trained model.
Input images: data/images; YOLO polygon labels: data/labels.
Model-assisted: choose YOLO-seg masks or YOLO boxes -> SAM 2 masks.
SAM weights: data/sam2_t.pt (optional; auto-download if missing).
"""
from __future__ import annotations

import math
import os
import queue
import shutil
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

try:
    from PIL import Image, ImageTk, ImageOps
except ImportError:
    raise SystemExit("Pillow missing. Run: python -m pip install pillow")

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
IMAGES = DATA / "images"
LABELS = DATA / "labels"
ORIGINALS = DATA / "originals"  # automatic backups before changing image orientation
WEIGHTS = DATA / "best.pt"
SAM_WEIGHTS = DATA / "sam2_t.pt"   # optional: Ultralytics downloads tiny SAM 2 on first use
SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def polygon_area(poly):
    return abs(sum(a[0] * b[1] - b[0] * a[1]
                   for a, b in zip(poly, poly[1:] + poly[:1]))) / 2.0


def point_inside(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i, (xi, yi) in enumerate(poly):
        xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and x < (xj - xi) * (y - yi) / (yj - yi + 1e-20) + xi:
            inside = not inside
        j = i
    return inside


def segment_distance(pt, a, b):
    vx, vy = b[0] - a[0], b[1] - a[1]
    denom = vx * vx + vy * vy
    if denom < 1e-12:
        return math.dist(pt, a)
    t = min(1., max(0., ((pt[0] - a[0]) * vx + (pt[1] - a[1]) * vy) / denom))
    return math.dist(pt, (a[0] + t * vx, a[1] + t * vy))


def simplify(points, tolerance):
    """Ramer-Douglas-Peucker; accepts a closed contour without a repeated final point."""
    if len(points) <= 3:
        return points

    def rdp(p):
        if len(p) < 3:
            return p
        start, end = p[0], p[-1]
        distances = [segment_distance(pt, start, end) for pt in p[1:-1]]
        if not distances or max(distances) <= tolerance:
            return [start, end]
        idx = distances.index(max(distances)) + 1
        return rdp(p[:idx+1])[:-1] + rdp(p[idx:])

    # Choose a distant splitting point to avoid the coincident-endpoint issue.
    far = max(range(1, len(points)), key=lambda i: math.dist(points[i], points[0]))
    lhs = rdp(points[:far+1])
    rhs = rdp(points[far:] + [points[0]])
    result = lhs[:-1] + rhs[:-1]
    return result if len(result) >= 3 else points


def source_files(folder):
    if not folder.is_dir():
        return []
    # Images in one folder; no ambiguous duplicate basenames from nested folders.
    return sorted((p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in SUFFIXES),
                  key=lambda p: p.name.casefold())



def transform_points(points, orientation, width, height):
    """Map polygon pixels when applying an EXIF orientation / manual rotation.

    Coordinates refer to actual image pixels, not to a transformed preview.
    Orientations 2–8 follow the EXIF standard. Manual rotations use 6/8.
    """
    def one(x, y):
        if orientation == 2: return width - 1 - x, y
        if orientation == 3: return width - 1 - x, height - 1 - y
        if orientation == 4: return x, height - 1 - y
        if orientation == 5: return y, x
        if orientation == 6: return height - 1 - y, x
        if orientation == 7: return height - 1 - y, width - 1 - x
        if orientation == 8: return y, width - 1 - x
        return x, y
    return [one(float(x), float(y)) for x, y in points]


def backup_once(source, label=None):
    """Copy an original photo (and existing annotations) before reorienting it."""
    ORIGINALS.mkdir(parents=True, exist_ok=True)
    dest = ORIGINALS / source.name
    if not dest.exists():
        shutil.copy2(source, dest)
    if label and label.is_file():
        dest_label = ORIGINALS / (label.stem + ".labels_before_rotation.txt")
        if not dest_label.exists():
            shutil.copy2(label, dest_label)


def save_image_atomic(image, path):
    """Save upright image bytes, then replace the original only after saving."""
    temp = path.with_name("." + path.stem + "_upright" + path.suffix)
    options = {}
    if path.suffix.lower() in (".jpg", ".jpeg"):
        options = {"quality": 95, "subsampling": 0}
    elif path.suffix.lower() == ".webp":
        options = {"quality": 95}
    try:
        image.save(temp, **options)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def transform_yolo_file(path, orientation, old_size, new_size):
    """Transform previously saved labels together with image pixels.

    Existing labels from the old sideways-image GUI remain attached to the
    same physical holds. Non-YOLO rows are preserved verbatim rather than lost.
    """
    if not path.is_file():
        return
    w, h = old_size
    new_w, new_h = new_size
    new_lines = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        parts = line.split()
        if len(parts) < 7 or len(parts) % 2 != 1:
            new_lines.append(line)
            continue
        try:
            coords = [float(v) for v in parts[1:]]
            old_pts = [(coords[i] * w, coords[i + 1] * h)
                       for i in range(0, len(coords), 2)]
            rotated = transform_points(old_pts, orientation, w, h)
            items = [parts[0]]
            for x, y in rotated:
                items.extend((f"{max(0, min(1, x / new_w)):.6f}",
                              f"{max(0, min(1, y / new_h)):.6f}"))
            new_lines.append(" ".join(items))
        except ValueError:
            new_lines.append(line)
    path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""), encoding="utf-8")


def load_upright_photo(path, label_file):
    """Read a photo as the user sees it, and normalise its pixel orientation.

    Save corrected pixels into data/images so any future model/training tool
    sees the same orientation as the annotation GUI. Originals are backed up.
    """
    with Image.open(path) as source:
        orientation = int(source.getexif().get(274, 1) or 1)
        before_size = source.size
        if orientation == 1:
            return source.convert("RGB"), False
        corrected = ImageOps.exif_transpose(source).convert("RGB")
    backup_once(path, label_file)
    # Perform the label transform before finalising the photo. The backup
    # ensures both original photo and original labels remain recoverable.
    transform_yolo_file(label_file, orientation, before_size, corrected.size)
    save_image_atomic(corrected, path)
    return corrected, True


def detect_yolo_candidates(image, detector, confidence=0.23, tile_size=768, overlap=0.25):
    """Run YOLO once per tile. Retain BOTH its bounding boxes and segmentation masks.

    Coordinates are in the complete upright photograph; duplicate overlapping
    detections from tiles are suppressed before making annotation proposals.
    """
    import numpy as np
    w, h = image.size
    stride = max(1, int(tile_size * (1. - overlap)))
    def positions(n):
        if n <= tile_size: return [0]
        seq = list(range(0, n-tile_size+1, stride))
        if seq[-1] != n-tile_size: seq.append(n-tile_size)
        return seq
    candidates = []
    for y in positions(h):
        for x in positions(w):
            bgr = np.asarray(image.crop((x,y,min(x+tile_size,w),min(y+tile_size,h))))[:,:,::-1].copy()
            result = detector.predict(source=bgr, conf=confidence, imgsz=tile_size,
                                      retina_masks=True, verbose=False)[0]
            if result.boxes is None: continue
            masks = result.masks.xy if result.masks is not None else []
            for i, box in enumerate(result.boxes):
                if hasattr(box, 'cls') and int(box.cls[0]) != 0: continue
                rect = box.xyxy[0].cpu().numpy().astype(float)
                rect += [x,y,x,y]
                rect[0::2] = np.clip(rect[0::2],0,w-1)
                rect[1::2] = np.clip(rect[1::2],0,h-1)
                if (rect[2]-rect[0])*(rect[3]-rect[1]) < 35: continue
                poly = None
                if i < len(masks) and len(masks[i]) >= 3:
                    raw = [(float(px+x),float(py+y)) for px,py in masks[i]]
                    cleaned = polish_outline(raw,scale=1.,closed=True,smooth=False)
                    if (len(cleaned)>=3 and polygon_area(cleaned)>=25
                            and not has_self_intersection(cleaned)):
                        poly = cleaned
                candidates.append(dict(score=float(box.conf[0]),box=rect.tolist(),polygon=poly))
    kept=[]
    for cand in sorted(candidates,key=lambda item:-item['score']):
        bb=cand['box']
        cx,cy=(bb[0]+bb[2])/2,(bb[1]+bb[3])/2
        if any(bbox_iou(bb,other['box'])>0.38 or
               (abs(cx-(other['box'][0]+other['box'][2])/2)<
                     min(bb[2]-bb[0],other['box'][2]-other['box'][0])*.35
                and abs(cy-(other['box'][1]+other['box'][3])/2)<
                     min(bb[3]-bb[1],other['box'][3]-other['box'][1])*.35)
               for other in kept):
            continue
        kept.append(cand)
    return kept


def detect_boxes(image, detector, confidence=0.23, tile_size=768, overlap=0.25):
    """Compatibility helper: boxes only (SAM prompts)."""
    return [item['box'] for item in detect_yolo_candidates(
        image,detector,confidence,tile_size,overlap)]


def sam_mask_for_box(image, sam, box):
    """Crop around the proposed box, prompt SAM 2, return ONE cleaned outline.

    The crop gives small holds more SAM input pixels; the box prompt remains
    inside this crop. Return coordinates in the full original image.
    """
    import numpy as np
    import cv2
    w,h=image.size
    x0,y0,x1,y1=[float(z) for z in box]
    bw,bh=x1-x0,y1-y0
    pad=max(16.,.35*max(bw,bh))
    left=max(0,int(x0-pad)); top=max(0,int(y0-pad))
    right=min(w,int(x1+pad+1));bottom=min(h,int(y1+pad+1))
    if right-left<8 or bottom-top<8:return None
    crop=np.array(image.crop((left,top,right,bottom)).convert('RGB'))[:,:,::-1].copy()
    prompt=[x0-left,y0-top,x1-left,y1-top]
    results=sam.predict(source=crop,bboxes=prompt,retina_masks=True,verbose=False)
    if not results or results[0].masks is None:return None
    polys=results[0].masks.xy
    centre=((x0+x1)/2-left,(y0+y1)/2-top)
    best=None
    best_score=float('-inf')
    for pts in polys:
        if len(pts)<3:continue
        contour=np.asarray(pts,np.float32).reshape(-1,1,2)
        area=abs(cv2.contourArea(contour))
        if area<max(25,.10*bw*bh) or area>4.5*bw*bh:continue
        inside=cv2.pointPolygonTest(contour,centre,False)>=0
        # Prefer the prompted object's region over disconnected noise.
        score=(100 if inside else 0)-abs(math.log(max(area,1)/(bw*bh)))*9
        if score>best_score:
            best_score=score
            best=contour
    if best is None:return None
    perimeter=cv2.arcLength(best,True)
    # Real contour geometry, reduced to manageable editable vertices.
    approx=cv2.approxPolyDP(best,max(.9,.002*perimeter),True).reshape(-1,2)
    pts=[(float(pt[0]+left),float(pt[1]+top)) for pt in approx]
    pts=polish_outline(pts,scale=1.,closed=True,smooth=True)
    if len(pts)<3 or has_self_intersection(pts):return None
    return pts


def model_polygons(image, detector, sam=None, notify=None, method='sam'):
    """Propose polygons using YOLO's own masks OR YOLO boxes prompted through SAM.

    Both choices use the same detection boxes, and neither modifies saved labels.
    """
    if method not in ('sam','yolo'):
        raise ValueError('Unknown model proposal method: '+str(method))
    if method=='yolo' and getattr(detector,'task','segment')!='segment':
        raise RuntimeError('YOLO segmentation requires a -seg model in data/best.pt. '
                           'Switch to YOLO boxes + SAM for detection-only weights.')
    if method=='sam' and sam is None:
        raise RuntimeError('SAM model is not loaded.')
    candidates=detect_yolo_candidates(image,detector)
    proposals=[]
    mask_count=0
    for i,item in enumerate(candidates):
        if notify: notify(i+1,len(candidates))
        poly=item['polygon'] if method=='yolo' else sam_mask_for_box(image,sam,item['box'])
        if method=='yolo' and poly is not None: mask_count+=1
        if poly is None:continue
        if conflicts_with_existing(poly,proposals,threshold=.60):continue
        proposals.append(poly)
    if method=='yolo' and candidates and mask_count==0:
        raise RuntimeError('This YOLO checkpoint has boxes but no segmentation masks. '
                           'Use YOLO boxes + SAM or install your YOLO26n-seg weights.')
    return proposals


def bbox_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    aa = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    bb = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return inter / max(1e-12, aa + bb - inter)


def polygon_bbox(points):
    return [min(p[0] for p in points), min(p[1] for p in points),
            max(p[0] for p in points), max(p[1] for p in points)]



def resample_path(points, step, closed=False, max_points=180):
    """Evenly-spaced points without hundreds of tiny overlapping handles."""
    if len(points)<2:return list(points)
    path=[tuple(map(float,p)) for p in points]
    if closed:path.append(path[0])
    lengths=[math.dist(a,b) for a,b in zip(path,path[1:])]
    total=sum(lengths)
    if total<1e-6:return list(points)
    n=min(max_points,max(3 if closed else 2,int(math.ceil(total/max(1.,step)))))
    marks=[i*total/n for i in range(n)] if closed else [i*total/(n-1) for i in range(n)]
    out=[]; j=0;walk=0.
    for t in marks:
        while j<len(lengths)-1 and walk+lengths[j]<t:
            walk+=lengths[j];j+=1
        fraction=min(1.,max(0.,(t-walk)/max(1e-9,lengths[j])))
        a,b=path[j],path[j+1]
        out.append((a[0]*(1-fraction)+b[0]*fraction,a[1]*(1-fraction)+b[1]*fraction))
    return out


def polish_outline(points, scale, closed=True, smooth=True):
    """Gentle denoising, still accurate at the original pixel resolution."""
    pts=[tuple(map(float,p)) for p in points]
    if closed and len(pts)>2 and math.dist(pts[0],pts[-1])<2/max(scale,.1):pts.pop()
    dedup=[]
    for pt in pts:
        if not dedup or math.dist(pt,dedup[-1])>max(.6,1.3/max(scale,.1)):
            dedup.append(pt)
    if len(dedup)<(3 if closed else 2):return dedup
    # Keep fine segmentation without crowding an entire outline with yellow dots.
    pts=resample_path(dedup,max(2.3,3.4/max(scale,.1)),closed=closed,max_points=180)
    if smooth and len(pts)>=6:
        for _ in range(2):
            out=[]
            for i,p in enumerate(pts):
                if not closed and i in (0,len(pts)-1):out.append(p);continue
                before,after=pts[(i-1)%len(pts)],pts[(i+1)%len(pts)]
                alpha=.13
                out.append((p[0]*(1-alpha)+(before[0]+after[0])*alpha/2,
                            p[1]*(1-alpha)+(before[1]+after[1])*alpha/2))
            pts=out
    return pts


def nearest_edge(poly, point):
    """Distance, projected point, edge index, arclength from vertex zero."""
    best = (float('inf'), None, None, 0.)
    walked = 0.
    for i, a in enumerate(poly):
        b = poly[(i+1)%len(poly)]
        dx, dy = b[0]-a[0], b[1]-a[1]
        denom = dx*dx+dy*dy
        t = min(1., max(0., ((point[0]-a[0])*dx+(point[1]-a[1])*dy)/denom)) if denom else 0.
        q = (a[0]+t*dx, a[1]+t*dy)
        dist = math.dist(point, q)
        if dist < best[0]:
            best = (dist, q, i, walked+t*math.sqrt(denom))
        walked += math.sqrt(denom)
    return best


def arc_points(poly, start_distance, end_distance):
    """Inclusive, correctly ordered arc in the *forward* polygon direction."""
    n=len(poly)
    lengths=[math.dist(poly[i],poly[(i+1)%n]) for i in range(n)]
    total=sum(lengths)
    if total<1e-8:raise ValueError('Degenerate polygon')
    a=start_distance%total
    b=end_distance%total
    if b <= a:b+=total
    def point_at(d):
        d=d%total;progress=0.
        for i,length in enumerate(lengths):
            if d<=progress+length+1e-9:
                t=max(0.,min(1.,(d-progress)/max(length,1e-12)))
                p,q=poly[i],poly[(i+1)%n]
                return (p[0]*(1-t)+q[0]*t,p[1]*(1-t)+q[1]*t)
            progress+=length
        return tuple(poly[0])
    out=[point_at(a)]
    # IMPORTANT: wrapped vertices must be in arclength order, not original
    # vertex-index order. Otherwise a second redraw stitches the complement
    # in the wrong order and creates huge crossing triangles.
    events=[]
    progress=0.
    for i,length in enumerate(lengths):
        progress+=length
        for offset in (0.,total):
            if a+1e-7 < progress+offset < b-1e-7:
                events.append((progress+offset,tuple(poly[(i+1)%n])))
    out.extend(p for _,p in sorted(events,key=lambda it:it[0]))
    out.append(point_at(b))
    return out


def splice_boundary(poly, stroke, replace='shorter'):
    """Replace either side of a closed boundary with a traced stroke.

    Both endpoints snap onto original edges. The user explicitly previews and
    chooses which old arc is replaced; long arcs are supported too.
    """
    if len(stroke)<3 or len(poly)<3:return None
    _,start,_,s0=nearest_edge(poly,stroke[0])
    _,end,_,s1=nearest_edge(poly,stroke[-1])
    if math.dist(start,end)<3.:return None
    perimeter=sum(math.dist(poly[i],poly[(i+1)%len(poly)]) for i in range(len(poly)))
    forward=(s1-s0)%perimeter
    if min(forward,perimeter-forward)<3.:return None
    trace=[start]+[tuple(p) for p in stroke[1:-1]]+[end]
    short_is_forward=forward<=perimeter/2
    replace_forward=short_is_forward if replace=='shorter' else not short_is_forward
    if replace in ('forward','backward'): replace_forward=replace=='forward'
    if replace_forward:
        # New start->end plus old complement end->start.
        complement=arc_points(poly,s1,s0+perimeter)
        trial=trace+complement[1:-1]
    else:
        # Old forward start->end plus new end->start.
        remaining=arc_points(poly,s0,s1)
        trial=remaining+list(reversed(trace))[1:-1]
    clean=[]
    for p in trial:
        if not clean or math.dist(clean[-1],p)>.35:clean.append(tuple(p))
    if len(clean)>=2 and math.dist(clean[0],clean[-1])<.5:clean.pop()
    return clean if len(clean)>=3 else None


def sampled_open_path(points, count=28):
    """Equally spaced points along an open line, including its endpoints."""
    if len(points) < 2:
        return list(points)
    lengths=[math.dist(a,b) for a,b in zip(points,points[1:])]
    total=sum(lengths)
    if total < 1e-8:
        return [tuple(points[0])]*count
    samples=[]
    j=0
    travelled=0.0
    for k in range(count):
        target=total*k/(count-1)
        while j<len(lengths)-1 and travelled+lengths[j]<target:
            travelled+=lengths[j];j+=1
        t=min(1.,max(0.,(target-travelled)/max(lengths[j],1e-9)))
        a,b=points[j],points[j+1]
        samples.append((a[0]+t*(b[0]-a[0]),a[1]+t*(b[1]-a[1])))
    return samples


def redraw_arc_order(poly, stroke):
    """Pick the existing boundary arc following the user's stroke.

    A closed polygon has TWO arcs between any endpoints. The previous editor
    always chose the shorter one, which could create unwanted straight joins
    when a user retraced half a hold. Instead compare the actual stroke to
    both possible old arcs in drawing order. Each independent edit uses the
    *current* polygon; no previous strokes are retained or joined together.
    """
    if len(poly)<3 or len(stroke)<2:
        return ('shorter','longer')
    _,a,_,start=nearest_edge(poly,stroke[0])
    _,b,_,end=nearest_edge(poly,stroke[-1])
    perimeter=sum(math.dist(poly[i],poly[(i+1)%len(poly)]) for i in range(len(poly)))
    if perimeter<1e-8 or math.dist(a,b)<2:
        return ('shorter','longer')
    forward=arc_points(poly,start,end)
    backward=list(reversed(arc_points(poly,end,start+perimeter)))
    trace=sampled_open_path([a]+list(stroke[1:-1])+[b])
    def distance(arc):
        old=sampled_open_path(arc,len(trace))
        rms=math.sqrt(sum((x-u)**2+(y-v)**2 for (x,y),(u,v) in zip(trace,old))/len(trace))
        old_length=sum(math.dist(x,y) for x,y in zip(arc,arc[1:]))
        new_length=sum(math.dist(x,y) for x,y in zip(stroke,stroke[1:]))
        return rms + .08*abs(old_length-new_length)
    a_score=distance(forward)
    b_score=distance(backward)
    short_is_forward=((end-start)%perimeter)<=perimeter/2
    preferred_forward = a_score<=b_score
    primary='shorter' if preferred_forward==short_is_forward else 'longer'
    secondary='longer' if primary=='shorter' else 'shorter'
    return primary,secondary


def replace_local_boundary(poly, stroke, validator=None):
    """Return an updated SINGLE closed hold outline or an explanatory error.

    Reject both candidates when they self-cross, nearly collapse or grow
    implausibly; never return a disconnected line/second mask.
    """
    if len(stroke)<3 or len(poly)<3:
        return None, 'Trace a longer piece of the boundary.'
    start_dist,start,_,_=nearest_edge(poly,stroke[0])
    end_dist,end,_,_=nearest_edge(poly,stroke[-1])
    if math.dist(start,end)<3:
        return None, 'Start and end must be different places on the boundary.'
    old_area=polygon_area(poly)
    # Redraw path endpoints are anchored to the CURRENT closed outline.
    trace=[start]+list(stroke[1:-1])+[end]
    options=redraw_arc_order(poly,trace)
    failures=[]
    for which in options:
        candidate=splice_boundary(poly,trace,replace=which)
        if not candidate or has_self_intersection(candidate):
            failures.append('crossing boundary')
            continue
        area=polygon_area(candidate)
        if area<25 or (old_area>25 and not .32 <= area/old_area <= 2.5):
            failures.append('unexpected area change')
            continue
        if validator:
            problem=validator(candidate)
            if problem:
                failures.append(problem)
                continue
        return candidate,None
    return None, 'Neither boundary replacement is valid ('+', '.join(dict.fromkeys(failures))+').'


def polygon_centre(poly):
    """Area-centroid with bbox fallback, for picking multiple masks."""
    import cv2, numpy as np
    arr=np.asarray(poly,np.float32)
    mm=cv2.moments(arr)
    if abs(mm['m00'])>1e-9:return mm['m10']/mm['m00'],mm['m01']/mm['m00']
    return sum(x for x,y in poly)/len(poly),sum(y for x,y in poly)/len(poly)


def masks_in_rectangle(polys, box):
    x0,y0,x1,y1=box
    l,r=sorted((x0,x1));t,b=sorted((y0,y1))
    return [i for i,poly in enumerate(polys) if l<=polygon_centre(poly)[0]<=r and
            t<=polygon_centre(poly)[1]<=b]


def has_self_intersection(poly):
    """Reject crossed/touching non-neighbour segments and backtracking edges.

    Tkinter fills malformed paths unpredictably (giant triangles / missing fill).
    Consecutive shared endpoints are allowed; all other crossings are not.
    """
    if len(poly)<4:return False
    def orient(a,b,c):
        return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
    def within(a,b,c):
        return (min(a[0],b[0])-1e-7<=c[0]<=max(a[0],b[0])+1e-7 and
                min(a[1],b[1])-1e-7<=c[1]<=max(a[1],b[1])+1e-7)
    def intersects(a,b,c,d):
        o1,o2,o3,o4=orient(a,b,c),orient(a,b,d),orient(c,d,a),orient(c,d,b)
        if (o1>1e-7 and o2<-1e-7 or o1<-1e-7 and o2>1e-7) and (o3>1e-7 and o4<-1e-7 or o3<-1e-7 and o4>1e-7):return True
        return (abs(o1)<1e-7 and within(a,b,c) or abs(o2)<1e-7 and within(a,b,d) or
                abs(o3)<1e-7 and within(c,d,a) or abs(o4)<1e-7 and within(c,d,b))
    n=len(poly)
    for i in range(n):
        a,b=poly[i],poly[(i+1)%n]
        if math.dist(a,b)<.1:return True
        for j in range(i+2,n):
            if (j+1)%n==i:continue
            c,d=poly[j],poly[(j+1)%n]
            if max(a[0],b[0])+1e-7<min(c[0],d[0]) or max(c[0],d[0])+1e-7<min(a[0],b[0]):continue
            if max(a[1],b[1])+1e-7<min(c[1],d[1]) or max(c[1],d[1])+1e-7<min(a[1],b[1]):continue
            if intersects(a,b,c,d):return True
    return False


def overlapping_fraction(a, b):
    """Polygon intersection divided by smaller area, not bounding-box IoU.

    Renders only a small temporary bitmap for comparing two shapes; no external
    geometry library or new dependencies are required.
    """
    from PIL import ImageDraw, ImageChops, ImageStat
    aa,bb=polygon_bbox(a),polygon_bbox(b)
    if aa[2]<=bb[0] or bb[2]<=aa[0] or aa[3]<=bb[1] or bb[3]<=aa[1]:
        return 0.
    x0,y0=min(aa[0],bb[0]),min(aa[1],bb[1])
    x1,y1=max(aa[2],bb[2]),max(aa[3],bb[3])
    mul = 240/max(x1-x0,y1-y0,1.)
    def mask(poly):
        im=Image.new('L',(max(1,int((x1-x0)*mul)+3),max(1,int((y1-y0)*mul)+3)),0)
        ImageDraw.Draw(im).polygon([((x-x0)*mul+1,(y-y0)*mul+1) for x,y in poly],fill=255)
        return im
    ma,mb=mask(a),mask(b)
    sa=ImageStat.Stat(ma).sum[0]
    sb=ImageStat.Stat(mb).sum[0]
    if min(sa,sb)<1:
        return 0.
    inter=ImageStat.Stat(ImageChops.multiply(ma,mb)).sum[0]
    return float(inter/min(sa,sb))


def conflicts_with_existing(poly, saved, skip=None, threshold=.42):
    """Prevent near-duplicate masks while retaining distinct adjacent holds."""
    return any(i != skip and overlapping_fraction(poly, other) > threshold
               for i,other in enumerate(saved))


class Annotator:
    def __init__(self, root, files):
        self.root = root
        self.files = files
        self.root.title("Speed hold annotation | Local YOLO segmentation")
        self.root.geometry("1450x880")
        self.root.minsize(950, 570)
        IMAGES.mkdir(parents=True, exist_ok=True)
        LABELS.mkdir(parents=True, exist_ok=True)

        self.keys = [p.name for p in files]

        self.index = -1
        self.image = None
        self.image_path = None
        self.image_size = (1, 1)
        self.polys = []
        self.draft = []
        self.trace = False
        self.selected = None
        self.mode = "polygon"
        self.drag = None
        self.pending_redraw = None  # candidate, never saved until confirmed
        self.box_rect = None
        self.sam = None
        self.selected_vertex = None
        self.selected_edge = None
        self.selected_edge_position = None
        self.just_selected = False
        self.history = []
        self.shift_pan = False
        self.space_down = False
        self.scale = 1.0
        self.origin = [0.0, 0.0]
        self.tk_image = None
        self.dirty = False
        self.suggestions = []       # orange previews; never saved until accepted
        self.assisted = False
        self.assist_var = tk.StringVar(value="manual")
        self.proposal_method_var = tk.StringVar(value="YOLO boxes + SAM")
        self.model = None
        self.model_busy = False
        self.model_events = queue.Queue()
        self.model_job_counter = 0

        self.mode_var = tk.StringVar(value="polygon")
        self.status = tk.StringVar(value="Loading photographs...")
        self.info = tk.StringVar(value="")
        self._make_ui()
        self._bind()
        self.root.after(120, lambda: self.load(0))
        self.root.after(150, self._poll_model_events)

    def _make_ui(self):
        choice = ttk.Frame(self.root, padding=(8, 7, 8, 3))
        choice.pack(fill="x")
        ttk.Label(choice, text="Labelling:").pack(side="left", padx=(0, 7))
        ttk.Radiobutton(choice, text="Manual", variable=self.assist_var,
                        value="manual", command=lambda: self.set_assisted(False)).pack(side="left", padx=5)
        ttk.Radiobutton(choice, text="Model-assisted", variable=self.assist_var,
                        value="assisted", command=lambda: self.set_assisted(True)).pack(side="left", padx=5)
        ttk.Label(choice, text="Mask source:").pack(side="left",padx=(18,5))
        self.method_choice = ttk.Combobox(choice, textvariable=self.proposal_method_var,
                                         values=("YOLO boxes + SAM", "YOLO segmentation"),
                                         state="readonly",width=21)
        self.method_choice.pack(side="left",padx=2)
        self.method_choice.bind("<<ComboboxSelected>>",lambda event: self.change_proposal_method())
        self.accept_button = ttk.Button(choice, text="Use all new suggestions", command=self.accept_suggestions)
        self.accept_button.pack(side="left", padx=(9, 4))
        ttk.Label(choice, text="Click orange to accept/edit", foreground="#555").pack(side="left",padx=5)
        ttk.Label(choice, text="Weights: data/best.pt", foreground="#666").pack(side="right", padx=8)

        top = ttk.Frame(self.root, padding=7)
        top.pack(fill="x")
        ttk.Label(top, text="Draw/Edit:").pack(side="left", padx=(0, 5))
        for name, title in [("polygon", "Polygon (D)"),
                            ("trace", "Freehand (T)"),
                            ("edit", "Edit (E)"),
                            ("redraw", "Redraw edge (R)"),
                            ("delete_box", "Box delete (B)"),
                            ("pan", "Pan / Pause (P)")]:
            ttk.Radiobutton(top, text=title, variable=self.mode_var, value=name,
                            command=lambda n=name: self.set_mode(n)).pack(side="left", padx=4)
        ttk.Separator(top, orient="vertical").pack(side="left", fill="y", padx=7)
        ttk.Button(top, text="Finish hold (Enter)", command=self.finish).pack(side="left", padx=3)
        ttk.Button(top, text="Undo (Ctrl+Z)", command=self.undo_edit).pack(side="left", padx=3)
        ttk.Button(top, text="Save (Ctrl+S)", command=self.save).pack(side="right", padx=5)

        editbar=ttk.Frame(self.root,padding=(7,0,7,5))
        editbar.pack(fill="x")
        ttk.Button(editbar, text="Retrace whole selected hold", command=self.retrace_selected).pack(side="left",padx=3)
        ttk.Button(editbar, text="Delete selected hold", command=self.delete_selected).pack(side="left",padx=3)
        ttk.Button(editbar, text="Undo draft point", command=self.undo_point).pack(side="left",padx=3)

        tools = ttk.Frame(self.root, padding=(7, 0, 7, 7))
        tools.pack(fill="x")
        ttk.Button(tools, text="◀ Previous", command=lambda: self.load(self.index-1)).pack(side="left", padx=3)
        ttk.Button(tools, text="Next ▶", command=lambda: self.load(self.index+1)).pack(side="left", padx=3)
        ttk.Button(tools, text="Next unlabelled", command=self.next_unlabelled).pack(side="left", padx=3)
        ttk.Button(tools, text="Zoom +", command=lambda: self.zoom(1.25)).pack(side="left", padx=(20,3))
        ttk.Button(tools, text="Zoom −", command=lambda: self.zoom(1/1.25)).pack(side="left", padx=3)
        ttk.Button(tools, text="Fit image (F)", command=self.fit).pack(side="left", padx=3)
        ttk.Button(tools, text="Focus hold (Z)", command=self.focus_selected).pack(side="left", padx=3)
        ttk.Button(tools, text="+ Point (A)", command=self.add_point).pack(side="left", padx=3)
        ttk.Button(tools, text="− Point (X)", command=self.remove_point).pack(side="left", padx=3)
        ttk.Button(tools, text="Smooth (S)", command=self.smooth_selected).pack(side="left", padx=3)
        ttk.Button(tools, text="Rotate left", command=lambda: self.rotate_photo(8)).pack(side="left", padx=(18, 3))
        ttk.Button(tools, text="Rotate right", command=lambda: self.rotate_photo(6)).pack(side="left", padx=3)
        ttk.Label(tools, textvariable=self.info).pack(side="right", padx=8)

        body = ttk.Panedwindow(self.root, orient="horizontal")
        body.pack(fill="both", expand=True)
        left = ttk.Frame(body, padding=(7, 2, 3, 4))
        body.add(left, weight=0)
        ttk.Label(left, text="Photographs (select to open)", font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(2, 7))
        list_frame = ttk.Frame(left)
        list_frame.pack(fill="both", expand=True)
        self.listbox = tk.Listbox(list_frame, width=33, exportselection=False,
                                  font=("Consolas", 10), activestyle="dotbox")
        scroll = ttk.Scrollbar(list_frame, orient="vertical", command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.listbox.pack(side="left", fill="both", expand=True)
        self.listbox.bind("<<ListboxSelect>>", self._on_list_select)
        ttk.Label(left, text="● = saved label    ○ = not saved", foreground="#555").pack(anchor="w", pady=(6, 2))
        ttk.Label(left, text="Only complete large red speed\nhandholds. No footholds.",
                  foreground="#555", justify="left").pack(anchor="w")

        right = ttk.Frame(body, padding=(3, 2, 7, 4))
        body.add(right, weight=1)
        self.canvas = tk.Canvas(right, bg="#202329", cursor="crosshair", highlightthickness=1,
                                highlightbackground="#777")
        self.canvas.pack(fill="both", expand=True)

        bottom = ttk.Frame(self.root, padding=7)
        bottom.pack(fill="x")
        ttk.Label(bottom, textvariable=self.status, wraplength=1300).pack(anchor="w")
        ttk.Label(bottom, text="E select a hold. R redraw any section: start and finish on its boundary; release to replace that piece immediately. Repeat as often as needed. B box-delete. Ctrl+Z undo.",
                  foreground="#555", wraplength=1350).pack(anchor="w", pady=(3, 0))
        self._refresh_list()
        self._update_assist_buttons()

    def _bind(self):
        c = self.canvas
        c.bind("<Configure>", self._resized)
        c.bind("<ButtonPress-1>", self._press)
        c.bind("<B1-Motion>", self._move)
        c.bind("<ButtonRelease-1>", self._release)
        c.bind("<ButtonPress-2>", self._middle_press)
        c.bind("<B2-Motion>", self._middle_move)
        c.bind("<ButtonRelease-2>", self._release)
        c.bind("<MouseWheel>", self._wheel)
        c.bind("<Button-4>", lambda e: self._wheel_units(e, 1))
        c.bind("<Button-5>", lambda e: self._wheel_units(e, -1))
        c.bind("<Button-3>", lambda e: self.finish() if self.mode == "polygon" else None)
        c.bind("<Double-Button-1>", self._double_click)
        self.root.bind_all("<KeyPress-space>", lambda e: self._space(True))
        self.root.bind_all("<KeyRelease-space>", lambda e: self._space(False))
        self.root.bind_all("<Control-s>", lambda e: self.save())
        self.root.bind_all("<Return>", lambda e: self.finish())
        self.root.bind_all("<BackSpace>", self._backspace)
        self.root.bind_all("<Delete>", lambda e: self.delete_selected())
        self.root.bind_all("<Control-z>", lambda e: self.undo_edit())
        self.root.bind_all("<Control-Z>", lambda e: self.undo_edit())
        self.root.bind_all("<Left>", lambda e: self.nudge_vertex(-1, 0, e))
        self.root.bind_all("<Right>", lambda e: self.nudge_vertex(1, 0, e))
        self.root.bind_all("<Up>", lambda e: self.nudge_vertex(0, -1, e))
        self.root.bind_all("<Down>", lambda e: self.nudge_vertex(0, 1, e))
        self.root.bind_all("<Escape>", lambda e: self.cancel_draft())
        for key, mode in (("d", "polygon"), ("t", "trace"), ("e", "edit"), ("r", "redraw"), ("b", "delete_box"), ("p", "pan")):
            self.root.bind_all(key, lambda e, m=mode: self.set_mode(m) if e.widget is not self.listbox else None)
        self.root.bind_all("f", lambda e: self.fit())
        self.root.bind_all("z", lambda e: self.focus_selected())
        self.root.bind_all("a", lambda e: self.add_point())
        self.root.bind_all("x", lambda e: self.remove_point())
        self.root.bind_all("s", lambda e: self.smooth_selected() if not (e.state & 4) else None)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _space(self, pressed):
        self.space_down = pressed
        if not self.drag:
            self.canvas.configure(cursor="fleur" if pressed else
                                  ("fleur" if self.mode == "pan" else "crosshair"))

    def _refresh_list(self):
        choice = self.index
        self.listbox.delete(0, "end")
        for p, key in zip(self.files, self.keys):
            saved = (LABELS / (Path(key).stem + ".txt")).exists()
            self.listbox.insert("end", ("● " if saved else "○ ") + p.name)
        if 0 <= choice < len(self.files):
            self.listbox.selection_set(choice)
            self.listbox.see(choice)

    def _on_list_select(self, event):
        sel = self.listbox.curselection()
        if sel and sel[0] != self.index:
            self.load(sel[0])

    def _resized(self, event):
        if self.image and not hasattr(self, "_fitted_once"):
            self.fit()
        elif self.image:
            self._render()

    def _dest_name(self, index=None):
        return self.keys[self.index if index is None else index]

    def _label_file(self, index=None):
        return LABELS / (Path(self._dest_name(index)).stem + ".txt")

    def load(self, index):
        if not (0 <= index < len(self.files)):
            return
        if self.draft:
            if not messagebox.askyesno("Unfinished hold", "Discard the unfinished polygon and change photograph?", parent=self.root):
                self._refresh_list()
                return
            self.draft = []
        if self.index >= 0 and self.dirty:
            self.save()
        self.suggestions = []
        self.pending_redraw = None
        self.box_rect = None
        self.model_job_counter += 1  # invalidate predictions for the previous image
        self.index = index
        self.image_path = self.files[index]
        try:
            self.image, was_rotated = load_upright_photo(self.image_path, self._label_file())
        except Exception as err:
            messagebox.showerror("Cannot open photograph", f"{self.image_path}\n{err}")
            return
        self.image_size = self.image.size
        self.polys = []
        self.selected = None
        self.selected_vertex = None
        self.selected_edge = None
        self.pending_redraw = None
        self.box_rect = None
        self.history = []
        self.dirty = False
        lab = self._label_file()
        rejected = 0
        if lab.is_file():
            for row in lab.read_text(encoding="utf-8-sig").splitlines():
                vals = row.split()
                if not vals: continue
                if vals[0] != "0" or len(vals) < 7 or len(vals) % 2 == 0:
                    rejected += 1
                    continue
                try:
                    coord = [float(a) for a in vals[1:]]
                    if not all(math.isfinite(v) and 0 <= v <= 1 for v in coord):
                        rejected += 1
                        continue
                    poly = [(coord[i] * self.image.width,coord[i+1] * self.image.height)
                            for i in range(0,len(coord),2)]
                    if (len(poly) < 3 or polygon_area(poly) < 25 or
                            has_self_intersection(poly) or conflicts_with_existing(poly,self.polys)):
                        rejected += 1
                        continue
                    self.polys.append(poly)
                except (ValueError, TypeError):
                    rejected += 1
        if rejected:
            # Make a one-time copy before deleting malformed/duplicate polygons.
            backup = lab.with_name(lab.stem + '.before_polygon_cleanup.bak')
            if not backup.exists():shutil.copy2(lab,backup)
            self.save()
        self._fitted_once = False
        self._refresh_list()
        self.root.title(f"Speed hold annotation  |  {self.image_path.name}")
        self.root.after(30, self.fit)
        if rejected:
            self._update_status(f"Automatically discarded {rejected} malformed/duplicate saved polygons. Original labels backed up beside the .txt file.")
        elif was_rotated:
            self._update_status("Photo rotated upright automatically. Original backed up in data/originals. Labels stay aligned.")
        else:
            self._update_status("Photo loaded. Draw one polygon per red speed handhold.")
        self._update_assist_buttons()
        if self.assisted and not self.model_busy:
            self.request_suggestions()

    def rotate_photo(self, orientation):
        """Manual rescue when a file has sideways pixels without EXIF metadata."""
        if self.image is None or self.index < 0:
            return
        if self.model_busy:
            self._update_status("Model is still running; rotate after it finishes.")
            return
        if self.draft:
            self._update_status("Finish or cancel the current polygon before rotating.")
            return
        original_w, original_h = self.image_size
        label = self._label_file()
        if self.dirty:
            self.save()
        try:
            backup_once(self.image_path, label)
            new_image = self.image.transpose(Image.Transpose.ROTATE_270 if orientation == 6
                                             else Image.Transpose.ROTATE_90)
            self.polys = [transform_points(poly, orientation, original_w, original_h)
                          for poly in self.polys]
            save_image_atomic(new_image, self.image_path)
            self.image = new_image
            self.image_size = self.image.size
            self.suggestions = []
            self.model_job_counter += 1
            self.selected = None
            self.dirty = True
            self.save()
            self.fit()
            self._update_status("Photo rotated and labels updated. Original is backed up in data/originals.")
            if self.assisted:
                self.request_suggestions()
        except Exception as exc:
            messagebox.showerror("Rotation failed", str(exc), parent=self.root)

    def fit(self):
        if not self.image: return
        cw, ch = max(200, self.canvas.winfo_width()), max(150, self.canvas.winfo_height())
        w, h = self.image_size
        self.scale = min((cw - 36) / w, (ch - 36) / h)
        self.origin = [(cw - w * self.scale) / 2, (ch - h * self.scale) / 2]
        self._fitted_once = True
        self._render()

    def focus_selected(self):
        """Enlarge the selected hold in the main editor; not a second tiny window."""
        if self.selected is None or self.selected >= len(self.polys):
            return
        bb = polygon_bbox(self.polys[self.selected])
        cx,cy=(bb[0]+bb[2])/2,(bb[1]+bb[3])/2
        cw,ch=max(200,self.canvas.winfo_width()),max(150,self.canvas.winfo_height())
        scale=min(14., .62*cw/max(25.,bb[2]-bb[0]), .72*ch/max(25.,bb[3]-bb[1]))
        self.scale=max(self.scale,scale)
        self.origin=[cw/2-cx*self.scale,ch/2-cy*self.scale]
        self._render()
        self._update_status("Editing enlarged hold. Press R to redraw any section; F returns to full photo.")

    def _remember(self):
        import copy
        self.history.append(copy.deepcopy(self.polys))
        self.history=self.history[-30:]

    def undo_edit(self):
        if self.draft:
            self.cancel_draft()
            return
        if self.history:
            self.polys=self.history.pop()
            self.selected=None
            self.selected_vertex=None
            self.selected_edge=None
            self.dirty=True
            self.save()
            self._draw_overlay()
            self._update_status("Undid previous mask edit (Ctrl+Z).")

    def _valid_shape(self, poly, skip=None):
        if poly is None or len(poly)<3 or polygon_area(poly)<25:
            return "Outline too small."
        if has_self_intersection(poly):
            return "That edge crosses itself; edit cancelled."
        if conflicts_with_existing(poly,self.polys,skip=skip):
            return "This overlaps another saved hold. Edit cancelled (Ctrl+Z is also available)."
        return None

    def add_point(self):
        if self.mode != "edit" or self.selected is None or self.selected_edge is None:
            self._update_status("Select a hold and click its boundary, then use A to add a point.")
            return
        k=self.selected
        poly=self.polys[k]
        i=self.selected_edge
        a,b=poly[i],poly[(i+1)%len(poly)]
        mid=self.selected_edge_position or ((a[0]+b[0])/2,(a[1]+b[1])/2)
        self._remember()
        poly.insert(i+1,mid)
        self.selected_vertex=i+1
        self.selected_edge=None
        self.selected_edge_position=None
        self.dirty=True
        self.save()
        self._draw_overlay()
        self._update_status("Point added; drag it or use arrow keys to move it.")

    def remove_point(self):
        if self.mode != "edit" or self.selected is None or self.selected_vertex is None:
            self._update_status("Click a yellow vertex, then press X or Backspace to remove it.")
            return
        k=self.selected
        if len(self.polys[k])<=3:
            self._update_status("A polygon must have at least three points.")
            return
        trial=self.polys[k][:]
        del trial[self.selected_vertex]
        bad=self._valid_shape(trial,skip=k)
        if bad:
            self._update_status(bad)
            return
        self._remember()
        self.polys[k]=trial
        self.selected_vertex=None
        self.dirty=True
        self.save()
        self._draw_overlay()
        self._update_status("Point removed.")

    def smooth_selected(self):
        """One gentle clean-up of an existing hand-drawn or model-predicted mask."""
        if self.selected is None or self.mode != "edit":
            self._update_status("Select a hold in Edit mode before smoothing.")
            return
        k=self.selected
        trial=polish_outline(self.polys[k],self.scale,closed=True,smooth=True)
        bad=self._valid_shape(trial,skip=k)
        if bad:
            self._update_status("Cannot smooth this outline: "+bad)
            return
        self._remember()
        self.polys[k]=trial
        self.selected_vertex=None
        self.selected_edge=None
        self.selected_edge_position=None
        self.dirty=True
        self.save()
        self._draw_overlay()
        self._update_status("Outline smoothed and control points evenly spaced. Ctrl+Z to undo.")

    def nudge_vertex(self,dx,dy,event=None):
        if self.mode != "edit" or self.selected is None or self.selected_vertex is None:
            return
        if event is not None and event.widget is self.listbox:
            return
        delta=(5 if event is not None and (event.state & 1) else 1)/max(.05,self.scale)
        k,v=self.selected,self.selected_vertex
        pt=self.polys[k][v]
        w,h=self.image_size
        trial=self.polys[k][:]
        trial[v]=(min(w-1,max(0,pt[0]+dx*delta)),min(h-1,max(0,pt[1]+dy*delta)))
        if self._valid_shape(trial,skip=k):
            return
        self._remember()
        self.polys[k]=trial
        self.dirty=True
        self._draw_overlay()
        # Write after each keyboard movement, but retain undo history.
        self.save()

    def zoom(self, factor, px=None, py=None):
        if not self.image: return
        if px is None: px = self.canvas.winfo_width() / 2
        if py is None: py = self.canvas.winfo_height() / 2
        old = self.scale
        new = min(10.0, max(0.025, old * factor))
        ix, iy = (px - self.origin[0]) / old, (py - self.origin[1]) / old
        self.scale = new
        self.origin = [px - ix * new, py - iy * new]
        self._render()

    def _wheel(self, event):
        if event.delta:
            self._wheel_units(event, 1 if event.delta > 0 else -1)

    def _wheel_units(self, event, sign):
        self.zoom(1.25 if sign > 0 else 1/1.25, event.x, event.y)

    def _image_coords(self, px, py, clamp=True):
        x = (px - self.origin[0]) / self.scale
        y = (py - self.origin[1]) / self.scale
        if clamp:
            w, h = self.image_size
            x = max(0., min(w - 1., x))
            y = max(0., min(h - 1., y))
        return x, y

    def _screen_coords(self, point):
        return point[0] * self.scale + self.origin[0], point[1] * self.scale + self.origin[1]

    def _render(self):
        if self.image is None: return
        self.canvas.delete("photo")
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        w, h = self.image_size
        x0 = max(0, math.floor(-self.origin[0] / self.scale))
        y0 = max(0, math.floor(-self.origin[1] / self.scale))
        x1 = min(w, math.ceil((cw - self.origin[0]) / self.scale))
        y1 = min(h, math.ceil((ch - self.origin[1]) / self.scale))
        if x1 > x0 and y1 > y0:
            cropped = self.image.crop((x0, y0, x1, y1))
            sw = max(1, round((x1 - x0) * self.scale))
            sh = max(1, round((y1 - y0) * self.scale))
            rendered = cropped.resize((sw, sh), Image.Resampling.BILINEAR)
            self.tk_image = ImageTk.PhotoImage(rendered)
            self.canvas.create_image(self.origin[0] + x0 * self.scale,
                                     self.origin[1] + y0 * self.scale,
                                     image=self.tk_image, anchor="nw", tags="photo")
            self.canvas.tag_lower("photo")
        self._draw_overlay()

    def _draw_overlay(self):
        c = self.canvas
        c.delete("shapes")
        for k, poly in enumerate(self.polys):
            xy = [z for point in poly for z in self._screen_coords(point)]
            selected = (self.selected == k)
            invalid = has_self_intersection(poly)
            c.create_polygon(*xy, fill="" if invalid else ("#00bb80" if not selected else "#ffe45c"),
                             stipple="gray25" if not invalid else "",
                             outline="#ff4040" if invalid else ("#12ffa5" if not selected else "#ffff00"),
                             width=3 if invalid or selected else 2, tags="shapes")
            centre = (sum(x for x, y in poly) / len(poly), sum(y for x, y in poly) / len(poly))
            sx, sy = self._screen_coords(centre)
            c.create_text(sx, sy, text=str(k+1), fill="white", font=("Segoe UI", 12, "bold"),
                          tags="shapes")
            if selected and self.mode == "edit":
                # Avoid an unreadable forest of handles on a dense model mask.
                jump = max(1, math.ceil(len(poly)/40))
                for v, point in enumerate(poly):
                    if v % jump and v != self.selected_vertex:
                        continue
                    x, y = self._screen_coords(point)
                    r=6 if v==self.selected_vertex else 4
                    c.create_oval(x-r,y-r,x+r,y+r,fill="#ff8a00" if v==self.selected_vertex else "yellow",
                                  outline="black",width=1,tags="shapes")
                if self.selected_edge is not None and self.selected_edge < len(poly):
                    a,b=poly[self.selected_edge],poly[(self.selected_edge+1)%len(poly)]
                    ax,ay=self._screen_coords(a);bx,by=self._screen_coords(b)
                    c.create_line(ax,ay,bx,by,fill="#ff6100",width=5,tags="shapes")
        for poly in self.suggestions:
            xy = [z for point in poly for z in self._screen_coords(point)]
            if len(xy) >= 6:
                c.create_polygon(*xy, fill="#ff9333", stipple="gray25",
                                 outline="#ff9333", width=2, dash=(6, 4), tags="shapes")
        if self.pending_redraw:
            trial=self._pending_candidate()
            if trial:
                xy=[z for p in trial for z in self._screen_coords(p)]
                c.create_polygon(*xy, fill="",outline="#00eaff",width=4,tags="shapes")
                a,b=self.pending_redraw['stroke'][0],self.pending_redraw['stroke'][-1]
                for pt in (a,b):
                    x,y=self._screen_coords(pt)
                    c.create_oval(x-6,y-6,x+6,y+6,fill="#00eaff",outline="black",tags="shapes")
        if self.box_rect:
            x0,y0,x1,y1=self.box_rect
            ax,ay=self._screen_coords((x0,y0));bx,by=self._screen_coords((x1,y1))
            c.create_rectangle(ax,ay,bx,by,outline="#f95555",width=3,dash=(6,4),tags="shapes")
        stroke = self.drag[4] if self.drag and self.drag[0]=="redraw" else self.draft
        if stroke:
            xy = [z for point in stroke for z in self._screen_coords(point)]
            if len(xy) >= 4:
                c.create_line(*xy, fill="#ffef57", width=3, tags="shapes")
            for pt in stroke[::max(1, len(stroke)//150)]:
                x, y = self._screen_coords(pt)
                c.create_oval(x-2, y-2, x+2, y+2, fill="#ffff55", outline="", tags="shapes")
        self.info.set(f"Photo {self.index+1}/{len(self.files)}   |   {len(self.polys)} saved masks"
                      + (f"   |   {len(self.suggestions)} proposed" if self.suggestions else "")
                      + (f"   |   {len(self.draft)} draft points" if self.draft else ""))

    def _pending_candidate(self):
        if not self.pending_redraw:return None
        pending=self.pending_redraw
        poly=pending['original']
        mode='shorter' if pending['side']==0 else 'longer'
        trial=splice_boundary(poly,pending['stroke'],replace=mode)
        return trial

    def _select_for_edit(self, point):
        candidates=[]
        for k,poly in enumerate(self.polys):
            d,q,edge,arc=nearest_edge(poly,point)
            if point_inside(*point,poly) or d<=18/max(self.scale,.1):
                candidates.append((0 if k==self.selected else 1,d,k,q,edge))
        if not candidates:return False
        _,dist,k,q,edge=min(candidates)
        changed=(k!=self.selected)
        self.selected=k;self.selected_vertex=None
        self.selected_edge=edge;self.selected_edge_position=q
        if changed:self.focus_selected()
        return True

    def toggle_redraw_side(self):
        if not self.pending_redraw:
            self._update_status('Redraw an edge first (R), then press V to swap which original arc is removed.')
            return
        self.pending_redraw['side']=1-self.pending_redraw['side']
        trial=self._pending_candidate()
        err=self._valid_shape(trial,skip=self.pending_redraw['hold']) if trial else 'Degenerate edge.'
        self._draw_overlay()
        self._update_status(('Swapped arc. ' + (err or 'Preview shown in cyan. Press Enter to apply.')))

    def apply_redraw(self):
        if not self.pending_redraw:return
        pending=self.pending_redraw
        trial=self._pending_candidate()
        err=self._valid_shape(trial,skip=pending['hold']) if trial else 'Invalid replacement path.'
        if err:
            self._update_status('Cannot apply redraw: '+err+' Press V to try the other arc or Escape to cancel.')
            return
        self._remember()
        self.polys[pending['hold']]=trial
        self.pending_redraw=None
        self.dirty=True
        self.save()
        self.set_mode('edit')
        self._update_status('Boundary replaced, rest of hold preserved. Ctrl+Z to undo.')

    def retrace_selected(self):
        if self.selected is None:
            self._update_status('Select a hold in Edit mode first, then click Retrace whole hold.')
            return
        self.pending_redraw=None
        self.set_mode('retrace')
        self._update_status('Retrace selected hold: drag all the way around its boundary. Release to replace; Ctrl+Z undoes.')

    def _press(self, event):
        if not self.image:return
        if self.space_down or self.mode=='pan':
            self.drag=('pan',event.x,event.y,self.origin[:]);return
        pt=self._image_coords(event.x,event.y)
        if self.mode=='delete_box':
            self.box_rect=(*pt,*pt)
            self.drag=('delete_box',pt)
            self._draw_overlay();return
        if self.assisted and not self.draft and self.mode not in ('redraw','retrace'):
            for i in reversed(range(len(self.suggestions))):
                if point_inside(*pt,self.suggestions[i]):
                    self.use_one_suggestion(i);return
        if self.mode=='polygon':
            if not self.draft or math.dist(self.draft[-1],pt)>.8:self.draft.append(pt)
            self._draw_overlay();return
        if self.mode in ('trace','retrace'):
            self.draft=[pt];self.drag=(self.mode,);self._draw_overlay();return
        if self.mode not in ('edit','redraw'):return
        redraw=self.mode=='redraw' or bool(event.state&1)
        if redraw:
            # If a different hold is selected, click it once to focus first.
            if self.selected is None or not (point_inside(*pt,self.polys[self.selected]) or
                    nearest_edge(self.polys[self.selected],pt)[0]<=20/max(self.scale,.1)):
                if self._select_for_edit(pt):self._update_status('Hold selected. Now start the redraw on its boundary and finish on its boundary.')
                else:self._update_status('Click a hold first, then start the redraw from its outline.')
                self._draw_overlay();return
            dist,q,edge,_=nearest_edge(self.polys[self.selected],pt)
            if dist>22/max(self.scale,.1):
                self._update_status('Start the redraw directly on the selected hold boundary.');return
            self.pending_redraw=None
            self.drag=('redraw',self.selected,pt,self.polys[self.selected][:],[q])
            self._draw_overlay();return
        # Standard editing: choose a hold, then move vertex, edge or whole shape.
        point=pt;radius=12/max(self.scale,.05);edge_radius=15/max(self.scale,.05)
        candidates=[]
        for k,poly in enumerate(self.polys):
            d,q,edge,_=nearest_edge(poly,point)
            if d<=edge_radius or point_inside(*point,poly):candidates.append((k,d,q,edge))
        if not candidates:
            self.selected=None;self.selected_vertex=None;self.selected_edge=None
            self._draw_overlay();return
        chosen=min(candidates,key=lambda item:(0 if item[0]==self.selected else 1,item[1]))
        k,dist,q,edge=chosen
        if self.selected!=k:
            self.selected=k;self.selected_vertex=None;self.selected_edge=edge
            self.selected_edge_position=q
            self.drag=('select',k,event.x,event.y)
            self._draw_overlay();return
        self.selected_edge=edge;self.selected_edge_position=q
        verts=self.polys[k]
        vi=min(range(len(verts)),key=lambda j:math.dist(verts[j],point))
        if math.dist(verts[vi],point)<=radius:
            self.selected_vertex=vi;self._remember()
            self.drag=('vertex',k,vi,self.polys[k][:])
        elif dist<=edge_radius:
            self.selected_vertex=None;self.drag=None
            self._update_status('Edge selected. Double-click to insert point, or R to redraw a longer section.')
        else:
            self.selected_vertex=None;self._remember()
            self.drag=('shape',k,point,self.polys[k][:])
        self._draw_overlay()


    def _middle_press(self, event):
        self.drag = ("pan", event.x, event.y, self.origin[:])

    def _middle_move(self, event):
        if self.drag and self.drag[0] == "pan": self._move(event)

    def _move(self, event):
        if not self.drag: return
        mode=self.drag[0]
        if mode=="pan":
            _,px,py,offset=self.drag
            self.origin=[offset[0]+event.x-px,offset[1]+event.y-py]
            self._render()
        elif mode=="delete_box":
            start=self.drag[1]
            pt=self._image_coords(event.x,event.y)
            self.box_rect=(*start,*pt)
            self._draw_overlay()
        elif mode in ("trace","retrace"):
            pt=self._image_coords(event.x,event.y)
            # Dense sampling; smooth and evenly space the outline on release.
            if math.dist(self.draft[-1],pt)>=max(.5,1.35/max(self.scale,.05)):
                self.draft.append(pt)
                self._draw_overlay()
        elif mode=="redraw":
            pt=self._image_coords(event.x,event.y)
            stroke=self.drag[4]
            if math.dist(stroke[-1],pt)>=max(.5,1.3/max(self.scale,.05)):
                stroke.append(pt)
                self._draw_overlay()
        elif mode=="vertex":
            _,k,v,_=self.drag
            trial=self.polys[k][:]
            trial[v]=self._image_coords(event.x,event.y)
            if not self._valid_shape(trial,skip=k):
                self.polys[k]=trial
                self.dirty=True
                self._draw_overlay()
        elif mode=="shape":
            _,k,start,initial=self.drag
            x,y=self._image_coords(event.x,event.y,clamp=False)
            dx,dy=x-start[0],y-start[1]
            w,h=self.image_size
            trial=[(min(w-1,max(0,a+dx)),min(h-1,max(0,b+dy))) for a,b in initial]
            if not self._valid_shape(trial,skip=k):
                self.polys[k]=trial
                self.dirty=True
                self._draw_overlay()

    def _release(self, event):
        if not self.drag:return
        mode=self.drag[0]
        if mode=='delete_box':
            if self.box_rect:
                x0,y0,x1,y1=self.box_rect
                if abs(x0-x1)*self.scale>8 and abs(y0-y1)*self.scale>8:
                    saved=masks_in_rectangle(self.polys,self.box_rect)
                    proposed=masks_in_rectangle(self.suggestions,self.box_rect)
                    if saved:self._remember()
                    for i in reversed(saved):del self.polys[i]
                    for i in reversed(proposed):del self.suggestions[i]
                    if saved:
                        self.selected=None;self.selected_vertex=None;self.selected_edge=None
                        self.dirty=True;self.save()
                    self._update_status(f'Box removed {len(saved)} saved holds and {len(proposed)} pending suggestions. Ctrl+Z restores saved masks.')
                else:self._update_status('Drag a box around the centres of the holds to delete.')
            self.box_rect=None
        elif mode in ('trace','retrace'):
            if len(self.draft)>=4:
                self.draft=polish_outline(self.draft,self.scale,closed=True)
                if mode=='retrace' and self.selected is not None:
                    k=self.selected
                    bad=self._valid_shape(self.draft,skip=k)
                    if bad:self._update_status('Cannot retrace hold: '+bad)
                    else:
                        self._remember()
                        self.polys[k]=self.draft[:]
                        self.dirty=True
                        self.save()
                        self._update_status('Replaced entire selected hold. Ctrl+Z undoes.')
                    self.draft=[]
                    self.set_mode('edit')
                else:self.finish()
            else:self.draft=[]
        elif mode=='select':
            _,k,px,py=self.drag
            if abs(event.x-px)<10 and abs(event.y-py)<10:self.focus_selected()
        elif mode=='redraw':
            _,k,_,original,stroke=self.drag
            point=self._image_coords(event.x,event.y)
            dist,endpoint,_,_=nearest_edge(original,point)
            if dist>24/max(self.scale,.05):
                self._update_status('End the redraw on this hold boundary; no change was made.')
            elif len(stroke)<3:
                self._update_status('Trace a longer section from edge to edge; no change was made.')
            else:
                stroke=stroke[:]+[point]
                trace=polish_outline(stroke,self.scale,closed=False)
                if len(trace)>=3:
                    trace[0]=nearest_edge(original,trace[0])[1]
                    trace[-1]=endpoint
                    trial,error=replace_local_boundary(original,trace,
                                       validator=lambda item:self._valid_shape(item,skip=k))
                    if trial is not None:
                        self._remember()
                        self.polys[k]=trial
                        self.pending_redraw=None
                        self.dirty=True
                        self.save()
                        self._update_status('Replaced one boundary section. Redraw another part of the SAME hold, or Ctrl+Z to undo.')
                    else:
                        self._update_status('Redraw rejected: '+str(error)+' Keep the start and end on the same outline.')
                else:
                    self._update_status('Trace a longer piece of the boundary. No change was made.')
        elif mode in ('vertex','shape'):
            _,k,_,original=self.drag
            if self.polys[k]!=original:
                bad=self._valid_shape(self.polys[k],skip=k)
                if bad:
                    self.polys[k]=original
                    if self.history:self.history.pop()
                    self._update_status(bad)
                else:self.save()
            elif self.history:self.history.pop()
        self.drag=None
        self._draw_overlay()


    def _double_click(self, event):
        if self.mode == "polygon":
            self.finish()
            return
        if self.mode!="edit" or self.selected is None:return
        pt=self._image_coords(event.x,event.y)
        k=self.selected
        dist,q,edge,_=nearest_edge(self.polys[k],pt)
        if dist>20/max(self.scale,.05):return
        if math.dist(q,self.polys[k][edge])<2/max(self.scale,.05):return
        self._remember()
        self.polys[k].insert(edge+1,q)
        self.selected_vertex=edge+1
        self.selected_edge=None
        self.dirty=True
        self.save()
        self._draw_overlay()
        self._update_status("Inserted vertex. Drag it, use arrow keys, or Backspace to remove it.")

    def set_mode(self, mode):
        if self.draft and mode not in ('polygon','trace','retrace'):
            if not messagebox.askyesno('Unfinished hold','Discard unfinished outline to change tools?',parent=self.root):
                self.mode_var.set(self.mode);return
            self.draft=[]
        self.mode=mode;self.mode_var.set(mode)
        self.canvas.configure(cursor='fleur' if mode=='pan' else 'crosshair')
        self._draw_overlay()
        self._update_status({
            'polygon':'Polygon: click around a new hold; Enter finishes.',
            'trace':'Freehand: drag around a NEW hold and release. Smooth outline auto-saved.',
            'retrace':'Retrace: drag all around the SELECTED hold to replace its complete outline.',
            'edit':'Edit: click hold to enlarge; move points, or press R to redraw an edge.',
            'redraw':'Redraw: drag from one point of the selected hold boundary to another; release to replace that section. Repeat as needed; Ctrl+Z undoes.',
            'delete_box':'Box delete: drag rectangle over hold centres to remove multiple holds; Ctrl+Z to undo.',
            'pan':'Pan: drag image. Mouse wheel zooms.'
        }[mode])


    def finish(self):
        if len(self.draft) < 3:
            if self.draft: self._update_status("Need at least three points to finish a hold.")
            return
        pts = self.draft[:]
        if len(pts) >= 4 and math.dist(pts[0], pts[-1]) < max(2., 4/self.scale):
            pts.pop()
        if len(pts) < 3 or polygon_area(pts) < 25:
            self._update_status("Hold outline is too small. Retry.")
            return
        # Keep new outlines distinct and non-self-crossing. Adjacent holds are fine.
        bad=self._valid_shape(pts)
        if bad:
            self._update_status(bad+" Use Edit to correct an existing mask instead of drawing on top.")
            self.draft=[]
            self._draw_overlay()
            return
        self._remember()
        self.polys.append(pts)
        self.selected = len(self.polys) - 1
        self.selected_vertex=None
        self.selected_edge=None
        self.draft = []
        self.dirty = True
        self._draw_overlay()
        self.save()
        self._update_status(f"Saved hold {len(self.polys)}. Continue tracing every target hold in this photo.")

    def undo_point(self):
        if self.draft:
            self.draft.pop()
            self._draw_overlay()
        else:
            self._update_status("No draft point to undo. Select a mask in Edit mode to delete it.")

    def _backspace(self, event):
        if event.widget is self.listbox: return
        if self.draft: self.undo_point()
        elif self.mode == "edit": self.remove_point()

    def cancel_draft(self):
        self.draft=[]
        self.pending_redraw=None
        self.box_rect=None
        self.drag=None
        self._draw_overlay()
        self._update_status('Cancelled current draft/preview. Existing labels unchanged.')

    def delete_selected(self):
        if self.selected is None or self.selected >= len(self.polys):
            self._update_status("Select a hold in Edit mode first.")
            return
        k = self.selected
        self._remember()
        del self.polys[k]
        self.selected = None
        self.selected_vertex=None
        self.selected_edge=None
        self.dirty = True
        self.save()
        self._draw_overlay()
        self._update_status(f"Deleted hold {k+1}.")

    def save(self):
        if self.index < 0: return
        w, h = self.image_size
        lines = []
        for poly in self.polys:
            if len(poly) < 3 or polygon_area(poly) < 25 or has_self_intersection(poly): continue
            line = ["0"]
            for x, y in poly:
                line.extend((f"{max(0.,min(1.,x/w)):.6f}", f"{max(0.,min(1.,y/h)):.6f}"))
            lines.append(" ".join(line))
        self._label_file().write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        self.dirty = False
        self._refresh_list()
        self._update_status(f"Saved {len(lines)} handholds to data/labels/{self._label_file().name}")

    def next_unlabelled(self):
        for i in list(range(self.index+1, len(self.files))) + list(range(0, self.index+1)):
            if not self._label_file(i).exists():
                self.load(i)
                return
        self._update_status("All photographs have saved labels. Use the list to review any image.")

    def _update_status(self, text):
        self.status.set(text)

    def _update_assist_buttons(self):
        ready = self.assisted and not self.model_busy and self.index >= 0
        self.accept_button.configure(state="normal" if self.assisted and bool(self.suggestions) else "disabled")

    def change_proposal_method(self):
        """Regenerate proposals when switching sources; never change saved polygons."""
        self.suggestions=[]
        self.model_job_counter+=1
        self._draw_overlay()
        if self.assisted:
            if self.model_busy:
                self._update_status('Model choice changed. New proposals will start after the current run.')
            else:
                self.request_suggestions()

    def set_assisted(self, enabled):
        if enabled and not WEIGHTS.is_file():
            messagebox.showwarning("Model not found", "Copy your existing YOLO weights to data/best.pt for rough boxes. SAM 2 supplies the mask boundaries.\nManual labelling works without weights.", parent=self.root)
            self.assist_var.set("manual")
            self.assisted = False
            self._update_assist_buttons()
            return
        self.assisted = enabled
        self.assist_var.set("assisted" if enabled else "manual")
        if not enabled:
            self.discard_suggestions()
            self._update_status("Manual mode. Draw polygons or freehand; use Edit to refine them.")
        else:
            self._update_status("Model-assisted: choose YOLO segmentation or YOLO boxes + SAM. Orange masks are suggestions only.")
            self.request_suggestions()
        self._update_assist_buttons()

    def request_suggestions(self):
        if not self.assisted or self.index < 0 or self.model_busy:
            return
        if not WEIGHTS.is_file():
            messagebox.showerror("Missing weights", f"Place your YOLO model at {WEIGHTS}", parent=self.root)
            return
        index=self.index
        job=self.model_job_counter
        method='yolo' if self.proposal_method_var.get()=='YOLO segmentation' else 'sam'
        rgb=self.image.copy()
        self.model_busy=True
        self.suggestions=[]
        self._update_assist_buttons()
        self._update_status(f"{self.proposal_method_var.get()} running on {self.files[index].name} ...")
        def task():
            try:
                from ultralytics import YOLO
                if self.model is None:
                    self.model=YOLO(str(WEIGHTS))
                if method=='sam' and self.sam is None:
                    from ultralytics import SAM
                    self.sam=SAM(str(SAM_WEIGHTS) if SAM_WEIGHTS.is_file() else 'sam2_t.pt')
                    downloaded=Path.cwd()/'sam2_t.pt'
                    if not SAM_WEIGHTS.is_file() and downloaded.is_file():
                        SAM_WEIGHTS.parent.mkdir(parents=True,exist_ok=True)
                        shutil.move(str(downloaded),str(SAM_WEIGHTS))
                def progress(done,total):
                    self.model_events.put(('progress',job,index,(done,total,method)))
                polygons=model_polygons(rgb,self.model,self.sam,progress,method=method)
                self.model_events.put(('ok',job,index,(polygons,method)))
            except Exception as exc:
                self.model_events.put(('error',job,index,str(exc)))
        threading.Thread(target=task,daemon=True).start()

    def _poll_model_events(self):
        try:
            while True:
                kind, job, index, payload = self.model_events.get_nowait()
                if kind=='progress':
                    if job==self.model_job_counter and index==self.index:
                        self._update_status(f'{"YOLO segmentation" if payload[2]=="yolo" else "YOLO boxes → SAM"}: {payload[0]}/{payload[1]} holds processed.')
                    continue
                self.model_busy = False
                if kind == "error":
                    if job == self.model_job_counter and index == self.index:
                        self._update_status(f"Model failed: {payload}. Manual annotation still works.")
                        messagebox.showerror("Model inference failed", str(payload), parent=self.root)
                elif job == self.model_job_counter and index == self.index and self.assisted:
                    # Keep only unlabelled proposals, not layered duplicates.
                    raw_proposals, completed_method=payload
                    proposals=[]
                    for poly in raw_proposals:
                        fixed=poly[:]
                        if len(fixed)<3 or polygon_area(fixed)<25 or has_self_intersection(fixed):
                            continue
                        if conflicts_with_existing(fixed,self.polys) or conflicts_with_existing(fixed,proposals):
                            continue
                        proposals.append(fixed)
                    self.suggestions=proposals
                    self._draw_overlay()
                    name='YOLO segmentation' if completed_method=='yolo' else 'YOLO boxes + SAM'
                    self._update_status(f"{len(proposals)} {name} outlines (orange). Click to accept/edit, or use all.")
                self._update_assist_buttons()
                if self.assisted and self.index >= 0 and job != self.model_job_counter:
                    self.request_suggestions()
        except queue.Empty:
            pass
        if self.root.winfo_exists():
            self.root.after(150, self._poll_model_events)

    def use_one_suggestion(self,index):
        if not (0<=index<len(self.suggestions)):return
        poly=self.suggestions.pop(index)
        if conflicts_with_existing(poly,self.polys):
            self._update_status("That suggestion already overlaps a saved mask; it was skipped.")
            self._draw_overlay()
            return
        self._remember()
        self.polys.append(poly)
        self.selected=len(self.polys)-1
        self.selected_vertex=None
        self.selected_edge=None
        self.selected_edge_position=None
        self.mode="edit"
        self.mode_var.set("edit")
        self.dirty=True
        self.save()
        self._update_assist_buttons()
        self.focus_selected()
        self._update_status("Suggestion accepted. You can now edit the enlarged hold immediately.")

    def accept_suggestions(self):
        if not self.suggestions:return
        accepted=0
        self._remember()
        for poly in self.suggestions:
            if conflicts_with_existing(poly,self.polys):
                continue
            self.polys.append(poly)
            accepted+=1
        self.suggestions=[]
        if accepted:
            self.dirty=True
            self.save()
        else:
            self.history.pop()
        self._draw_overlay()
        self._update_assist_buttons()
        self._update_status(f"Used {accepted} new masks; existing labels were left unchanged. Click any hold to zoom and refine it.")

    def discard_suggestions(self):
        self.suggestions=[]
        self._draw_overlay()
        self._update_assist_buttons()

    def close(self):
        if (self.draft or self.pending_redraw) and not messagebox.askyesno("Unfinished annotation", "An unfinished polygon/redraw preview is open. Discard it and close?", parent=self.root):
            return
        if self.dirty: self.save()
        self.root.destroy()


def main():
    IMAGES.mkdir(parents=True, exist_ok=True)
    LABELS.mkdir(parents=True, exist_ok=True)
    photos = source_files(IMAGES)
    stems = [p.stem.casefold() for p in photos]
    if len(set(stems)) != len(stems):
        raise SystemExit("Two images share the same filename stem (e.g. photo.jpg and photo.png). Rename one to avoid overwriting annotations.")
    if not photos:
        raise SystemExit(f"No images in {IMAGES}. Copy your JPG/PNG photographs there, then run python main.py.")
    root = tk.Tk()
    root.option_add("*Font", ("Segoe UI", 10))
    Annotator(root, photos)
    root.mainloop()


if __name__ == "__main__":
    main()
