"""Stage 10 (annotation helper) - write pre-labelled frames for correction in tools/annotate_points.html.

For one frame per video (default: the first sampled frame) this copies the frame and writes <frame>.points.json with
  worker / queen points  from stage 02-03 (flat-bug outlines classified by size and darkness)
  larva points           only with --brood_candidates 1: blobs from a simple finder. Off by default: on the
                         current ribbed floors most of them sit on ribs and moulded text (1136 candidates on
                         EG-07, most wrong), so clicking brood from scratch is faster than deleting.
All pre-labels carry auto=true so corrected and untouched points can be told apart later.

Reads  <work_root>/<stem>/sample.json, detections.json, static_max.png
Writes <out_dir>/<stem>_<frame>.png  and  <stem>_<frame>.points.json

Run:  python 10_make_prelabels.py
"""
import argparse, json, shutil
from pathlib import Path
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--work_root", default="/media/tom/MSATA_500GB/ant_census_work/census")
ap.add_argument("--out_dir", default="/media/tom/MSATA_500GB/ant_census_work/annotation_round1")
ap.add_argument("--frame_index", type=int, default=0)
ap.add_argument("--ant_darkness", type=float, default=0.55)
ap.add_argument("--queen_area_ratio", type=float, default=2.3)
ap.add_argument("--brood_candidates", type=int, default=0)
ap.add_argument("--brood_min_contrast", type=float, default=0.12, help="blob darker than background by at least this fraction")
ap.add_argument("--brood_max_contrast", type=float, default=0.75, help="darker than this = ant body, not brood")
a = ap.parse_args()
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)


def ratio_image(gray):
    s = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).astype(np.float32)
    bg = cv2.morphologyEx(s, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    bg = cv2.resize(cv2.GaussianBlur(bg, (0, 0), 6), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return gray.astype(np.float32) / np.maximum(bg, 1)


def brood_candidates(gray, ant_mask, floor):
    """Pale-object blobs 8-50 px across: difference-of-Gaussians on darkness, outside ant bodies, on the lit floor."""
    dark = np.clip(1 - ratio_image(gray), 0, 1)
    dark[ant_mask] = 0
    dog = cv2.GaussianBlur(dark, (0, 0), 4) - cv2.GaussianBlur(dark, (0, 0), 14)
    peak = (dog == cv2.dilate(dog, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))) & (dog > 0.05)
    sm = cv2.GaussianBlur(dark, (0, 0), 3)
    ys, xs = np.where(peak & floor & (sm > a.brood_min_contrast) & (sm < a.brood_max_contrast))
    return list(zip(xs.tolist(), ys.tolist()))


for sj in sorted(Path(a.work_root).glob("*/sample.json")):
    rec = json.load(open(sj)); d = sj.parent
    if not (d / "detections.json").exists():
        print(f"{rec['stem']}: no detections.json - run stage 02 first"); continue
    det = json.load(open(d / "detections.json"))["frames"][a.frame_index]
    gray = cv2.imread(str(d / "frames" / det["file"]), cv2.IMREAD_GRAYSCALE)
    g = cv2.cvtColor(cv2.imread(str(d / "static_max.png")), cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    _, b = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    floor = cv2.resize(cv2.erode(b, np.ones((3, 3), np.uint8)), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    ants = [q for q in det["detections"] if q["darkness"] >= a.ant_darkness]
    areas = np.array([q["area"] for q in ants]); w_area = float(np.median(areas)) if len(areas) else 1.0
    ant_mask = np.zeros(gray.shape, np.uint8)
    pts = []
    for q in ants:
        if q["area"] > 6 * w_area:
            continue                                       # merged blob outline: let the annotator click its ants
        cv2.fillPoly(ant_mask, [np.array(q["outline"], np.int32)], 1)
        pts.append(dict(x=q["x"], y=q["y"], cls="queen" if q["area"] > a.queen_area_ratio * w_area else "worker", auto=True))
    ant_mask = cv2.dilate(ant_mask, np.ones((7, 7), np.uint8)) > 0
    ant_mask |= ratio_image(gray) < 1 - a.brood_max_contrast
    brood = brood_candidates(gray, ant_mask, floor) if a.brood_candidates else []
    pts += [dict(x=float(x), y=float(y), cls="larva", auto=True) for x, y in brood]
    name = f"{rec['stem']}_{Path(det['file']).stem}"
    shutil.copy(d / "frames" / det["file"], out / f"{name}.png")
    json.dump(dict(image=f"{name}.png", width=gray.shape[1], height=gray.shape[0], source_frame=str(d / "frames" / det["file"]),
                   points=pts, done=[]), open(out / f"{name}.points.json", "w"))
    print(f"{name}: {sum(p['cls'] == 'worker' for p in pts)} workers, {sum(p['cls'] == 'queen' for p in pts)} queen cand., {len(brood)} brood candidates")
