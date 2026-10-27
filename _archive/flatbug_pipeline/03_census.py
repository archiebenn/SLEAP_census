"""Stage 03 - turn per-frame detections into one census row per video, plus a QC overlay image.

Reads  <work_root>/<stem>/sample.json, detections.json (stage 02), static_max.png (stage 01)
Writes <out_dir>/census.csv           one row per video
       <out_dir>/qc/<stem>_qc.jpg     the frame nearest the median count, detections coloured by class
       <work_root>/<stem>/census_labels.json   the label given to every detection (for checking and scoring)

Each detection is judged relative to that video's typical worker (so camera distance can vary):
  outside arena    centre outside the lit floor + a wall band                      -> ignored
  pale object      darkness < ant_darkness (brood, cocoons, debris)                -> not an ant
  merged blob      outline holds >= 3 other detections' centres, or area > max_area_ratio x worker -> not counted
  fragment         area < min_area_ratio x worker                                  -> ignored
  queen candidate  area > queen_area_ratio x worker AND length > queen_length_ratio x worker
  worker           everything else
Ants in piles: stage 02 records ant-body pixels that no outline covers (pile cores). They are converted to ants as
  uncovered body pixels / body pixels of a typical isolated worker, reported separately (pile_ants) and added to the total.
With a trained 4-class model (stage 02 --backend yolo) the model's class labels replace the size rules, and larvae and
pupae are counted (brood barely moves, so the per-frame median is used).
Per-video worker estimate = the chosen summary (default 90th percentile) of the per-frame totals: ants hide under
each other, so single frames err low and an upper percentile is less biased than the median.
"""
import argparse, csv, json
from pathlib import Path
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--work_root", default="/media/tom/MSATA_500GB/ant_census_work/census")
ap.add_argument("--out_dir", default="/media/tom/MSATA_500GB/Dropbox/SLEAP-test/census_output")
ap.add_argument("--wall_band_px", type=int, default=160, help="band outside the lit floor still counted (box walls)")
ap.add_argument("--ant_darkness", type=float, default=0.70, help="ants 0.75-0.9, brood/cocoons 0.45-0.67 (checked on EG-07, HH-21)")
ap.add_argument("--min_area_ratio", type=float, default=0.3)
ap.add_argument("--max_area_ratio", type=float, default=15.0)
ap.add_argument("--queen_area_ratio", type=float, default=2.3)
ap.add_argument("--queen_length_ratio", type=float, default=1.5)
ap.add_argument("--pile_estimate", type=int, default=1, help="1 = add ants estimated from uncovered pile cores")
ap.add_argument("--worker_summary", default="p90", choices=["median", "p90", "max"])
a = ap.parse_args()
Path(a.out_dir, "qc").mkdir(parents=True, exist_ok=True)


def arena_mask(static_max):
    g = cv2.cvtColor(static_max, cv2.COLOR_BGR2GRAY)
    s = cv2.resize(g, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    _, b = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b)
    floor = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    hull = cv2.convexHull(cv2.findContours(floor, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0])
    floor = np.zeros_like(floor); cv2.drawContours(floor, [hull], -1, 1, -1)
    r = max(1, a.wall_band_px // 8)
    arena = cv2.dilate(floor, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    return cv2.resize(arena, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def classify(dets, inside, w_area, w_len):
    """Label each detection of one frame (see module docstring)."""
    cen = np.array([[q["x"], q["y"]] for q in dets]).reshape(-1, 2)
    labels = []
    for i, q in enumerate(dets):
        if not inside(q["x"], q["y"]):
            labels.append(None); continue
        if "cls" in q:
            labels.append(q["cls"]); continue
        x1, y1, x2, y2 = q["box"]
        near = np.where((cen[:, 0] >= x1) & (cen[:, 0] <= x2) & (cen[:, 1] >= y1) & (cen[:, 1] <= y2))[0]
        poly = np.array(q["outline"], np.float32).reshape(-1, 1, 2)
        held = sum(cv2.pointPolygonTest(poly, (float(cen[j, 0]), float(cen[j, 1])), False) >= 0 for j in near if j != i)
        if q["darkness"] < a.ant_darkness:
            labels.append("pale")
        elif held >= 3 or q["area"] > a.max_area_ratio * w_area:
            labels.append("merged")
        elif q["area"] < a.min_area_ratio * w_area:
            labels.append(None)
        elif q["area"] > a.queen_area_ratio * w_area and q["length"] > a.queen_length_ratio * w_len:
            labels.append("queen")
        else:
            labels.append("worker")
    return labels


COLORS = dict(worker=(0, 0, 255), queen=(255, 0, 255), larva=(0, 255, 0), pupa=(255, 255, 0), pale=(0, 200, 255), merged=(255, 128, 0))
rows = []
for sj in sorted(Path(a.work_root).glob("*/sample.json")):
    rec = json.load(open(sj)); d = sj.parent
    if not (d / "detections.json").exists():
        print(f"{rec['stem']}: no detections.json - run stage 02 first"); continue
    det = json.load(open(d / "detections.json"))
    arena = arena_mask(cv2.imread(str(d / "static_max.png")))
    H, W = arena.shape
    inside = lambda x, y: 0 <= int(y) < H and 0 <= int(x) < W and bool(arena[int(y), int(x)])
    trained = any("cls" in q for f in det["frames"] for q in f["detections"])
    # typical worker: dark, in-arena detections of middling size, pooled over all frames
    dark = [q for f in det["frames"] for q in f["detections"] if q["darkness"] >= a.ant_darkness and inside(q["x"], q["y"])]
    if len(dark) < 20 and not trained:
        raise SystemExit(f"{rec['stem']}: only {len(dark)} dark detections - check stage 02 output")
    areas = np.array([q["area"] for q in dark]); core = (areas > np.percentile(areas, 10)) & (areas < np.percentile(areas, 80))
    w_area = float(np.median(areas[core])); w_len = float(np.median(np.array([q["length"] for q in dark])[core]))
    w_core = float(np.median(np.array([q.get("core_px", 0) for q in dark])[core]))

    per_frame = []
    for f in det["frames"]:
        lab = classify(f["detections"], inside, w_area, w_len)
        pile_px = sum(px for x, y, px in f.get("residual_dark", []) if inside(x, y) and px >= 0.5 * w_core)
        pile = pile_px / w_core if (a.pile_estimate and w_core > 0) else 0.0
        n = {k: lab.count(k) for k in COLORS}
        per_frame.append(dict(file=f["file"], labels=lab, n=n, pile=pile, total=n["worker"] + pile))

    json.dump(dict(worker_area=w_area, worker_length=w_len, worker_core_px=w_core, frames=[
        dict(file=p["file"], labels=p["labels"], pile=p["pile"]) for p in per_frame]), open(d / "census_labels.json", "w"))
    tot = np.array([p["total"] for p in per_frame]); wk = np.array([p["n"]["worker"] for p in per_frame])
    qn = np.array([p["n"]["queen"] for p in per_frame]); pile = np.array([p["pile"] for p in per_frame])
    pick = dict(median=np.median, p90=lambda v: np.percentile(v, 90), max=np.max)[a.worker_summary]
    brood = {k: int(np.median([p["n"][k] for p in per_frame])) if trained else "" for k in ("larva", "pupa")}
    rows.append(dict(video=Path(rec["video"]).name, colony=rec["colony"], date=rec["date"], n_frames=len(tot),
                     workers_estimate=int(round(pick(tot))), workers_detected=int(round(pick(wk))), pile_ants=int(round(pick(pile))),
                     total_median=int(round(np.median(tot))), total_min=int(round(tot.min())), total_max=int(round(tot.max())),
                     total_cv_pct=round(100 * tot.std() / max(tot.mean(), 1), 1),
                     queen_candidate_frames=int((qn > 0).sum()), queen_candidates_max=int(qn.max()),
                     merged_blobs_median=int(np.median([p["n"]["merged"] for p in per_frame])),
                     pale_objects_median=int(np.median([p["n"]["pale"] for p in per_frame])),
                     worker_area_px=round(w_area), worker_length_px=round(w_len), worker_core_px=round(w_core),
                     brood_larvae=brood["larva"], brood_pupae=brood["pupa"], model=str(det["model"]), summary=a.worker_summary))

    # QC overlay on the frame whose total is closest to the median
    k = int(np.argmin(np.abs(tot - np.median(tot)))); f = det["frames"][k]
    img = cv2.imread(str(d / "frames" / f["file"]))
    img[~arena] = (img[~arena] * 0.4).astype(np.uint8)
    for q, c in zip(f["detections"], per_frame[k]["labels"]):
        if c:
            cv2.polylines(img, [np.array(q["outline"], np.int32)], True, COLORS[c], 3)
    for x, y, px in f.get("residual_dark", []):
        if inside(x, y) and px >= 0.5 * w_core:
            r = int(np.sqrt(px / np.pi)) + 6
            cv2.circle(img, (int(x), int(y)), r, (0, 255, 0), 3)
            cv2.putText(img, f"{px / w_core:.1f}", (int(x) + r, int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    txt = (f"{rec['stem']}  {f['file']}  detected {wk[k]} + piles {pile[k]:.0f} = {tot[k]:.0f}  "
           f"(estimate {rows[-1]['workers_estimate']}, {a.worker_summary})  queen cand. {qn[k]}")
    cv2.putText(img, txt, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (0, 0, 0), 8); cv2.putText(img, txt, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 3)
    cv2.imwrite(str(Path(a.out_dir, "qc", f"{rec['stem']}_qc.jpg")), cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA),
                [cv2.IMWRITE_JPEG_QUALITY, 88])
    r = rows[-1]
    print(f"{rec['stem']}: workers {r['workers_estimate']} ({a.worker_summary}: detected {r['workers_detected']} + piles {r['pile_ants']}; "
          f"median total {r['total_median']}, range {r['total_min']}-{r['total_max']}), queen candidate in {r['queen_candidate_frames']}/{len(tot)} frames")

with open(Path(a.out_dir, "census.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"wrote {Path(a.out_dir, 'census.csv')}")
