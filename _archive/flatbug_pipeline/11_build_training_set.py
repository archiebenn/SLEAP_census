"""Stage 11 (training) - turn corrected point annotations into a tiled YOLO detection dataset.

Input: an annotation folder holding <name>.png and the corrected <name>.points.json saved from tools/annotate_points.html.
Each point becomes a box: workers take the box of the flat-bug outline under the point when there is one (stage 02),
otherwise a box of the video's typical worker size; queens, larvae and pupae get fixed class sizes (counting only needs
the centre, so box size precision does not matter much). Only tiles the annotator marked done (key T) are used, so
unchecked areas never become false "background"; if no tile was marked, the whole frame counts as checked.

Writes <dataset_dir>/images/{train,val}/*.png, labels/{train,val}/*.txt, data.yaml
Validation = all tiles from the --val_frames (default: none -> a seeded 15% of tiles).

Run:  python 11_build_training_set.py --ann_dir /media/tom/MSATA_500GB/ant_census_work/annotation_round1
"""
import argparse, json, random
from pathlib import Path
import numpy as np, cv2

CLASSES = ["worker", "queen", "larva", "pupa"]
ap = argparse.ArgumentParser()
ap.add_argument("--ann_dir", default="/media/tom/MSATA_500GB/ant_census_work/annotation_round1")
ap.add_argument("--dataset_dir", default="/media/tom/MSATA_500GB/ant_census_work/datasets/round1")
ap.add_argument("--work_root", default="/media/tom/MSATA_500GB/ant_census_work/census", help="stage 02 outputs, for worker boxes")
ap.add_argument("--tile", type=int, default=640, help="must match the annotation tool's checking grid")
ap.add_argument("--queen_box_ratio", type=float, default=2.2, help="queen box side / worker length")
ap.add_argument("--larva_box", type=int, default=30)
ap.add_argument("--pupa_box", type=int, default=42)
ap.add_argument("--val_frames", nargs="*", default=[], help="annotated image names (without .png) used only for validation")
ap.add_argument("--val_fraction", type=float, default=0.15)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--only_corrected", type=int, default=0, help="1 = drop pre-label points the annotator never touched")
a = ap.parse_args()
random.seed(a.seed)
root = Path(a.dataset_dir)
for s in ("train", "val"):
    (root / "images" / s).mkdir(parents=True, exist_ok=True); (root / "labels" / s).mkdir(parents=True, exist_ok=True)

n_tiles = {"train": 0, "val": 0}; n_obj = {c: 0 for c in CLASSES}
for pj in sorted(Path(a.ann_dir).glob("*.points.json")):
    ann = json.load(open(pj)); name = pj.name[:-len(".points.json")]
    img = cv2.imread(str(pj.parent / f"{name}.png"))
    if img is None:
        raise SystemExit(f"missing image for {pj.name}")
    H, W = img.shape[:2]
    pts = [p for p in ann["points"] if p["cls"] in CLASSES and not (a.only_corrected and p.get("auto"))]
    # worker outlines from stage 02 for this exact frame (if available) give real box sizes
    stem, frame = name.rsplit("_", 1)
    det_path = Path(a.work_root) / stem / "detections.json"
    outlines, w_len = [], 50.0
    if det_path.exists():
        fr = [f for f in json.load(open(det_path))["frames"] if Path(f["file"]).stem == frame]
        if fr:
            dark = [q for q in fr[0]["detections"] if q["darkness"] >= 0.55]
            outlines = [q for q in dark if q["area"] < 6 * np.median([d["area"] for d in dark])]
            w_len = float(np.median([q["length"] for q in outlines])) if outlines else w_len
    boxes = []
    for p in pts:
        c = CLASSES.index(p["cls"]); x, y = p["x"], p["y"]
        if p["cls"] == "worker":
            near = [q for q in outlines if (q["x"] - x) ** 2 + (q["y"] - y) ** 2 < (0.3 * w_len) ** 2]
            if near:
                x1, y1, x2, y2 = min(near, key=lambda q: (q["x"] - x) ** 2 + (q["y"] - y) ** 2)["box"]
                boxes.append((c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1)); n_obj["worker"] += 1; continue
            s = 0.8 * w_len
        else:
            s = {"queen": a.queen_box_ratio * w_len, "larva": a.larva_box, "pupa": a.pupa_box}[p["cls"]]
        boxes.append((c, x, y, s, s)); n_obj[p["cls"]] += 1
    boxes = np.array(boxes, float).reshape(-1, 5)
    done = set(ann.get("done") or [])
    for j in range(0, H, a.tile):
        for i in range(0, W, a.tile):
            if done and f"{i // a.tile},{j // a.tile}" not in done:
                continue
            x0, y0 = i, j; x1, y1 = min(W, i + a.tile), min(H, j + a.tile)
            if x1 - x0 < a.tile // 2 or y1 - y0 < a.tile // 2:
                continue
            split = "val" if (name in a.val_frames or (not a.val_frames and random.random() < a.val_fraction)) else "train"
            tile = img[y0:y1, x0:x1]; th, tw = tile.shape[:2]
            sel = boxes[(boxes[:, 1] >= x0) & (boxes[:, 1] < x1) & (boxes[:, 2] >= y0) & (boxes[:, 2] < y1)] if len(boxes) else boxes
            lines = []
            for c, cx, cy, bw, bh in sel:
                bx1, by1 = max(cx - bw / 2, x0), max(cy - bh / 2, y0); bx2, by2 = min(cx + bw / 2, x1), min(cy + bh / 2, y1)
                lines.append(f"{int(c)} {((bx1 + bx2) / 2 - x0) / tw:.6f} {((by1 + by2) / 2 - y0) / th:.6f} {(bx2 - bx1) / tw:.6f} {(by2 - by1) / th:.6f}")
            tid = f"{name}_x{x0}_y{y0}"
            cv2.imwrite(str(root / "images" / split / f"{tid}.png"), tile)
            (root / "labels" / split / f"{tid}.txt").write_text("\n".join(lines))
            n_tiles[split] += 1
if n_tiles["val"] == 0 and n_tiles["train"] > 1:          # Ultralytics needs a validation set: hold out the last tile
    last = sorted((root / "images" / "train").glob("*.png"))[-1]
    last.rename(root / "images" / "val" / last.name)
    (root / "labels" / "train" / f"{last.stem}.txt").rename(root / "labels" / "val" / f"{last.stem}.txt")
    n_tiles["train"] -= 1; n_tiles["val"] += 1
(root / "data.yaml").write_text(f"path: {root}\ntrain: images/train\nval: images/val\nnames:\n" + "".join(f"  {k}: {c}\n" for k, c in enumerate(CLASSES)))
print(f"tiles: {n_tiles}, objects: {n_obj}, dataset: {root}")
