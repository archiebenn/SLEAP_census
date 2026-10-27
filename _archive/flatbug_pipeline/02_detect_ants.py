"""Stage 02 - detect ants (and, with a trained model, brood) on every sampled frame.

Two backends, same output format:
  flatbug  zero-shot flat-bug, a public arthropod detector/segmenter (Svenning et al. 2025, MIT code) trained on
           >100k insects from 23 imaging setups. Good for workers on the floor; no classes, so stage 03 decides
           worker / queen / not-an-ant from size and darkness.
  yolo     the 4-class detector fine-tuned in stage 12 (worker, queen, larva, pupa), run on overlapping 640-px tiles at
           full resolution; each tile keeps only detections centred in its own core area, so nothing is counted twice.

Reads  <work_root>/<stem>/sample.json + frames/  (stage 01)
Writes <work_root>/<stem>/detections.json : per frame, one record per detection
       [x, y (centre), box x1 y1 x2 y2, area_px, length_px, width_px, score, darkness, core_px, outline, cls (yolo only)]
       darkness = 1 - (pixel / local background) at the object's darkest 5%: ants 0.75-0.9, brood/cocoons 0.45-0.67.
       core_px  = ant-body pixels (pixel/background < core_ratio) inside the outline.
       Per frame also residual_dark: [x, y, px] patches of ant-body pixels outside every outline, i.e. the cores of
       piles the detector could not split (stage 03 converts them to ants by area).

Run:  python 02_detect_ants.py                                              (zero-shot flat-bug)
      python 02_detect_ants.py --backend yolo --weights <models>/.../best.pt --overwrite
"""
import argparse, json, os
from pathlib import Path
import numpy as np, cv2, torch

ap = argparse.ArgumentParser()
ap.add_argument("--work_root", default="/media/tom/MSATA_500GB/ant_census_work/census")
ap.add_argument("--backend", default="flatbug", choices=["flatbug", "yolo"])
ap.add_argument("--weights", default="/media/tom/MSATA_500GB/ant_census_work/weights/flat_bug_M_v2.pt")
ap.add_argument("--tile", type=int, default=640, help="yolo backend: tile size (training tile size)")
ap.add_argument("--tile_margin", type=int, default=64, help="yolo backend: overlap kept out of each tile's core area")
ap.add_argument("--score_thr", type=float, default=0.2)
ap.add_argument("--min_size_px", type=int, default=12, help="flat-bug minimum object size (sqrt of box area)")
ap.add_argument("--single_scale", type=int, default=1, help="1 = native-resolution pass only (5x faster, same result here)")
ap.add_argument("--batch_size", type=int, default=4)
ap.add_argument("--core_ratio", type=float, default=0.35, help="pixel/background below this = ant body core")
ap.add_argument("--min_residual_px", type=int, default=60, help="smallest uncovered dark patch recorded (pile cores)")
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()

os.environ.setdefault("YOLO_AUTOINSTALL", "false")       # never let Ultralytics pip-install into the environment
if a.backend == "flatbug":
    from flat_bug.predictor import Predictor
    cfg = dict(SCORE_THRESHOLD=a.score_thr, MINIMUM_TILE_OVERLAP=384, EDGE_CASE_MARGIN=16, MIN_MAX_OBJ_SIZE=[a.min_size_px, 10 ** 8],
               MAX_MASK_SIZE=1024, PREFER_POLYGONS=True, EXPERIMENTAL_NMS_OPTIMIZATION=True, TIME=False, TILE_SIZE=1024,
               BATCH_SIZE=a.batch_size)
    pred = Predictor(a.weights, device="cuda:0" if torch.cuda.is_available() else "cpu", dtype=torch.float32, cfg=cfg)
else:
    from ultralytics import YOLO
    yolo = YOLO(a.weights)


def flatbug_objects(path):
    """(polygon, box, score, cls) per detection."""
    r = pred.pyramid_predictions(path, single_scale=bool(a.single_scale))
    return [(poly.cpu().numpy().astype(np.float32), b, float(s), None)
            for poly, b, s in zip(r.contours, r.boxes.cpu().numpy(), r.confs.cpu().numpy())]


def yolo_objects(path):
    """Tile at full resolution; keep a detection only if its centre lies in the tile's core (or at the image edge)."""
    img = cv2.imread(path); H, W = img.shape[:2]; T, m = a.tile, a.tile_margin; step = T - 2 * m
    origins = [(x, y) for y in range(-m, H - m, step) for x in range(-m, W - m, step)]
    out = []
    for k in range(0, len(origins), a.batch_size):
        batch = origins[k:k + a.batch_size]
        tiles = [cv2.copyMakeBorder(img[max(0, y):min(H, y + T), max(0, x):min(W, x + T)],
                                    max(0, -y), max(0, y + T - H), max(0, -x), max(0, x + T - W), cv2.BORDER_REPLICATE) for x, y in batch]
        for (x0, y0), r in zip(batch, yolo.predict(tiles, imgsz=T, conf=a.score_thr, max_det=1000, verbose=False)):
            for b, s, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)):
                cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
                if not (m <= cx < T - m and m <= cy < T - m):
                    continue
                b = b + [x0, y0, x0, y0]
                if not (0 <= (b[0] + b[2]) / 2 < W and 0 <= (b[1] + b[3]) / 2 < H):
                    continue
                poly = np.array([[b[0], b[1]], [b[2], b[1]], [b[2], b[3]], [b[0], b[3]]], np.float32)
                out.append((poly, b, float(s), yolo.names[c]))
    return out


def background_ratio(gray):
    """Pixel / local background (grey closing removes dark objects smaller than ~2 ant lengths)."""
    s = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA).astype(np.float32)
    bg = cv2.morphologyEx(s, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    bg = cv2.resize(cv2.GaussianBlur(bg, (0, 0), 6), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return gray.astype(np.float32) / np.maximum(bg, 1)


for sj in sorted(Path(a.work_root).glob("*/sample.json")):
    rec = json.load(open(sj)); d = sj.parent
    if (d / "detections.json").exists() and not a.overwrite:
        print(f"{rec['stem']}: detections exist, skipping (use --overwrite)"); continue
    frames_out = []
    for fr in rec["frames"]:
        path = str(d / "frames" / fr["file"])
        objs = flatbug_objects(path) if a.backend == "flatbug" else yolo_objects(path)
        ratio = background_ratio(cv2.imread(path, cv2.IMREAD_GRAYSCALE))
        core = ratio < a.core_ratio                             # ant-body pixels (legs and brood are paler)
        covered = np.zeros(ratio.shape, np.uint8)
        dets = []
        for poly, b, s, cls in objs:
            if len(poly) < 3:
                continue
            m = cv2.moments(poly); area = abs(m["m00"])
            if area < 1:
                continue
            cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            (_, _), (rw, rh), _ = cv2.minAreaRect(poly)
            x1, y1, x2, y2 = [int(v) for v in np.clip(b, 0, [ratio.shape[1] - 1, ratio.shape[0] - 1] * 2)]
            mask = np.zeros((y2 - y1 + 1, x2 - x1 + 1), np.uint8)
            cv2.fillPoly(mask, [np.round(poly - [x1, y1]).astype(np.int32)], 1)
            inside = ratio[y1:y2 + 1, x1:x2 + 1][mask[:ratio[y1:y2 + 1, x1:x2 + 1].shape[0], :ratio[y1:y2 + 1, x1:x2 + 1].shape[1]] > 0]
            dark = float(1 - np.percentile(inside, 5)) if inside.size else 0.0    # darkest 5% = body core, not legs
            core_px = int(core[y1:y2 + 1, x1:x2 + 1][mask[:core[y1:y2 + 1, x1:x2 + 1].shape[0], :core[y1:y2 + 1, x1:x2 + 1].shape[1]] > 0].sum())
            cv2.fillPoly(covered, [np.round(poly).astype(np.int32)], 1)
            simple = cv2.approxPolyDP(poly.reshape(-1, 1, 2), 1.5, True).reshape(-1, 2)
            dets.append(dict(x=round(cx, 1), y=round(cy, 1), box=[x1, y1, x2, y2], area=round(area), length=round(max(rw, rh)),
                             width=round(min(rw, rh)), score=round(float(s), 3), darkness=round(dark, 3), core_px=core_px,
                             outline=np.round(simple).astype(int).tolist(), **({"cls": cls} if cls else {})))
        # dark ant-body pixels not inside any outline: mostly the cores of piles the detector could not split
        covered = cv2.dilate(covered, np.ones((9, 9), np.uint8))
        left = cv2.morphologyEx((core & (covered == 0)).astype(np.uint8), cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        n, _, st, cen = cv2.connectedComponentsWithStats(left)
        residual = [[round(float(cen[i][0]), 1), round(float(cen[i][1]), 1), int(st[i, cv2.CC_STAT_AREA])]
                    for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= a.min_residual_px]
        frames_out.append(dict(file=fr["file"], t_s=fr["t_s"], detections=dets, residual_dark=residual))
        print(f"{rec['stem']} {fr['file']}: {len(dets)} detections", flush=True)
    json.dump(dict(stem=rec["stem"], model=a.weights, settings=vars(a), frames=frames_out), open(d / "detections.json", "w"))
