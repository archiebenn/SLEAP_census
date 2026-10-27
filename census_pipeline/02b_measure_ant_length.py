"""Stage 02b - measure how long a worker ant is, in pixels, in each setup's labelled frames.

Also decides each setup's resize factor, used by stages 03 and 05: if setups.tsv says 'auto', factor = standard
length (--standard_ant_px, 50 px = a box-video worker) / measured length; if it gives a number, that number is used
and the setup is not measured.
Method (same for every setup, so lengths are comparable):
  1. Darkness relative to the local background (frame / blurred background), so uneven lighting does not matter.
  2. Dark pixels = below an automatic (Otsu) threshold; each labelled point picks the dark blob it sits on.
  3. Only blobs holding exactly one labelled point are measured (isolated ants, not piles).
  4. Thin parts (legs, antennae) are removed by a small opening scaled to a first rough length; the ant's length is
     the long side of the smallest rectangle around the remaining blob. The median over all isolated ants is reported.
A check image shows a sample of measured ants with their measured length drawn, to confirm by eye.

Reads   setups.tsv (setup_id, points_file), the points files it names in outputs/
Writes  outputs/02b_ant_length.tsv        one row per setup: ants measured, median length, IQR, resize factor + source
        outputs/02b_ant_length_<setup>.jpg  sample of measured ants

Run:  /home/tom/.local/share/uv/tools/sleap/bin/python3 02b_measure_ant_length.py
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2
import sleap_io as sio

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--setups", nargs="*", default=None, help="setup ids to measure (default: all rows of setups.tsv)")
ap.add_argument("--max_frames", type=int, default=None, help="labelled frames per setup to measure (default: all; crowded frames have few isolated ants)")
ap.add_argument("--standard_ant_px", type=float, default=50, help="worker length all setups are resized to (box videos: 50)")
a = ap.parse_args()
out = HERE / "outputs"
rows = [r for r in csv.DictReader(open(HERE / "setups.tsv"), delimiter="\t") if a.setups is None or r["setup_id"] in a.setups]


def grey(img):
    img = np.asarray(img)
    g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 and img.shape[2] == 3 else img.reshape(img.shape[:2])
    return np.ascontiguousarray(g)


def darkness(g):
    """1 - pixel / local background (background = brightest surroundings, estimated at low resolution)."""
    f = 512 / max(g.shape)
    s = cv2.resize(g, None, fx=f, fy=f, interpolation=cv2.INTER_AREA).astype(np.float32)
    bg = cv2.GaussianBlur(cv2.dilate(s, np.ones((15, 15), np.uint8)), (0, 0), 6)
    bg = cv2.resize(bg, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_LINEAR)
    return np.clip(1 - g.astype(np.float32) / np.maximum(bg, 1), 0, 1)


def blob_lengths(dark, xy, open_px):
    t, _ = cv2.threshold((dark * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = (dark * 255 > t).astype(np.uint8)
    if open_px > 1:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_px, open_px)))
    n, lab = cv2.connectedComponents(m)
    ids = np.array([lab[int(y), int(x)] if 0 <= int(y) < lab.shape[0] and 0 <= int(x) < lab.shape[1] else 0 for x, y in xy])
    out = []
    for i, c in enumerate(ids):
        if c == 0 or (ids == c).sum() != 1:
            continue
        ys, xs = np.where(lab == c)
        if len(xs) < 20:
            continue
        rect = cv2.minAreaRect(np.stack([xs, ys], 1).astype(np.float32))
        out.append((max(rect[1]), rect, xy[i]))
    return out


results = []
for r in rows:
    if r["resize_to_standard"].strip().lower() != "auto":
        results.append(dict(setup_id=r["setup_id"], ants_measured="", median_length_px="", q25="", q75="",
                            resize_to_standard=float(r["resize_to_standard"]), source="setups.tsv"))
        print(f"{r['setup_id']:16s} resize factor {float(r['resize_to_standard'])} taken from setups.tsv (not measured)")
        continue
    labels = sio.load_slp(str(out / r["points_file"]))
    frames = [lf for lf in labels.labeled_frames if lf.instances][:a.max_frames]
    meas, samples = [], []
    for lf in frames:
        xy = np.array([i.numpy()[0] for i in lf.instances], float).reshape(-1, 2)
        xy = xy[np.isfinite(xy).all(1)]
        g = grey(lf.image); dark = darkness(g)
        rough = np.median([m[0] for m in blob_lengths(dark, xy, 1)] or [50])          # pass 1: legs included
        k = max(3, int(0.08 * rough)) | 1                                                # legs ~ under 8% of length
        found = blob_lengths(dark, xy, k)                                                # pass 2: legs removed
        meas += [m[0] for m in found]
        if len(samples) < 24:
            samples += [(g, m) for m in found[:4]]
    L = np.array(meas)
    if len(L) < 20:
        raise SystemExit(f"{r['setup_id']}: only {len(L)} isolated ants measured - this method needs dark ants on a plain pale "
                         f"floor. Measure a few ants by hand and put {a.standard_ant_px:.0f} / length in setups.tsv instead of 'auto'.")
    med = float(np.median(L))
    results.append(dict(setup_id=r["setup_id"], ants_measured=len(L), median_length_px=round(med, 1),
                        q25=round(float(np.percentile(L, 25)), 1), q75=round(float(np.percentile(L, 75)), 1),
                        resize_to_standard=round(a.standard_ant_px / med, 4), source="measured"))
    tiles = []
    for g, (length, rect, (x, y)) in samples[:24]:
        h = int(1.5 * med)
        c = cv2.copyMakeBorder(g, h, h, h, h, cv2.BORDER_REPLICATE)[int(y):int(y) + 2 * h, int(x):int(x) + 2 * h]
        c = cv2.cvtColor(cv2.resize(c, (180, 180)), cv2.COLOR_GRAY2BGR)
        box = cv2.boxPoints(((rect[0][0] - x + h, rect[0][1] - y + h), rect[1], rect[2])) * (180 / (2 * h))
        cv2.drawContours(c, [box.astype(np.int32)], -1, (0, 0, 255), 1)
        cv2.putText(c, f"{length:.0f}px", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        tiles.append(c)
    while len(tiles) % 8:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(str(out / f"02b_ant_length_{r['setup_id']}.jpg"), np.vstack([np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]))
    print(f"{r['setup_id']:16s} {len(L):5d} isolated ants: median length {med:.0f} px (middle half {results[-1]['q25']:.0f}-"
          f"{results[-1]['q75']:.0f}) -> resize factor {results[-1]['resize_to_standard']}")

with open(out / "02b_ant_length.tsv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(results[0]), delimiter="\t"); w.writeheader(); w.writerows(results)
