"""Stage 03 - put every setup's labelled frames at the standard ant size and cut them into training tiles.

1. Each labelled frame is resized by its setup's factor (stage 02b), so ants are the same size in pixels in every
   setup (the standard: a worker ~50 px long, as in the box videos), and converted to greyscale.
2. Frames are cut into overlapping 1136 x 1136 tiles (overlap >= 96 px, more than an ant's length, so every ant is
   whole in at least one tile). Whole 4K frames do not fit the 6 GB GPU for training; tiles do.
   Each tile keeps the points whose centre lies inside it. Tiles with no ants are left out.
3. All tiles are saved, images embedded, in one SLEAP package for stage 04.

Only training frames are used: stage 02's test frames are never tiled.

Reads   setups.tsv (which points file belongs to each setup), outputs/02b_ant_length.tsv (resize factor per setup),
        the points files (plus their _frames.tsv, if any, to keep test frames out)
Writes  outputs/03_tiles_<name>.pkg.slp     training tiles (embedded images), skeleton "ant"
        outputs/03_tiles_<name>_qc.jpg      one tile per setup side by side, at the standard size, with the points

Run:  /home/tom/.local/share/uv/tools/sleap/bin/python3 03_build_training_tiles.py --name box_petri
      /home/tom/.local/share/uv/tools/sleap/bin/python3 03_build_training_tiles.py --name box_only --setups box_4k_2026
"""
import argparse, csv, math, shutil, tempfile
from pathlib import Path
import numpy as np, cv2
import sleap_io as sio

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="name of this training set, used in output file names")
ap.add_argument("--setups", nargs="*", required=True, help="setup ids from setups.tsv to include")
ap.add_argument("--tile", type=int, default=1136, help="tile size (multiple of 16); 1136 holds a whole petri frame at x0.2 (1100 x 1130 px) so petri ants are not copied into overlapping tiles")
ap.add_argument("--min_overlap", type=int, default=96)
ap.add_argument("--out_dir", default=str(HERE / "outputs"))
a = ap.parse_args()
out = Path(a.out_dir)
f02b = out / "02b_ant_length.tsv"
if not f02b.exists():
    raise SystemExit("outputs/02b_ant_length.tsv not found - run stage 02b first")
factor = {r["setup_id"]: float(r["resize_to_standard"]) for r in csv.DictReader(open(f02b), delimiter="\t")}
SOURCES = {}
for r in csv.DictReader(open(HERE / "setups.tsv"), delimiter="\t"):
    pts = out / r["points_file"]; ftsv = pts.with_name(pts.stem + "_frames.tsv")
    SOURCES[r["setup_id"]] = (pts, ftsv if ftsv.exists() else None)       # no frames table = every frame trains
missing = [s for s in a.setups if s not in SOURCES or s not in factor]
if missing:
    raise SystemExit(f"setups {missing}: need a row in setups.tsv and a factor from stage 02b")


def starts(size):
    """Evenly spaced tile origins covering [0, size) with at least min_overlap between neighbours."""
    if size <= a.tile:
        return [0]
    n = math.ceil((size - a.tile) / (a.tile - a.min_overlap)) + 1
    return [round(i * (size - a.tile) / (n - 1)) for i in range(n)]


def grey(img):
    img = np.asarray(img)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    return img.reshape(img.shape[0], img.shape[1])


tmp = Path(tempfile.mkdtemp(prefix="census_tiles_"))
paths, points, origin, qc = [], [], [], {}
try:
    for setup in a.setups:
        slp, frames_tsv = SOURCES[setup]
        f = factor[setup]
        train = None
        if frames_tsv:
            train = {(r["video"], int(r["frame_idx"])) for r in csv.DictReader(open(frames_tsv), delimiter="\t") if r["split"] == "train"}
        labels = sio.load_slp(str(slp))
        n_frames = n_tiles = n_ants = 0
        for lf in labels.labeled_frames:
            if train is not None and (Path(lf.video.filename).name, lf.frame_idx) not in train:
                continue
            xy = np.array([i.numpy()[0] for i in lf.instances], float).reshape(-1, 2)
            xy = xy[np.isfinite(xy).all(1)] * f
            img = grey(lf.image)
            img = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_LINEAR)
            H, W = img.shape
            if H < a.tile or W < a.tile:
                img = cv2.copyMakeBorder(img, 0, max(0, a.tile - H), 0, max(0, a.tile - W), cv2.BORDER_REFLECT)
                H, W = img.shape
            n_frames += 1
            for y0 in starts(H):
                for x0 in starts(W):
                    inside = (xy[:, 0] >= x0) & (xy[:, 0] < x0 + a.tile) & (xy[:, 1] >= y0) & (xy[:, 1] < y0 + a.tile)
                    if not inside.any():
                        continue
                    p = tmp / f"{len(paths):06d}.png"
                    cv2.imwrite(str(p), img[y0:y0 + a.tile, x0:x0 + a.tile])
                    paths.append(str(p)); points.append(xy[inside] - [x0, y0])
                    origin.append((setup, Path(lf.video.filename).name, lf.frame_idx, x0, y0))
                    n_tiles += 1; n_ants += int(inside.sum())
                    if setup not in qc or inside.sum() > len(qc[setup][1]):
                        qc[setup] = (str(p), xy[inside] - [x0, y0])
        print(f"{setup:12s} x{f:<5} {n_frames:3d} frames -> {n_tiles:4d} tiles, {n_ants} ant points (tile copies included)")

    skel = sio.Skeleton(nodes=[sio.Node("ant")], name="ant_point")
    video = sio.Video.from_filename(paths)
    tiles = sio.Labels(videos=[video], skeletons=[skel])
    for k, xy in enumerate(points):
        tiles.append(sio.LabeledFrame(video=video, frame_idx=k,
                                      instances=[sio.Instance.from_numpy(c[None], skeleton=skel) for c in xy]))
    tiles.provenance["tile_origins"] = [list(o) for o in origin]
    tiles.provenance["setups_resize"] = {s: factor[s] for s in a.setups}
    dst = out / f"03_tiles_{a.name}.pkg.slp"
    sio.save_slp(tiles, str(dst), embed=True, verbose=False)

    panels = []
    for setup, (p, xy) in qc.items():
        im = cv2.cvtColor(cv2.imread(p, cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
        for x, y in xy:
            cv2.circle(im, (int(x), int(y)), 10, (0, 0, 255), 2)
        cv2.putText(im, f"{setup} x{factor[setup]}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        panels.append(im)
    cv2.imwrite(str(out / f"03_tiles_{a.name}_qc.jpg"), np.hstack(panels), [cv2.IMWRITE_JPEG_QUALITY, 85])
finally:
    shutil.rmtree(tmp, ignore_errors=True)

check = sio.load_slp(str(dst))
assert len(check.labeled_frames) == len(points) and sum(len(lf.instances) for lf in check.labeled_frames) == sum(len(p) for p in points)
print(f"wrote {dst}: {len(points)} tiles, {sum(len(p) for p in points)} ant points ({dst.stat().st_size / 1e6:.0f} MB)")
