"""Stage 05 - predict one point per ant with a centroid model, on full-resolution frames cut into tiles.

Why not SLEAP's own inference: on 4K frames it resizes/pads to the model's training size (GPU out of memory), and
sleap-nn 0.1.2 cannot run a centroid model alone on new frames. This stage runs only the centroid network, tile by
tile, after resizing frames by --scale, then finds one confidence peak per ant and drops points in the dark area
outside the lit box (keeping a wall band).

Frames: either labelled frames of a setup (default: its held-back test frames, for stage 06), or with --videos,
frames sampled from whole videos (--fps per second, at most --max_frames per video), for counting (stage 07).
Resize: the setup's factor from stage 02b, so ants reach the standard size the models were trained at. --scale
overrides it (e.g. the old n=110 model needs box frames x1.45).

Reads   outputs/02b_ant_length.tsv; outputs/02_points_<setup>.slp + _frames.tsv (labelled-frame mode)
Writes  outputs/05_pred_<run>.slp   predicted points (skeleton "ant"); open in SLEAP to check them
        outputs/05_pred_<run>.tsv   one row per frame: video, frame, predicted ants

Run:  python 05_predict_points.py --model outputs/04_model_box_only --run test_box_only
      python 05_predict_points.py --model outputs/04_model_box_only --run census_oct --videos "../*.MP4"
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2, torch
import sleap_io as sio
from omegaconf import OmegaConf
from sleap_nn.training.lightning_modules import CentroidLightningModule

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True, help="model folder with best.ckpt and training_config.yaml")
ap.add_argument("--run", required=True, help="name for the output files")
ap.add_argument("--setup", default="box_4k_2026", help="setups.tsv row the frames come from (sets the resize factor)")
ap.add_argument("--scale", type=float, default=None, help="override the setup's resize factor")
ap.add_argument("--split", default="test", help="labelled-frame mode: which frames to predict ('all' for every frame)")
ap.add_argument("--videos", default=None, help='video-file pattern, e.g. "../*.MP4": predict sampled frames of whole videos')
ap.add_argument("--fps", type=float, default=1.0, help="video mode: frames sampled per second")
ap.add_argument("--max_frames", type=int, default=20, help="video mode: at most this many frames per video")
ap.add_argument("--peak_thr", type=float, default=0.2)
ap.add_argument("--min_dist_px", type=float, default=14, help="minimum distance between two ant centres (frame px)")
ap.add_argument("--tile", type=int, default=1280, help="tile size in frame px")
ap.add_argument("--overlap", type=int, default=96)
ap.add_argument("--wall_band_px", type=int, default=160, help="keep points this far outside the lit floor; -1 = keep all")
a = ap.parse_args()
import csv as _csv, glob
OUT = HERE / "outputs"
if a.scale is None:
    f02b = {r["setup_id"]: float(r["resize_to_standard"]) for r in _csv.DictReader(open(OUT / "02b_ant_length.tsv"), delimiter="\t")}
    if a.setup not in f02b:
        raise SystemExit(f"no resize factor for setup {a.setup} in outputs/02b_ant_length.tsv - run stage 02b")
    a.scale = f02b[a.setup]

M = Path(a.model); cfg = OmegaConf.load(M / "training_config.yaml"); tc = cfg.trainer_config
bt = [k for k, v in cfg.model_config.backbone_config.items() if v is not None][0]
dev = "cuda" if torch.cuda.is_available() else "cpu"
net = CentroidLightningModule.load_from_checkpoint(
    checkpoint_path=str(M / "best.ckpt"), model_type="centroid", backbone_type=bt,
    backbone_config=cfg.model_config.backbone_config, head_configs=cfg.model_config.head_configs,
    pretrained_backbone_weights=None, pretrained_head_weights=None, init_weights=cfg.model_config.init_weights,
    lr_scheduler=tc.lr_scheduler, online_mining=tc.online_hard_keypoint_mining.online_mining,
    hard_to_easy_ratio=tc.online_hard_keypoint_mining.hard_to_easy_ratio,
    min_hard_keypoints=tc.online_hard_keypoint_mining.min_hard_keypoints,
    max_hard_keypoints=tc.online_hard_keypoint_mining.max_hard_keypoints,
    loss_scale=tc.online_hard_keypoint_mining.loss_scale, optimizer=tc.optimizer_name,
    learning_rate=tc.optimizer.lr, amsgrad=tc.optimizer.amsgrad, map_location=dev, weights_only=False).eval().to(dev)


def confidence_map(g):
    """Frame-resolution confidence map built tile by tile; each tile keeps only its core."""
    H, W = g.shape; cm = np.zeros((H, W), np.float32); step = a.tile - 2 * a.overlap
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            ya, xa = max(0, y0 - a.overlap), max(0, x0 - a.overlap)
            yb, xb = min(H, y0 + step + a.overlap), min(W, x0 + step + a.overlap)
            t = cv2.resize(g[ya:yb, xa:xb], None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_AREA if a.scale < 1 else cv2.INTER_LINEAR)
            th, tw = t.shape
            t = cv2.copyMakeBorder(t, 0, (-th) % 16, 0, (-tw) % 16, cv2.BORDER_REFLECT)
            with torch.no_grad():
                c = net.model(torch.from_numpy(t).float().div(255)[None, None].to(dev))["CentroidConfmapsHead"][0, 0].float().cpu().numpy()
            c = cv2.resize(c, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_LINEAR)[:th, :tw]
            c = cv2.resize(c, (xb - xa, yb - ya), interpolation=cv2.INTER_LINEAR)
            cy, cx = y0 - ya, x0 - xa; h, w = min(step, H - y0), min(step, W - x0)
            cm[y0:y0 + h, x0:x0 + w] = c[cy:cy + h, cx:cx + w]
    return cm


def lit_arena(g):
    s = cv2.resize(g, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    s = cv2.morphologyEx(s, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    _, b = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b)
    floor = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    r = max(1, a.wall_band_px // 8)
    floor = cv2.dilate(floor, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    return cv2.resize(floor, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


def frames_to_predict():
    """(video, frame index) pairs: sampled from whole videos, or the setup's labelled frames of --split."""
    if a.videos:
        for path in sorted(glob.glob(a.videos)):
            v = sio.load_video(path)
            step = max(1, round((v.fps or 25) / a.fps))
            for k in list(range(0, len(v), step))[:a.max_frames]:
                yield v, k
        return
    labels = sio.load_slp(str(OUT / f"02_points_{a.setup}.slp"))
    ftsv = OUT / f"02_points_{a.setup}_frames.tsv"
    pick = None if a.split == "all" else {(r["video"], int(r["frame_idx"])) for r in csv.DictReader(open(ftsv), delimiter="\t") if r["split"] == a.split}
    for lf in labels.labeled_frames:
        if pick is None or (Path(lf.video.filename).name, lf.frame_idx) in pick:
            yield lf.video, lf.frame_idx


skel = sio.Skeleton(nodes=[sio.Node("ant")], name="ant_point")
pred = sio.Labels(skeletons=[skel]); rows = []
for video, fidx in frames_to_predict():
    if video not in pred.videos:
        pred.videos.append(video)
    key = (Path(video.filename).name, fidx)
    img = np.asarray(video[fidx])
    g = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 and img.shape[2] == 3 else img.reshape(img.shape[:2]))
    cm = confidence_map(g)
    k = int(2 * a.min_dist_px + 1)
    ys, xs = np.where((cm >= cv2.dilate(cm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))) & (cm > a.peak_thr))
    keep = lit_arena(g)[ys, xs] if a.wall_band_px >= 0 else np.ones(len(xs), bool)
    pred.append(sio.LabeledFrame(video=video, frame_idx=fidx, instances=[
        sio.PredictedInstance.from_numpy(points_data=np.array([[float(x), float(y)]]), skeleton=skel,
                                         point_scores=np.array([float(cm[y, x])]), score=float(cm[y, x]))
        for x, y in zip(xs[keep], ys[keep])]))
    rows.append(dict(video=key[0], frame_idx=key[1], predicted_ants=int(keep.sum()), dropped_outside_box=int((~keep).sum())))
    print(f"{key[0]:32s} frame {key[1]:4d}: {int(keep.sum())} ants predicted", flush=True)
out = OUT
sio.save_slp(pred, str(out / f"05_pred_{a.run}.slp"))
with open(out / f"05_pred_{a.run}.tsv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t"); w.writeheader(); w.writerows(rows)
print(f"wrote {out / f'05_pred_{a.run}.slp'} ({len(rows)} frames; model {a.model}, scale {a.scale})")
