"""Stage 20 (SLEAP route) - pre-label ant centres in a SLEAP project with a trained SLEAP centroid model.

Why a script instead of the SLEAP GUI's "Run inference":
  1. The GUI resizes/pads each 3840x2160 frame to the model's training size (5648x5500) -> GPU out of memory.
  2. Top-down inference then sends each centre to the centered-instance (pose) model, which finds no body points
     on these small backlit ants, so almost every ant is dropped (12 of ~300 kept on EG-07 frame 0).
  3. Centroid-only inference in sleap-nn 0.1.2 / 0.2.0 needs existing labels and crashes on new frames.
This script runs only the centroid network, on overlapping tiles at full resolution, rescaled so ants match the
size the model was trained on, and writes one predicted point per ant (1-node skeleton "ant") into a new .slp.
Open that file in the SLEAP GUI, double-click predictions to accept them, delete false ones, add missed ants.

Run with the SLEAP GUI's Python (same sleap-io version as the GUI):
  /home/tom/.local/share/uv/tools/sleap/bin/python3 20_sleap_prelabel_centroids.py \
      --slp ../<project>.slp --out ../ant_census_prelabels_n110.slp --video_index 5 --frames 0
"""
import argparse
from pathlib import Path
import numpy as np, cv2, torch
import sleap_io as sio
from omegaconf import OmegaConf
from sleap_nn.training.ligon (same sleap-io version as the GUI):
  /home/tom/.local/share/uv/toolhtning_modules import CentroidLightningModule

ap = argparse.ArgumentParser()
ap.add_argument("--slp", required=True, help="input SLEAP project (videos already added)")
ap.add_argument("--out", required=True, help="output .slp (never the same as --slp)")
ap.add_argument("--model", default="/media/tom/MSATA_500GB/Dropbox/PoseIDAnt/SLEAP_Project/models/260530_201011.centroid.n=110")
ap.add_argument("--video_index", type=int, nargs="*", default=None, help="default: all videos")
ap.add_argument("--frames", type=int, nargs="*", default=[0], help="frame indices to pre-label in each video")
ap.add_argument("--scale", type=float, default=1.45, help="absolute resize so ants match training size (~45 -> ~65 px)")
ap.add_argument("--peak_thr", type=float, default=0.2)
ap.add_argument("--min_dist_px", type=float, default=14, help="minimum distance between two ant centres (native px)")
ap.add_argument("--tile", type=int, default=1280, help="native-pixel tile size")
ap.add_argument("--overlap", type=int, default=96)
ap.add_argument("--wall_band_px", type=int, default=160, help="keep points within this distance outside the lit floor; -1 = keep all")
a = ap.parse_args()
if Path(a.out).resolve() == Path(a.slp).resolve():
    raise SystemExit("--out must differ from --slp (the input project is never overwritten)")

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
grey = bool(cfg.data_config.preprocessing.ensure_grayscale)


def confidence_map(img):
    """Native-resolution centroid confidence map, built tile by tile (each tile keeps only its core)."""
    H, W = img.shape[:2]; cm = np.zeros((H, W), np.float32); step = a.tile - 2 * a.overlap
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            ya, xa = max(0, y0 - a.overlap), max(0, x0 - a.overlap)
            yb, xb = min(H, y0 + step + a.overlap), min(W, x0 + step + a.overlap)
            t = cv2.resize(img[ya:yb, xa:xb], None, fx=a.scale, fy=a.scale, interpolation=cv2.INTER_LINEAR)
            th, tw = t.shape[:2]
            t = cv2.copyMakeBorder(t, 0, (-th) % 16, 0, (-tw) % 16, cv2.BORDER_REFLECT)
            x = torch.from_numpy(t).float().div(255)
            x = (x[None, None] if x.ndim == 2 else x.permute(2, 0, 1)[None]).to(dev)
            with torch.no_grad():
                c = net.model(x)["CentroidConfmapsHead"][0, 0].float().cpu().numpy()
            c = cv2.resize(c, (t.shape[1], t.shape[0]), interpolation=cv2.INTER_LINEAR)[:th, :tw]
            c = cv2.resize(c, (xb - xa, yb - ya), interpolation=cv2.INTER_LINEAR)
            cy, cx = y0 - ya, x0 - xa; h, w = min(step, H - y0), min(step, W - x0)
            cm[y0:y0 + h, x0:x0 + w] = c[cy:cy + h, cx:cx + w]
    return cm


def lit_arena(gray):
    """Lit box floor (largest bright region) widened by the wall band; points outside are margin false positives."""
    s = cv2.resize(gray, None, fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)
    s = cv2.morphologyEx(s, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    _, b = cv2.threshold(s, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    n, lab, st, _ = cv2.connectedComponentsWithStats(b)
    floor = (lab == 1 + np.argmax(st[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
    r = max(1, a.wall_band_px // 8)
    floor = cv2.dilate(floor, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1,) * 2))
    return cv2.resize(floor, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_NEAREST) > 0


labels = sio.load_slp(a.slp)
if labels.skeletons and [n.name for n in labels.skeletons[0].nodes] != ["ant"]:
    raise SystemExit(f"project skeleton is {[n.name for n in labels.skeletons[0].nodes]}; this script writes a 1-node 'ant' skeleton")
skel = labels.skeletons[0] if labels.skeletons else sio.Skeleton(nodes=[sio.Node("ant")], name="ant_point")
if not labels.skeletons:
    labels.skeletons.append(skel)
done = {(id(lf.video), lf.frame_idx) for lf in labels.labeled_frames}
for vi in (a.video_index if a.video_index is not None else range(len(labels.videos))):
    video = labels.videos[vi]
    for fi in a.frames:
        if (id(video), fi) in done:
            print(f"video {vi} frame {fi}: already has labels, skipped"); continue
        img = np.asarray(video[fi])
        g = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
        cm = confidence_map(g if grey else img)
        k = int(2 * a.min_dist_px + 1)
        ys, xs = np.where((cm >= cv2.dilate(cm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))) & (cm > a.peak_thr))
        keep = lit_arena(g)[ys, xs] if a.wall_band_px >= 0 else np.ones(len(xs), bool)
        insts = [sio.PredictedInstance.from_numpy(points_data=np.array([[float(x), float(y)]]), skeleton=skel,
                                                  point_scores=np.array([float(cm[y, x])]), score=float(cm[y, x]))
                 for x, y in zip(xs[keep], ys[keep])]
        labels.append(sio.LabeledFrame(video=video, frame_idx=fi, instances=insts))
        print(f"video {vi} ({Path(video.filename).name}) frame {fi}: {len(insts)} ants pre-labelled ({(~keep).sum()} dropped outside the box)")
sio.save_slp(labels, a.out)
print(f"wrote {a.out}")
