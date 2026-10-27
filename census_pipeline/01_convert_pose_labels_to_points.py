"""Stage 01 - convert the petri-dish pose labels (Head, Thorax, Abdomen) into one point per ant.

Counting only needs one point per ant. Each hand-labelled ant becomes a single point at its body centre, defined
exactly as SLEAP's centroid model defines it when no anchor part is set (the n=110 centroid model's setting): the
midpoint of the bounding box of the ant's visible labelled points. sleap-nn's own function is used, so the points
sit where the existing centroid model learned to put them.

Only hand-labelled (user) instances are converted; predictions and empty frames are left out and reported.
Images are not copied: the output .slp refers to the frames embedded in the source .pkg.slp (no 2 GB duplicate).

Reads   Dropbox/PoseIDAnt/SLEAP_Project/Ants.v005_20_videos.pkg.slp      labelled petri frames, images embedded
        Dropbox/PoseIDAnt/SLEAP_Project/models/260530_201011.centroid.n=110/labels_gt.{train,val}.0.slp
                                                                         frames used by the n=110 model
Writes  outputs/01_petri_points.slp        one-node skeleton "ant"; hand-labelled points, same frames
        outputs/01_petri_points_frames.tsv one row per frame: source video, frame, ants, n=110 split membership,
                                           and whether its labels were edited after the n=110 model was trained
        outputs/01_petri_points_qc.jpg     one frame: original 3 labelled points (small) and the new centre (ring)

Run with the SLEAP GUI's Python (its sleap-io and sleap-nn versions match the GUI):
  /home/tom/.local/share/uv/tools/sleap/bin/python3 01_convert_pose_labels_to_points.py
"""
import argparse, csv
from pathlib import Path
import numpy as np, cv2, torch, h5py
import sleap_io as sio
from sleap_nn.data.instance_centroids import generate_centroids

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1] / "PoseIDAnt" / "SLEAP_Project"     # the PoseIDAnt SLEAP project holding the petri labels
ap = argparse.ArgumentParser()
ap.add_argument("--labels", default=str(PROJECT / "Ants.v005_20_videos.pkg.slp"))
ap.add_argument("--n110_dir", default=str(PROJECT / "models" / "260530_201011.centroid.n=110"),
                help="model folder whose labels_gt.train/val.0.slp define the n=110 split (for provenance only)")
ap.add_argument("--anchor_part", default=None, help="node to use as the centre instead of the bounding-box midpoint "
                                                     "(default None = same as the n=110 centroid model)")
ap.add_argument("--out_dir", default=str(HERE / "outputs"))
ap.add_argument("--qc_frame", type=int, default=0, help="index (in the output) of the labelled frame drawn in the QC image")
a = ap.parse_args()
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)


def split_frames(path):
    """(frame_idx, labelled points) for every frame of a labels_gt file, read directly with h5py: its frame-to-video
    indices point past its own video list, so neither sleap-io nor video names can identify the frames. Raw point
    coordinates equal sleap-io's (checked)."""
    with h5py.File(path, "r") as f:
        fr, inst, pts = f["frames"][:], f["instances"][:], f["points"][:]
        out = []
        for row in fr:
            rows = inst[row["instance_id_start"]:row["instance_id_end"]]
            rows = rows[rows["instance_type"] == 0]
            if not len(rows):
                continue
            p = np.concatenate([pts[r["point_id_start"]:r["point_id_end"]] for r in rows])
            xy = np.stack([p["x"], p["y"]], 1).astype(np.float64)
            out.append((int(row["frame_idx"]), xy[p["visible"]]))
    return out


def split_of(frame_idx, xy, splits, tol_px=3.0, min_share=0.8):
    """Which n=110 split a labelled frame belongs to: same frame number and >= 80% of its points within 3 px of a
    split point. 'unchanged' if every point matches exactly (labels not edited since the model was trained)."""
    xy = xy[np.isfinite(xy).all(1)]
    for name, frames in splits.items():
        for fi, ref in frames:
            if fi != frame_idx or not len(ref):
                continue
            d = np.min(np.linalg.norm(xy[:, None] - ref[None], axis=2), axis=1)
            if (d < tol_px).mean() >= min_share:
                same = len(ref) == len(xy) and bool(np.all(d < 1e-6))
                return name, "unchanged" if same else "edited_since"
    return "not_in_n110", ""


labels = sio.load_slp(a.labels)
src_skel = labels.skeletons[0]
anchor = None if a.anchor_part is None else [n.name for n in src_skel.nodes].index(a.anchor_part)
n110 = {}
for split in ("train", "val"):
    p = Path(a.n110_dir) / f"labels_gt.{split}.0.slp"
    n110[split] = split_frames(p) if p.exists() else []

skel = sio.Skeleton(nodes=[sio.Node("ant")], name="ant_point")
points_out = sio.Labels(videos=labels.videos, skeletons=[skel])
rows, n_in, n_dropped, skipped = [], 0, 0, []
for lf in labels.labeled_frames:
    user = lf.user_instances
    if not user:
        skipped.append(f"{Path(lf.video.source_video.filename if lf.video.source_video else lf.video.filename).name} frame {lf.frame_idx}")
        continue
    xy = np.stack([i.numpy() for i in user]).astype(np.float64)                 # (ants, 3 nodes, 2)
    cen = generate_centroids(torch.from_numpy(xy), anchor_ind=anchor).numpy()  # sleap-nn's own centre rule
    ok = np.isfinite(cen).all(1)
    n_in += len(user); n_dropped += int((~ok).sum())
    points_out.append(sio.LabeledFrame(video=lf.video, frame_idx=lf.frame_idx,
                                       instances=[sio.Instance.from_numpy(c[None], skeleton=skel) for c in cen[ok]]))
    split, status = split_of(lf.frame_idx, xy.reshape(-1, 2), n110)
    src = lf.video.source_video.filename if lf.video.source_video else lf.video.filename
    rows.append(dict(source_video=Path(src).name, frame_idx=lf.frame_idx, ants=int(ok.sum()),
                     ants_without_points=int((~ok).sum()), n110_split=split, labels_vs_n110=status))

# checks before writing: every visible hand-labelled ant became exactly one point
n_out = sum(len(lf.instances) for lf in points_out.labeled_frames)
assert n_out == n_in - n_dropped, (n_out, n_in, n_dropped)
assert len(points_out.labeled_frames) == len(rows)
sio.save_slp(points_out, str(out / "01_petri_points.slp"), embed=False, restore_original_videos=False)
with open(out / "01_petri_points_frames.tsv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t"); w.writeheader(); w.writerows(rows)

# re-open the written file and check it matches and its images load
check = sio.load_slp(str(out / "01_petri_points.slp"))
assert len(check.labeled_frames) == len(rows) and sum(len(lf.instances) for lf in check.labeled_frames) == n_out
lf_src = [lf for lf in labels.labeled_frames if lf.user_instances][a.qc_frame]
img = np.asarray(check.labeled_frames[a.qc_frame].image)
img = cv2.cvtColor(img[..., 0] if img.ndim == 3 and img.shape[2] == 1 else img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 or img.shape[-1] == 1 else img.copy()
for inst in lf_src.user_instances:
    for (x, y), col in zip(inst.numpy(), [(0, 200, 255), (0, 255, 0), (255, 120, 0)]):    # Head, Thorax, Abdomen
        if np.isfinite(x): cv2.circle(img, (int(x), int(y)), 6, col, -1)
for inst in check.labeled_frames[a.qc_frame].instances:
    x, y = inst.numpy()[0]; cv2.circle(img, (int(x), int(y)), 22, (0, 0, 255), 4)
cen_all = np.array([i.numpy()[0] for i in check.labeled_frames[a.qc_frame].instances])
x0, y0 = np.maximum(cen_all.min(0).astype(int) - 150, 0); x1, y1 = cen_all.max(0).astype(int) + 150
cv2.imwrite(str(out / "01_petri_points_qc.jpg"), img[y0:y1, x0:x1], [cv2.IMWRITE_JPEG_QUALITY, 85])

split_n = {s: sum(r["n110_split"] == s for r in rows) for s in ("train", "val", "not_in_n110")}
n_edit = sum(r["labels_vs_n110"] == "edited_since" for r in rows)
n_split = {s: len(f) for s, f in n110.items()}
print(f"read    {a.labels}: {len(labels.labeled_frames)} labelled frames, {n_in} hand-labelled ants")
print(f"wrote   {out / '01_petri_points.slp'}: {len(rows)} frames, {n_out} ant points "
      f"({n_dropped} ants had no visible points and were dropped)")
print(f"n=110 split ({n_split['train']} train + {n_split['val']} val frames): found {split_n['train']} train, "
      f"{split_n['val']} val ({n_edit} of them with labels edited since); {split_n['not_in_n110']} frames labelled since")
if skipped:
    print("skipped frames with no hand labels:", "; ".join(skipped))
