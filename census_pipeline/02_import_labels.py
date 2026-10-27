"""Stage 02 - import hand-labelled ant points from a SLEAP project (one point per ant, skeleton "ant") for one setup.

Keeps only hand-placed points (user instances). Predicted points that were never accepted in the GUI are ignored,
so frames that started from pre-labels count only what the labeller accepted or added.
Video paths are re-pointed to --video_dir (the labeller's machine had different paths).
One frame per video is held back as a test frame, never used for training: the highest-numbered frame that was
labelled from scratch (no accepted pre-labels), so the test is not biased towards what an earlier model found.

Reads   --labels  (default ../labels_box_4k_2026.slp)
Writes  outputs/02_points_<setup_id>.slp         all hand-labelled frames, videos pointing at --video_dir
        outputs/02_points_<setup_id>_frames.tsv  one row per frame: video, frame, ants, prelabelled?, split (train/test)
A setup needs a row in setups.tsv (setup_id, points_file = 02_points_<setup_id>.slp, resize_to_standard = auto).

Run:  /home/tom/.local/share/uv/tools/sleap/bin/python3 02_import_box_labels.py
"""
import argparse, csv
from pathlib import Path
import sleap_io as sio

HERE = Path(__file__).resolve().parent
DROPBOX = HERE.parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--labels", default=str(DROPBOX / "SLEAP-test" / "labels_box_4k_2026.slp"))
ap.add_argument("--video_dir", default=str(DROPBOX / "SLEAP-test"), help="folder holding the labelled videos on this machine")
ap.add_argument("--setup_id", default="box_4k_2026", help="row of setups.tsv these videos belong to (one camera/zoom/lighting setup)")
ap.add_argument("--n_test_per_video", type=int, default=1)
ap.add_argument("--out_dir", default=str(HERE / "outputs"))
a = ap.parse_args()
out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)

labels = sio.load_slp(a.labels, open_videos=False)
labels.replace_filenames(new_filenames=[str(Path(a.video_dir) / Path(v.filename).name) for v in labels.videos], open_videos=False)
missing = [v.filename for v in labels.videos if not Path(v.filename).exists()]
if missing:
    raise SystemExit(f"videos not found in --video_dir: {missing}")
if [n.name for n in labels.skeletons[0].nodes] != ["ant"]:
    raise SystemExit(f"expected a 1-node 'ant' skeleton, found {[n.name for n in labels.skeletons[0].nodes]}")

skel = labels.skeletons[0]
kept = sio.Labels(videos=labels.videos, skeletons=[skel])
rows = []
for lf in labels.labeled_frames:
    user = lf.user_instances
    if not user:
        continue
    pre = sum(i.from_predicted is not None for i in user)
    kept.append(sio.LabeledFrame(video=lf.video, frame_idx=lf.frame_idx, instances=list(user)))
    rows.append(dict(video=Path(lf.video.filename).name, frame_idx=lf.frame_idx, ants=len(user),
                     accepted_prelabels=pre, setup_id=a.setup_id, split="train"))

# hold back the highest-numbered from-scratch frame(s) of each video
for v in sorted({r["video"] for r in rows}):
    scratch = sorted([r for r in rows if r["video"] == v and r["accepted_prelabels"] == 0], key=lambda r: r["frame_idx"])
    for r in scratch[-a.n_test_per_video:]:
        r["split"] = "test"

stem = f"02_points_{a.setup_id}"
sio.save_slp(kept, str(out / f"{stem}.slp"))
with open(out / f"{stem}_frames.tsv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t"); w.writeheader(); w.writerows(rows)
n_tr = sum(r["split"] == "train" for r in rows); n_te = len(rows) - n_tr
print(f"read  {a.labels}: {len(labels.labeled_frames)} frames")
print(f"wrote {out / (stem + '.slp')}: {len(rows)} frames, {sum(r['ants'] for r in rows)} hand-labelled ants "
      f"({n_tr} train frames, {n_te} test frames)")
for r in rows:
    print(f"  {r['video']:32s} frame {r['frame_idx']:4d}: {r['ants']:3d} ants  {r['split']}{'  (started from pre-labels)' if r['accepted_prelabels'] else ''}")
known = [r["setup_id"] for r in csv.DictReader(open(HERE / "setups.tsv"), delimiter="\t")]
if a.setup_id not in known:
    print(f"NOTE: add a row to setups.tsv: {a.setup_id}<TAB>{stem}.slp<TAB>auto<TAB><description>")
