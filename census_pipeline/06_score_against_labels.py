"""Stage 06 - score predicted ant points against hand-labelled points on the same frames.

Each predicted point is paired with at most one labelled point (closest pairs first, optimal assignment) if they are
within --match_px. Paired = found; unpaired labels = missed ants; unpaired predictions = false detections.
Reported per frame and in total: labelled, predicted, found, recall (share of labelled ants found), precision
(share of predictions that are real ants), and count error (predicted - labelled, in %).

Reads   outputs/02_points_<setup>.slp (hand labels), outputs/05_pred_<run>.slp for each --runs
Writes  outputs/06_scores.tsv   one row per run x frame, plus one 'ALL' row per run

Run:  /home/tom/.local/share/uv/tools/sleap/bin/python3 06_score_against_labels.py --runs n110_x1.45 box_only box_petri
"""
import argparse, csv
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
import sleap_io as sio

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--runs", nargs="+", required=True)
ap.add_argument("--setup", default="box_4k_2026", help="setup whose hand labels (outputs/02_points_<setup>.slp) are the truth")
ap.add_argument("--match_px", type=float, default=20, help="max distance between a predicted and a labelled point (about 0.4 ant lengths in box videos)")
a = ap.parse_args()


def points(slp):
    lab = sio.load_slp(slp, open_videos=False)
    return {(Path(lf.video.filename).name, lf.frame_idx): np.array([i.numpy()[0] for i in lf.instances], float).reshape(-1, 2)
            for lf in lab.labeled_frames}


truth = points(str(HERE / "outputs" / f"02_points_{a.setup}.slp")); rows = []
for run in a.runs:
    pred = points(str(HERE / "outputs" / f"05_pred_{run}.slp"))
    tot = dict(labelled=0, predicted=0, found=0)
    for key in sorted(pred):
        t, p = truth[key], pred[key]
        d = np.linalg.norm(t[:, None] - p[None], axis=2) if len(t) and len(p) else np.zeros((len(t), len(p)))
        r, c = linear_sum_assignment(np.where(d <= a.match_px, d, 1e6)) if d.size else ([], [])
        found = int(sum(d[i, j] <= a.match_px for i, j in zip(r, c)))
        rows.append(dict(run=run, video=key[0], frame_idx=key[1], labelled=len(t), predicted=len(p), found=found,
                         recall=round(found / max(len(t), 1), 3), precision=round(found / max(len(p), 1), 3),
                         count_error_pct=round(100 * (len(p) - len(t)) / max(len(t), 1), 1)))
        for k in tot: tot[k] += rows[-1][k]
    rows.append(dict(run=run, video="ALL", frame_idx="", **tot, recall=round(tot["found"] / tot["labelled"], 3),
                     precision=round(tot["found"] / max(tot["predicted"], 1), 3),
                     count_error_pct=round(100 * (tot["predicted"] - tot["labelled"]) / tot["labelled"], 1)))
with open(HERE / "outputs" / "06_scores.tsv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t"); w.writeheader(); w.writerows(rows)
print(f"{'run':14s} {'video':32s} {'frame':>5s} {'labelled':>8s} {'pred':>5s} {'found':>5s} {'recall':>6s} {'precis':>6s} {'count err%':>10s}")
for r in rows:
    print(f"{r['run']:14s} {r['video']:32s} {str(r['frame_idx']):>5s} {r['labelled']:8d} {r['predicted']:5d} {r['found']:5d} "
          f"{r['recall']:6.2f} {r['precision']:6.2f} {r['count_error_pct']:10.1f}")
