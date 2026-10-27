"""Stage 07 - turn per-frame ant counts from whole videos (stage 05 with --videos) into one census row per video.

Videos are named <ColonyID>_<YYYY-MM-DD>.MP4; colony and date are read from the name. Ants hide under each other in
some frames, so single frames tend to undercount: the census value is an upper percentile (default 90th) of the
per-frame counts. The median, minimum and maximum are reported too, so unusual videos stand out.

Reads   outputs/05_pred_<run>.tsv
Writes  outputs/07_census_<run>.tsv   one row per video: colony, date, frames, ants (percentile), median, min, max

Run:  python 05_predict_points.py --model outputs/04_model_box_only --run census_oct --videos "../*.MP4"
      python 07_count_videos.py --run census_oct
"""
import argparse, csv
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True, help="stage 05 run name (video mode)")
ap.add_argument("--percentile", type=float, default=90, help="per-frame count summarised as this percentile")
a = ap.parse_args()

per = {}
for r in csv.DictReader(open(HERE / "outputs" / f"05_pred_{a.run}.tsv"), delimiter="\t"):
    per.setdefault(r["video"], []).append(int(r["predicted_ants"]))
rows = []
for video, n in sorted(per.items()):
    colony, _, rest = Path(video).stem.partition("_")
    n = np.array(n)
    rows.append(dict(video=video, colony=colony, date=rest, frames=len(n), ants=int(round(np.percentile(n, a.percentile))),
                     median=int(np.median(n)), min=int(n.min()), max=int(n.max())))
with open(HERE / "outputs" / f"07_census_{a.run}.tsv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]), delimiter="\t"); w.writeheader(); w.writerows(rows)
print(f"{'colony':16s} {'date':12s} {'frames':>6s} {'ants':>5s} {'median':>6s} {'min':>4s} {'max':>4s}")
for r in rows:
    print(f"{r['colony']:16s} {r['date']:12s} {r['frames']:6d} {r['ants']:5d} {r['median']:6d} {r['min']:4d} {r['max']:4d}")
print(f"wrote outputs/07_census_{a.run}.tsv (ants = {a.percentile:.0f}th percentile of per-frame counts)")
