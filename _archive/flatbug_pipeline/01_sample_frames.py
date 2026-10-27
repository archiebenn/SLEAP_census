"""Stage 01 - sample frames from each census video and build its static layers.

For every video named <ColonyID>_<YYYY-MM-DD>.MP4 this writes, under <work_root>/<video stem>/:
  frames/f0000.png ...   frames sampled at --fps (default 1 per second), full resolution
  static_median.png      per-pixel median over the sampled frames: what does not move (brood, resting ants, queen)
  static_max.png         per-pixel maximum: the lit floor with moving ants removed (used for the arena mask)
  sample.json            colony id, date, frame times and settings (the stage hand-off record)

Run:  python 01_sample_frames.py --videos "/media/tom/MSATA_500GB/Dropbox/SLEAP-test/*.MP4"
"""
import argparse, glob, json, subprocess
from pathlib import Path
import numpy as np, cv2

ap = argparse.ArgumentParser()
ap.add_argument("--videos", default="/media/tom/MSATA_500GB/Dropbox/SLEAP-test/*.MP4")
ap.add_argument("--work_root", default="/media/tom/MSATA_500GB/ant_census_work/census")
ap.add_argument("--fps", type=float, default=1.0, help="frames sampled per second of video")
ap.add_argument("--start_s", type=float, default=0.0, help="skip this many seconds at the start")
ap.add_argument("--max_frames", type=int, default=30)
ap.add_argument("--overwrite", action="store_true")
a = ap.parse_args()


def chunked_stat(frames, fn, rows=216):
    """Per-pixel statistic over a frame stack without a full float64 copy in memory."""
    out = np.empty(frames[0].shape, np.uint8)
    for y in range(0, out.shape[0], rows):
        out[y:y + rows] = fn(np.stack([f[y:y + rows] for f in frames]), axis=0).astype(np.uint8)
    return out


videos = sorted(glob.glob(a.videos))
if not videos:
    raise SystemExit(f"no videos match {a.videos}")
for v in videos:
    stem = Path(v).stem
    colony, _, date = stem.partition("_")
    out = Path(a.work_root) / stem
    if (out / "sample.json").exists() and not a.overwrite:
        print(f"{stem}: already sampled, skipping (use --overwrite)")
        continue
    (out / "frames").mkdir(parents=True, exist_ok=True)
    for old in (out / "frames").glob("f*.png"):
        old.unlink()
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(a.start_s), "-i", v, "-vf", f"fps={a.fps}",
                    "-frames:v", str(a.max_frames), "-start_number", "0", str(out / "frames" / "f%04d.png")], check=True)
    paths = sorted((out / "frames").glob("f*.png"))
    if len(paths) < 3:
        raise SystemExit(f"{stem}: only {len(paths)} frames sampled - video too short for --start_s/--fps?")
    frames = [cv2.imread(str(p)) for p in paths]
    cv2.imwrite(str(out / "static_median.png"), chunked_stat(frames, np.median))
    cv2.imwrite(str(out / "static_max.png"), chunked_stat(frames, np.max))
    rec = dict(video=str(v), stem=stem, colony=colony, date=date, fps=a.fps, start_s=a.start_s,
               frames=[dict(file=p.name, t_s=round(a.start_s + i / a.fps, 2)) for i, p in enumerate(paths)],
               width=frames[0].shape[1], height=frames[0].shape[0])
    json.dump(rec, open(out / "sample.json", "w"), indent=1)
    print(f"{stem}: colony {colony}, date {date}, {len(paths)} frames")
