# Ant colony census from backlit box videos

Counts workers and queens (and, once a model is trained on your corrections, larvae and pupae) in videos named
`<ColonyID>_<YYYY-MM-DD>.MP4`. One row per video goes to `census_output/census.csv`, plus a QC image per video
(`census_output/qc/`) showing what was counted.

Python: `/media/tom/MSATA_500GB/ant_census_work/.venv/bin/python`. This is a small add-on to the PoseIDAnt tracklet
environment (Ultralytics, flat-bug, polars-lts-cpu). Never let Ultralytics auto-install: the scripts set `YOLO_AUTOINSTALL=false`.
Large intermediate files (sampled frames, detections, models) live in `/media/tom/MSATA_500GB/ant_census_work/`, outside Dropbox.

## Counting (run in order)

| Stage | What it does |
|---|---|
| `01_sample_frames.py` | 1 frame per second from each video, plus the per-pixel median (what doesn't move) and maximum (the lit floor) |
| `02_detect_ants.py` | finds every ant in every sampled frame. Default: flat-bug, a published insect detector, used as-is |
| `03_census.py` | keeps detections inside the box, sorts worker / queen / other by size and darkness, summarises over frames |

The worker estimate is the 90th percentile of the per-frame counts (`--worker_summary`). Ants hide under each other
and in piles, so single frames count too few; the median and range are also reported.

## Teaching it brood, queens and wall ants (a few corrected frames)

| Stage | What it does |
|---|---|
| `10_make_prelabels.py` | one frame per video with pre-filled points: workers from flat-bug, brood candidates (over-inclusive) |
| `../tools/annotate_points.html` | open in a browser; load the frame `.png` and its `.points.json` together; correct; Save |
| `11_build_training_set.py` | corrected points -> 640-px training tiles (only tiles you marked done) |
| `12_train_detector.py` | fine-tunes a 4-class detector (worker, queen, larva, pupa) starting from flat-bug's insect weights |

Then rerun `02_detect_ants.py --backend yolo --weights <model>/weights/best.pt --overwrite` and `03_census.py`.

### Correcting a frame in the annotation tool
- Keys `1`–`4` choose the class: worker, queen, larva, pupa. Click to add a point; right-click a point to delete it.
  Hover over a point and press `C` to change it to the current class.
- Press `G` for the 640-px grid and `N` to jump to the next unchecked tile. Fix that tile, then press `T` to mark it done.
  Only tiles marked done are used for training, so you can stop part-way.
- Brood candidates all start as "larva" and many are wrong (floor marks, debris). Delete those, switch pupae to class 4,
  and click any brood that was missed.
- Put one point on every visible ant, including ants on the walls and in piles. Save writes `<frame>.points.json` to your
  Downloads folder; move it back next to the frame image.

## Known limits
- Zero-shot flat-bug misses ants on the box walls and at the floor edge, and does not see brood or the queen reliably.
  The trained model, or keeping ants off the walls, fixes this.
- Ants and brood hidden underneath others cannot be seen from below in any frame. Check a few colonies against hand counts.
