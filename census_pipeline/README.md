# Ant census with SLEAP: one point per ant

---

## INSTRUCTIONS - READ THIS FIRST

This tool is shared by the lab and is meant to **grow over time**: every labelled frame anyone adds makes the
model better for everyone. It runs on its own: you need SLEAP, an NVIDIA graphics card, and the commands below.

### How the labels are organised

- There is **one SLEAP project per recording setup**. A *setup* = one combination of camera, zoom, camera height
  and lighting. Projects live in `Dropbox/SLEAP-test/` and are named `labels_<setup_id>.slp`.
- `setups.tsv` (in this folder) lists every setup: its id, its imported points file, and its resize factor.
- The pipeline imports each project separately (step 02), then **combines all setups when training** (step 03).

Current projects:

| Project file | Setup id | What it is |
|---|---|---|
| `labels_box_4k_2026.slp` | `box_4k_2026` | Plastic boxes filmed from below, backlit, Sony FDR-AX53 4K, Sept 2026 |

### What you need to do

1. **Decide if your videos are an existing setup or a new one.**
   - Same camera, zoom, height and lighting as an existing setup -> **existing setup**.
   - Anything different (new camera, zoom, lighting panel, box type) -> **new setup**.
2. **Existing setup:** open that setup's project in SLEAP, add your videos (*File -> Add Videos*), label frames, save
   (same file name). Then run step 02 for that setup.
3. **New setup:** create a new SLEAP project with a one-node skeleton called `ant`, add your videos, label frames, and
   save it as `Dropbox/SLEAP-test/labels_<new_setup_id>.slp` (e.g. `labels_box_led_2026.slp`). Run step 02 with
   `--setup_id <new_setup_id>`, add the row it prints to `setups.tsv`, then run step 02b and check its picture.
4. **Label properly** (this matters more than anything else):
   - one point per ant, on the **middle of the ant (the thorax)**;
   - label **every** visible ant in the frame, including piles, ants on the walls, and the queen;
   - about 10 frames per new setup to start, spread across colonies and times, with plenty of dense piles.
5. **Retrain and check** (commands below): steps 02 -> 02b -> 03 -> 04 -> 05 -> 06. Use the new model only if step 06
   shows it is at least as good as the current one on the test frames.
6. **Record what you did** in the log at the bottom of this file (date, who, setup, frames added, new scores).

### What you must NOT do

- **Do not** put videos from a different camera, zoom, height or lighting into an existing project: make a new
  setup. Mixing them gives the setup the wrong ant size.
- **Do not** leave ants unlabelled in a frame you label. A half-labelled frame teaches the model that ants are
  background. If a frame is too hard, skip the whole frame.
- **Do not** let two people work on the same `.slp` project at the same time. Dropbox makes "conflicted copy" files
  and labels get lost. Agree who is labelling which project.
- **Do not** rename or move project files, or create `.vNNN` copies: the pipeline looks for `labels_<setup_id>.slp`.
- **Do not** delete or overwrite `outputs/04_model_*` folders. Train new models under a new `--name`.
- **Do not** edit anything in `outputs/` by hand: rerun the step that made it.
- **Do not** write anything into `Dropbox/PoseIDAnt/`. That is a separate project; this pipeline only reads it.
- **Do not** use SLEAP's own "Run inference" button on 4K videos (it runs out of memory and loses ants). Use step 05.

---

Counts ants in videos from **different recording setups** (cameras, zoom, lighting, box types). A SLEAP
**centroid model** marks one point on every ant; the count is the number of points. The model learns from
hand-labelled frames. Before training, every setup is resized so a worker ant is the same size in pixels
everywhere (50 px long, as in the 2026 box videos). That resize factor is measured automatically from the labels.

Everything is in this folder (`Dropbox/SLEAP-test/census_pipeline`). Each step is one numbered script that writes
its results to `outputs/`, with the step number at the start of the file name. Settings are command-line options,
described at the top of each script. Never edit an output by hand: change the options and run the step again.

Python: the one that came with your SLEAP install (it has sleap-io and sleap-nn). Training needs an NVIDIA graphics
card. On Tom's desktop: `/home/tom/.local/share/uv/tools/sleap/bin/python3`.

## The steps

| Step | What it does |
|---|---|
| `01_convert_pose_labels_to_points.py` | Tom's petri-dish labels (head, thorax, abdomen) -> one point per ant. Already run; needs the PoseIDAnt folder |
| `02_import_labels.py` | Reads a SLEAP labels file for one setup (hand-placed points only); holds back 1 test frame per video |
| `02b_measure_ant_length.py` | Measures worker length per setup and sets its resize factor (50 px / length) |
| `03_build_training_tiles.py` | Resizes each setup to the standard ant size and cuts frames into 1136-px tiles that fit the GPU |
| `04_train_centroid.py` | Trains the model, starting from the previous model |
| `05_predict_points.py` | Marks ants on held-back test frames, or on frames sampled from whole videos (`--videos`) |
| `06_score_against_labels.py` | Compares predictions with your test frames: ants missed, false detections, count error |
| `07_count_videos.py` | One census row per video: colony and date from the file name, ant count |

`setups.tsv` lists each setup: its id, its points file from step 02, its resize factor (`auto` = measured in step
02b, or a number to fix it by hand), and a description. A **setup** is one camera / zoom / height / lighting
combination; if any of those change, ant size in pixels changes, so give it a new setup id.

## Counting ants in new videos

Videos must be named `<ColonyID>_<YYYY-MM-DD>.MP4`.

```bash
cd ~/Dropbox/university/phd/others/2026/SLEAP-test/census_pipeline
PY=/path/to/the/python/that/came/with/your/sleap/install

# mark ants on 20 frames per video (1 per second), using the current model
$PY 05_predict_points.py --model outputs/04_model_box_only --setup box_4k_2026 --run census_2026-10 --videos "../*.MP4"

# one row per video -> outputs/07_census_census_2026-10.tsv
$PY 07_count_videos.py --run census_2026-10
```

`05` also writes `outputs/05_pred_<run>.slp`: open it in SLEAP to see the marked ants. If the videos come from a new
setup, first label some frames and retrain (below), and give that `--setup`.

## Retraining after labelling more frames

Label in SLEAP and save the setup's project under its usual name (e.g. `labels_box_4k_2026.slp`). Give each new model
a new name (here `box_2026-11`: date-based names are easiest to follow).

```bash
cd ~/Dropbox/university/phd/others/2026/SLEAP-test/census_pipeline
PY=/path/to/the/python/that/came/with/your/sleap/install

# 02: read your labels (hand-placed points only); holds back 1 test frame per video
$PY 02_import_labels.py --labels ../labels_box_4k_2026.slp --video_dir .. --setup_id box_4k_2026

# 02b: measure ant length and set the resize factor of every setup in setups.tsv
$PY 02b_measure_ant_length.py

# 03: resize to the standard ant size and cut into training tiles
$PY 03_build_training_tiles.py --name box_2026-11 --setups box_4k_2026

# 04: train, starting from the previous model (30-60 min on a GPU)
$PY 04_train_centroid.py --name box_2026-11 --init_model outputs/04_model_box_only

# 05 + 06: mark ants on the held-back test frames and score old and new model side by side
$PY 05_predict_points.py --model outputs/04_model_box_only --run test_box_only
$PY 05_predict_points.py --model outputs/04_model_box_2026-11 --run test_box_2026-11
$PY 06_score_against_labels.py --runs test_box_only test_box_2026-11
```

Step 06 prints, per test frame and in total: ants labelled, predicted, found, the share of labelled ants found
(recall), the share of predictions that are real ants (precision), and the count error in %. Keep the new model if
it scores better, and use it for counting.

## A new setup (different camera, zoom, height or lighting)

1. Label about 10 frames from the new videos in their own SLEAP project.
2. Import them under a new setup id, then add the row that step 02 prints to `setups.tsv` (factor `auto`):

   ```bash
   $PY 02_import_labels.py --labels ../labels_box_led_2026.slp --video_dir .. --setup_id box_led_2026
   ```
3. Run step 02b and check `outputs/02b_ant_length_<setup>.jpg`: the red boxes should sit around single ants.
   The method needs dark ants on a plain, pale floor. If it fails, measure a few ants by hand in any image viewer
   and write 50 / length into `setups.tsv` instead of `auto`.
4. Before training, see how the current model copes: run step 05 on the new setup's test frames, then step 06.
5. Train on all setups together (`--setups box_4k_2026 box_led_2026` in step 03), then steps 04-06.

## Rules for labelling frames

- Click the **middle of each ant (the thorax)**.
- Label **every** visible ant in the frame, including ants in piles and on the walls. An unlabelled ant teaches the
  model that ants are background.
- Spread frames across colonies and times. The most useful frames have the hard cases: dense piles, wall ants,
  the queen.

## Results so far (8 Oct 2026)

Six held-back test frames, 1,475 hand-labelled ants. The test frames come from the same videos as the training
frames, so new videos are a harder test.

| Model | Ants found | Real detections | Count error |
|---|---|---|---|
| PoseIDAnt n=110 model, untrained on boxes (frames x1.45) | 73% | 90% | -19% |
| Trained on 12 box frames (`04_model_box_only`) | 95% | 98% | -3.3% |
| Trained on 12 box + 132 petri frames (`04_model_box_petri`) | 95% | 99% | -4.1% |

The petri frames added nothing for box videos. Dense piles are the weakest point (HH-21: about -10%).

Measured worker lengths: box videos 50 px (2,321 isolated ants); petri 262 px (1,595 ants; factor 0.19 - the
`box_petri` tiles were built at 0.2).

## Known problems

- SLEAP's own "Run inference" on 4K box videos runs out of GPU memory, and sleap-nn 0.1.2 cannot run a centroid
  model alone on new frames. Use step 05 instead.
- Step 01: only 66 of the n=110 model's 110 training frames are in the current PoseIDAnt label file; the other 44
  were labelled in an older file that is no longer available.

## Log of labelling and models

Add a line every time you add labelled frames or train a model.

| Date | Who | Setup | Frames added (total) | Model | Test result (step 06, ALL row) |
|---|---|---|---|---|---|
| 2026-10-08 | [student] | box_4k_2026 | 18 (18) | `04_model_box_only` | 95% found, 98% real, count -3.3% |
