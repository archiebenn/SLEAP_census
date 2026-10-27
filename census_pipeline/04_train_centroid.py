"""Stage 04 - fine-tune a SLEAP centroid model (one point per ant) on a stage 03 tile set.

Starts from an existing centroid model's weights (default: the current box model, outputs/04_model_box_only; the first
box model itself started from the PoseIDAnt n=110 model, which you can still pass with --init_model) and trains on
1024-px tiles at the standard ant size, so preprocessing scale is 1.0 and there is no frame resizing or padding.
The training config is the n=110 config with only these changes: training data, skeleton (1 node "ant"),
scale 1.0 without max_height/max_width, batch size, epochs and the output folder.
10% of tiles are set aside by sleap-nn for validation (used for early stopping only; stage 06 does the real test).

Reads   outputs/03_tiles_<name>.pkg.slp, the --init_model folder's training_config.yaml and best.ckpt
Writes  outputs/04_model_<name>/  (best.ckpt, training_config.yaml, training_log.csv, ...)

Run:  /home/tom/.local/share/uv/tools/sleap/bin/python3 04_train_centroid.py --name box_petri
"""
import argparse, subprocess, sys
from pathlib import Path
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
N110 = HERE.parents[1] / "PoseIDAnt" / "SLEAP_Project" / "models" / "260530_201011.centroid.n=110"
ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True, help="tile set from stage 03 (outputs/03_tiles_<name>.pkg.slp)")
ap.add_argument("--init_model", default=str(HERE / "outputs" / "04_model_box_only"),
                help=f"model folder whose weights start the training (the original start was {N110})")
ap.add_argument("--max_epochs", type=int, default=100)
ap.add_argument("--patience", type=int, default=15, help="stop after this many epochs without validation improvement")
ap.add_argument("--batch_size", type=int, default=4)
ap.add_argument("--steps_per_epoch", type=int, default=150)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--workers", type=int, default=8, help="parallel data-loading processes (CPU augmentation is the bottleneck)")
a = ap.parse_args()

out = HERE / "outputs"
tiles = out / f"03_tiles_{a.name}.pkg.slp"
if not tiles.exists():
    raise SystemExit(f"{tiles} not found - run stage 03 with --name {a.name}")
if not (Path(a.init_model) / "best.ckpt").exists():
    raise SystemExit(f"no best.ckpt in --init_model {a.init_model}")
if Path(a.init_model).resolve() == (out / f"04_model_{a.name}").resolve():
    raise SystemExit("--name would overwrite the --init_model folder; choose a new name")
cfg = OmegaConf.load(Path(a.init_model) / "training_config.yaml")
ckpt = str(Path(a.init_model) / "best.ckpt")
d = cfg.data_config
d.train_labels_path = [str(tiles)]; d.val_labels_path = None; d.validation_fraction = 0.1
d.preprocessing.max_height = None; d.preprocessing.max_width = None; d.preprocessing.scale = 1.0
d.skeletons = [{"nodes": [{"name": "ant"}], "edges": [], "symmetries": [], "name": "ant_point"}]
m = cfg.model_config
m.pretrained_backbone_weights = ckpt; m.pretrained_head_weights = ckpt
m.head_configs.centroid.confmaps.anchor_part = None
t = cfg.trainer_config
t.max_epochs = a.max_epochs; t.early_stopping.patience = a.patience; t.seed = a.seed
t.train_data_loader.batch_size = a.batch_size; t.val_data_loader.batch_size = a.batch_size
t.train_data_loader.num_workers = a.workers; t.val_data_loader.num_workers = a.workers
t.train_steps_per_epoch = a.steps_per_epoch; t.min_train_steps_per_epoch = a.steps_per_epoch
t.ckpt_dir = str(out); t.run_name = f"04_model_{a.name}"; t.wandb.name = t.run_name
t.visualize_preds_during_training = False
cfg_path = out / f"04_model_{a.name}.config.yaml"
OmegaConf.save(cfg, cfg_path)
print(f"training {t.run_name} from {a.init_model} on {tiles}")
sys.exit(subprocess.call([str(Path(sys.executable).parent / "sleap-nn"), "train", "--config", str(cfg_path)]))
