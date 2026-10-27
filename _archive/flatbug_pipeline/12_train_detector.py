"""Stage 12 (training) - fine-tune a 4-class detector (worker, queen, larva, pupa) from flat-bug's insect weights.

flat-bug M v2 is a YOLO26m segmentation network trained on >100k arthropods; its backbone and neck are loaded into a
YOLO26m detection network with 4 classes, then trained on the stage 11 tiles. fp32 only: the GTX 1660 has no tensor
cores and Ultralytics switches mixed precision off on it automatically.

Writes <project>/<name>/weights/best.pt  (use it in stage 02 with --weights ... --backend yolo)

Run:  python 12_train_detector.py --data /media/tom/MSATA_500GB/ant_census_work/datasets/round1/data.yaml
"""
import argparse, os
os.environ.setdefault("YOLO_AUTOINSTALL", "false")      # never let Ultralytics pip-install into the environment
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/media/tom/MSATA_500GB/ant_census_work/datasets/round1/data.yaml")
ap.add_argument("--init_weights", default="/media/tom/MSATA_500GB/ant_census_work/weights/flat_bug_M_v2.pt")
ap.add_argument("--arch", default="yolo26m.yaml", help="must match the init weights' family (flat-bug M v2 = YOLO26m)")
ap.add_argument("--epochs", type=int, default=150)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=4)
ap.add_argument("--project", default="/media/tom/MSATA_500GB/ant_census_work/models")
ap.add_argument("--name", default="census_detector_round1")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

model = YOLO(a.arch).load(a.init_weights)               # copies every layer whose name and shape match
model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, project=a.project, name=a.name, seed=a.seed,
            deterministic=True, amp=False, max_det=1000, patience=40, exist_ok=True, plots=True, workers=4,
            fliplr=0.5, flipud=0.5, degrees=0.0, mosaic=1.0, close_mosaic=20, hsv_h=0.0, hsv_s=0.2, hsv_v=0.3,
            scale=0.2, translate=0.1)
