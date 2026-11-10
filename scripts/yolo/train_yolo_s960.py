"""Same pipeline as train_yolo.py (Optuna trials, final training, one test evaluation),
but with the larger YOLO26s model at image size 960, so small bees keep more pixels.

Time caps keep the whole run inside Kaggle's 12-hour session limit:
Optuna stops starting new trials after TUNE_HOURS, and the final training stops after FINAL_HOURS.
The first trial reuses the best settings of the yolo26n run, so tuning starts from a known good point.

Usage (GPU runtime):
    python scripts/yolo/train_yolo_s960.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import track_run
from train_yolo import TRAIN_ARGS, run

MODEL_NAME = "yolo26s.pt"
# batch 8: 960 px images need about 2.25x the GPU memory of 640. Ultralytics accumulates gradients
# up to a nominal batch of 64, so the tuned learning rate still fits.
# cache="ram": the 960 px disk cache would be about 11 GB; Ultralytics skips caching if RAM is short.
S960_ARGS = {**TRAIN_ARGS, "imgsz": 960, "batch": 8, "cache": "ram"}
TUNE_HOURS = 4.5
FINAL_HOURS = 4.5
# Best settings of the yolo26n run at 640 (Kaggle, MLflow run "train_yolo").
N640_BEST = {
    "lr0": 0.006859416411328701,
    "lrf": 0.1117255124590943,
    "momentum": 0.9698746541326323,
    "weight_decay": 0.0011103525577197586,
}


@track_run(run_name="train_yolo_s960")
def main() -> None:
    run(MODEL_NAME, S960_ARGS, tune_hours=TUNE_HOURS, final_hours=FINAL_HOURS, start_params=N640_BEST)


if __name__ == "__main__":
    main()
