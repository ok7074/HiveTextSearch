"""Train the final bee detector: YOLO26s at 960 on BeeDataset plus the own hive frames.

Same pipeline as train_yolo_s960.py (Optuna trials on dev/val, final training, one test evaluation),
on the updated BeeDataset. The first trial reuses the best settings of the s960 run.
Afterwards the final model is also scored on the 15 own test frames alone
(data_own_test.yaml from scripts/yolo/own_frames.py add).
Weights: experiments/yolo/final_s960/weights/best.pt. Intended for a GPU runtime (Kaggle).

Time: Optuna stops starting new trials after TUNE_HOURS; the final training runs FINAL_EPOCHS (60) epochs
with no time cap, about 3.5 h going by the s960 run, so the whole run stays inside Kaggle's 12 hours.

Usage:
    python scripts/yolo/train_yolo_final.py
"""
import sys
from pathlib import Path

import mlflow

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import track_run
from src.models.yolo_model import BEE_DATASET_DIR, evaluate
from train_yolo import run
from train_yolo_s960 import MODEL_NAME, S960_ARGS, TUNE_HOURS

# Best settings of the s960 Optuna run (Kaggle, experiments/yolo_runsS960/yolo_runs/train-11/args.yaml).
S960_BEST = {
    "lr0": 0.004430375245218269,
    "lrf": 0.16409119617016144,
    "momentum": 0.9084974076928959,
    "weight_decay": 0.0004311710058685488,
}


@track_run(run_name="train_yolo_final")
def main() -> None:
    own_yaml = BEE_DATASET_DIR / "data_own_test.yaml"
    if not own_yaml.exists():
        raise FileNotFoundError(f"{own_yaml} not found; run scripts/yolo/own_frames.py add first")
    model = run(MODEL_NAME, S960_ARGS, tune_hours=TUNE_HOURS, start_params=S960_BEST, final_name="final_s960")
    metrics = evaluate(model, own_yaml, split="test")
    mlflow.log_metrics({
        "own_test_map50_95": metrics.box.map,
        "own_test_map50": metrics.box.map50,
        "own_test_precision": metrics.box.mp,
        "own_test_recall": metrics.box.mr,
    })


if __name__ == "__main__":
    main()
