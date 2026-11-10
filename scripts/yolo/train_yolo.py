"""Tune YOLO with Optuna, train it on the devset, then test it on the held-out hives.

Steps:
  1. Optuna trials: train on dev/train, score each trial by mAP50-95 on dev/val
  2. Train with the best hyperparameters on dev/train
  3. Evaluate that model once on test

Needs data/BeeDataset/data.yaml from scripts/yolo/split_bee_dataset.py.
Ultralytics outputs go to experiments/yolo/. Intended for a GPU runtime.

Each run is tracked in MLflow under the experiment "HiveTextSearch",
with one nested run per Optuna trial and one for the final model.

Usage:
    python scripts/yolo/train_yolo.py
"""
import os
import sys
import mlflow
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import track_run
from src.tracking import EXPERIMENT_NAME
from src.models.yolo_model import BEE_DATASET_DIR, DEFAULT_MODEL, evaluate, train
from src.optuna_tuning import tune

# Ultralytics sets its own MLflow experiment before each training; keep it on ours.
os.environ["MLFLOW_EXPERIMENT_NAME"] = EXPERIMENT_NAME

N_TRIALS = 10
TUNE_EPOCHS = 10
FINAL_EPOCHS = 60
# optimizer must not be "auto": Ultralytics then ignores the tuned lr0 and momentum.
# cache="disk" decodes each 1920x1080 frame once for all trials instead of every epoch.
TRAIN_ARGS = {"imgsz": 640, "batch": 16, "optimizer": "SGD", "cache": "disk"}


def run(
    model_name: str = DEFAULT_MODEL,
    train_args: dict = TRAIN_ARGS,
    tune_hours: float | None = None,
    final_hours: float | None = None,
    start_params: dict | None = None,
    final_name: str | None = None,
):
    """
    Tune, train and test one model; returns the final model. tune_hours and final_hours cap the time of each
    stage (None = no cap). final_name names the final run's folder under experiments/yolo/ (None = train-N).
    """
    data_yaml = BEE_DATASET_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"{data_yaml} not found; run scripts/yolo/split_bee_dataset.py first")
    mlflow.log_params({
        "model_name": model_name, "n_trials": N_TRIALS, "tune_epochs": TUNE_EPOCHS, "final_epochs": FINAL_EPOCHS,
        "tune_hours": tune_hours, "final_hours": final_hours, **train_args,
    })

    study = tune(
        lambda params: train(data_yaml, model_name, epochs=TUNE_EPOCHS, **train_args, **params).metrics.box.map,
        N_TRIALS,
        timeout_hours=tune_hours,
        start_params=start_params,
    )
    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    mlflow.log_metric("best_dev_val_map50_95", study.best_value)
    mlflow.log_metric("trials_run", len(study.trials))

    with mlflow.start_run(run_name="final", nested=True):
        # Ultralytics' time (hours) overrides epochs and fits the learning-rate schedule into that time.
        model = train(
            data_yaml, model_name, epochs=FINAL_EPOCHS, time=final_hours, name=final_name,
            **train_args, **study.best_params,
        )
        mlflow.log_metric("dev_val_map50_95", model.metrics.box.map)
        metrics = evaluate(model, data_yaml, split="test")
        mlflow.log_metrics({
            "test_map50_95": metrics.box.map,
            "test_map50": metrics.box.map50,
            "test_precision": metrics.box.mp,
            "test_recall": metrics.box.mr,
        })
    return model


@track_run(run_name="train_yolo")
def main() -> None:
    run()


if __name__ == "__main__":
    main()
