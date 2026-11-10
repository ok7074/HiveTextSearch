from typing import Callable

import mlflow
import optuna


def suggest_hyperparameters(trial: optuna.Trial) -> dict:
    """Sample one set of YOLO training hyperparameters."""
    return {
        "lr0": trial.suggest_float("lr0", 1e-4, 1e-1, log=True),
        "lrf": trial.suggest_float("lrf", 0.01, 0.5, log=True),
        "momentum": trial.suggest_float("momentum", 0.8, 0.98),
        "weight_decay": trial.suggest_float("weight_decay", 1e-5, 1e-2, log=True),
    }


def tune(
    score: Callable[[dict], float],
    n_trials: int,
    seed: int = 0,
    timeout_hours: float | None = None,
    start_params: dict | None = None,
) -> optuna.Study:
    """
    Find the hyperparameters that maximise score(params) over n_trials.
    Each trial runs in its own nested MLflow run, so its logs stay separate from the other trials.
    timeout_hours stops starting new trials after that time; the running trial still finishes.
    start_params, if given, is tried first, e.g. the best settings of an earlier run.
    """
    def objective(trial: optuna.Trial) -> float:
        params = suggest_hyperparameters(trial)
        with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
            value = score(params)
            mlflow.log_metric("objective", value)
        return value

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    if start_params:
        study.enqueue_trial(start_params)
    study.optimize(objective, n_trials=n_trials, timeout=timeout_hours * 3600 if timeout_hours else None)
    return study
