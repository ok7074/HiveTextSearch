import functools
import mlflow
from pathlib import Path

EXPERIMENT_NAME = "HiveTextSearch"
# Fixed location, so runs land in one database whichever folder they start from.
TRACKING_DB = Path(__file__).resolve().parent.parent / "mlflow.db"

mlflow.set_tracking_uri(f"sqlite:///{TRACKING_DB.as_posix()}")
mlflow.set_experiment(EXPERIMENT_NAME)
mlflow.autolog(log_model_signatures=True, log_input_examples=False, silent=True)


def track_run(run_name: str | None = None, tags: dict | None = None):
    """Decorator that wraps a function in a named MLflow run.

    Usage:
        @track_run(run_name="clip_inference")
        def run_clip(...):
            ...

        @track_run()
        def my_experiment(...):
            ...
    """
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            name = run_name or fn.__name__
            with mlflow.start_run(run_name=name, tags=tags or {}):
                mlflow.log_params({k: v for k, v in kwargs.items() if _is_loggable(v)})
                result = fn(*args, **kwargs)
                return result
        return wrapper
    return decorator


def _is_loggable(value) -> bool:
    return isinstance(value, (str, int, float, bool))
