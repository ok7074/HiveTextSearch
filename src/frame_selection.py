import mlflow
import numpy as np
import cv2
from pathlib import Path
from datasets import Dataset

DATA_DIR = Path(__file__).parent.parent / "data"
FRAMES_DIR = DATA_DIR / "frames"
CLIPS_DATASET_DIR = DATA_DIR / "clips_dataset"
CLIPS_FRAMES_DATASET_DIR = DATA_DIR / "clips_frames_dataset"
N_FRAMES = 4


def extract_frames(clip_path: str, n_frames: int = N_FRAMES) -> list[np.ndarray]:
    """
    Return n_frames linearly spaced frames from a clip as BGR numpy arrays.
    Uses linspace over [0, total_frames) so frames are evenly spread across
    the clip without repeating the last frame.
    For a 2 s clip at 60 fps this gives frames at t = 0, 0.5, 1.0, 1.5 s.
    """
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open clip: {clip_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total, n_frames, endpoint=False).astype(int)

    frames: list[np.ndarray] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            raise RuntimeError(f"Could not read frame {idx} from {clip_path}")
        frames.append(frame)

    cap.release()
    return frames


def save_frames(frames: list[np.ndarray], clip_id: str, output_dir: Path) -> list[str]:
    """
    Save each frame as <clip_id><NN>.jpg (NN = 01, 02, …) under output_dir.
    Returns a list of absolute path strings in frame order.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for i, frame in enumerate(frames):
        name = f"{clip_id}{i + 1:02d}"
        path = output_dir / f"{name}.jpg"
        cv2.imwrite(str(path), frame)
        paths.append(str(path))
    return paths


def add_frames_column(
    dataset: Dataset,
    frames_dir: str | Path = FRAMES_DIR,
    n_frames: int = N_FRAMES,
) -> Dataset:
    """
    For every clip in the dataset, extract n_frames and save as JPGs.
    Adds two new columns:
      frame_paths  — list of n_frames absolute path strings
      frame_names  — list of n_frames filename stems (no extension)
    Logs cumulative frame count to the active MLflow run.
    Returns the updated Dataset.
    Raises ValueError if frame columns already exist.
    """
    if "frame_paths" in dataset.column_names:
        raise ValueError(
            "'frame_paths' column already exists in the dataset. "
            "Drop it first if you want to re-run frame extraction."
        )

    frames_dir = Path(frames_dir)
    n = len(dataset)
    all_frame_paths: list[list[str]] = []
    all_frame_names: list[list[str]] = []

    for i, row in enumerate(dataset):
        clip_id = row["clip_id"]
        frames = extract_frames(row["clip_path"], n_frames)
        paths = save_frames(frames, clip_id, frames_dir)
        all_frame_paths.append(paths)
        all_frame_names.append([Path(p).stem for p in paths])
        mlflow.log_metric("frames_saved_cumulative", (i + 1) * n_frames, step=i)

    dataset = dataset.add_column("frame_paths", all_frame_paths)
    dataset = dataset.add_column("frame_names", all_frame_names)
    mlflow.log_metric("total_frames_saved", n * n_frames)
    return dataset
