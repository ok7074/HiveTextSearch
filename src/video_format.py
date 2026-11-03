import mlflow
from pathlib import Path
from datasets import Dataset

DATA_DIR = Path(__file__).parent.parent / "data"
VIDEOS_DIR = DATA_DIR / "videos"


def get_video_data(row: dict) -> tuple[bytes, str]:
    """Return (raw_bytes, original_path) from a dataset row with decode=False."""
    video = row["video"]
    if isinstance(video, dict):
        raw = video.get("bytes") or b""
        orig_path = video.get("path") or ""
    else:
        raw = bytes(video)
        orig_path = ""
    return raw, orig_path


def make_output_filename(row: dict, orig_path: str, ext: str = ".mp4") -> str:
    """Derive a safe output filename: prefer the original name, fall back to sequence_id."""
    if orig_path:
        stem = Path(orig_path).stem
        suffix = Path(orig_path).suffix or ext
        return f"{stem}{suffix}"
    return f"{row['sequence_id']}{ext}"


def save_video(raw_bytes: bytes, output_path: Path) -> Path:
    """Write raw bytes to output_path, creating parent directories as needed."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(raw_bytes)
    return output_path


def convert_all_videos(
    dataset: Dataset,
    output_dir: str | Path = VIDEOS_DIR,
    ext: str = ".mp4",
) -> list[Path]:
    """
    Save every video in the dataset to output_dir as a playable file.
    Expects the dataset to have its 'video' column set to decode=False.
    Logs per-video size and a completion count to the active MLflow run.
    Returns the list of saved paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    n = len(dataset)
    paths: list[Path] = []

    for i, row in enumerate(dataset):
        raw, orig_path = get_video_data(row)
        filename = make_output_filename(row, orig_path, ext)
        out_path = output_dir / filename
        save_video(raw, out_path)
        paths.append(out_path)
        mlflow.log_metric("video_size_mb", len(raw) / 1_000_000, step=i)
        mlflow.log_metric("videos_saved", i + 1, step=i)

    mlflow.log_metric("total_videos_saved", len(paths))
    return paths
