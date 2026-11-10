import math
import os
import mlflow
import tempfile
import cv2
from pathlib import Path
from datasets import Dataset

DATA_DIR = Path(__file__).parent.parent / "data"
CLIPS_DIR = DATA_DIR / "clips"
CLIPS_DATASET_DIR = DATA_DIR / "clips_dataset"
CLIP_DURATION_S = 2.0


def video_bytes_to_temp_file(raw_bytes: bytes, suffix: str = ".mp4") -> str:
    """Write video bytes to a temporary file and return its path. Caller must delete it."""
    tf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tf.write(raw_bytes)
    except:
        tf.close()
        os.unlink(tf.name)
        raise
    tf.close()
    return tf.name


def cut_video_into_clips(
    video_path: str,
    video_name: str,
    output_dir: Path,
    fps: float,
    clip_duration: float = CLIP_DURATION_S,
) -> list[dict]:
    """
    Cut one video into clips of clip_duration seconds. Leftover frames at the
    end become a shorter final clip; duration_s and frame_count hold each
    clip's actual length.
    Each clip is saved as <video_name><NN>.mp4 (NN = 01, 02, …).
    Logs fps and clip count per video to the active MLflow run.
    Returns a list of metadata dicts, one per clip.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_clip = int(fps * clip_duration)
    n_clips = math.ceil(total_frames / frames_per_clip)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    clip_infos: list[dict] = []

    for idx in range(n_clips):
        clip_name = f"{video_name}{idx + 1:02d}"
        clip_path = output_dir / f"{clip_name}.mp4"
        writer = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open writer for: {clip_path}")

        frames_written = 0
        for _ in range(frames_per_clip):
            ret, frame = cap.read()
            if not ret:
                break
            writer.write(frame)
            frames_written += 1
        writer.release()

        if frames_written == 0:
            clip_path.unlink(missing_ok=True)
            break

        clip_infos.append({
            "clip_id": clip_name,
            "source_video_id": video_name,
            "clip_index": idx + 1,
            "start_time_s": float(idx * clip_duration),
            "duration_s": frames_written / fps,
            "frame_count": frames_written,
            "width": width,
            "height": height,
            "fps": fps,
            "clip_path": str(clip_path),
        })

        if frames_written < frames_per_clip:
            break

    cap.release()
    mlflow.log_metric(f"{video_name}_clips", len(clip_infos))
    return clip_infos


def save_clips_dataset(dataset: Dataset, output_dir: str | Path = CLIPS_DATASET_DIR) -> None:
    """Save the clips dataset to disk in HuggingFace format."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset.save_to_disk(str(output_dir))
    mlflow.log_metric("total_clips_saved", len(dataset))
