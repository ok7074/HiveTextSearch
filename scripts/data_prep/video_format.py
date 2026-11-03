"""Convert all Arrow-stored videos in the dataset to watchable MP4 files.

Output: data/videos/<original_name>.mp4  (one file per source video)

Each run is tracked in MLflow under the experiment "HiveTextSearch".
View results with:  mlflow ui  (then open http://127.0.0.1:5000)

Usage:
    python scripts/data_prep/video_format.py
"""
import sys
import mlflow
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import load_from_disk, Video as HFVideo
from src import track_run
from src.video_format import convert_all_videos

DATA_DIR = Path(__file__).parent.parent.parent / "data"
DATASET_DIR = DATA_DIR / "thehiveisreal"
VIDEOS_DIR = DATA_DIR / "videos"


@track_run(run_name="convert_videos")
def main() -> None:
    mlflow.log_param("dataset_dir", str(DATASET_DIR))
    mlflow.log_param("output_dir", str(VIDEOS_DIR))

    ds = load_from_disk(str(DATASET_DIR))
    train = ds["train"].cast_column("video", HFVideo(decode=False))
    mlflow.log_metric("source_videos", len(train))

    paths = convert_all_videos(train, VIDEOS_DIR)
    mlflow.log_metric("videos_written", len(paths))


if __name__ == "__main__":
    main()
