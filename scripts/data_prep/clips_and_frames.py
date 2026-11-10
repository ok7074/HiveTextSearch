"""Cut each of the 96 source videos into two-second clips and save a clips_dataset.

Leftover frames at the end of a video become a shorter final clip.

Output:
  data/clips/<clip_id>.mp4          — one MP4 per clip
  data/clips_dataset/               — HuggingFace Dataset with clip metadata
  data/clips.csv                    — same metadata as a flat CSV for quick inspection

Clip naming: <sequence_id><NN>  e.g.  "hive_a_00101" for clip 01 of video "hive_a_001"

Each run is tracked in MLflow under the experiment "HiveTextSearch".
View results with:  mlflow ui  (then open http://127.0.0.1:5000)

Usage:
    python scripts/data_prep/clips_and_frames.py
"""
import os
import sys
import mlflow
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import Dataset, load_from_disk, Video as HFVideo
from src import track_run
from src.clips_and_frames import (
    CLIPS_DATASET_DIR,
    CLIPS_DIR,
    DATA_DIR,
    video_bytes_to_temp_file,
    cut_video_into_clips,
    save_clips_dataset,
)
from src.video_format import get_video_data

DATASET_DIR = DATA_DIR / "thehiveisreal"


@track_run(run_name="cut_clips")
def main() -> None:
    mlflow.log_param("dataset_dir", str(DATASET_DIR))
    mlflow.log_param("clips_dir", str(CLIPS_DIR))
    mlflow.log_param("clip_duration_s", 2.0)

    ds = load_from_disk(str(DATASET_DIR))
    train = ds["train"].cast_column("video", HFVideo(decode=False))
    mlflow.log_metric("source_videos", len(train))

    all_clip_infos: list[dict] = []

    for i, row in enumerate(train):
        video_name = str(row["sequence_id"])
        fps = float(row["fps"])
        mlflow.log_metric("fps", fps, step=i)

        raw_bytes, _ = get_video_data(row)
        tmp_path = video_bytes_to_temp_file(raw_bytes)
        try:
            clips = cut_video_into_clips(tmp_path, video_name, CLIPS_DIR, fps)
            all_clip_infos.extend(clips)
            mlflow.log_metric("clips_cumulative", len(all_clip_infos), step=i)
        finally:
            os.unlink(tmp_path)

    dataset = Dataset.from_list(all_clip_infos)
    save_clips_dataset(dataset, CLIPS_DATASET_DIR)

    csv_path = DATA_DIR / "clips.csv"
    dataset.to_csv(str(csv_path))
    mlflow.log_artifact(str(csv_path))


if __name__ == "__main__":
    main()
