"""Extract 4 linearly spaced frames per clip and save as a new clips_frames_dataset.

For a 2 s clip at 60 fps the four frames fall at t = 0, 0.5, 1.0, 1.5 s.
Frame naming: <clip_id><NN>.jpg  e.g. hive_a_0010101.jpg … hive_a_0010104.jpg

Output:
  data/frames/<clip_id><NN>.jpg      — one JPG per frame (4 per clip)
  data/clips_frames_dataset/         — new dataset: all clips_dataset columns plus
                                       frame_paths and frame_names

data/clips_dataset/ is left untouched.

Each run is tracked in MLflow under the experiment "HiveTextSearch".
View results with:  mlflow ui  (then open http://127.0.0.1:5000)

Usage:
    python scripts/data_prep/frame_selection.py
"""
import sys
import mlflow
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import load_from_disk
from src import track_run
from src.frame_selection import (
    add_frames_column,
    CLIPS_DATASET_DIR,
    CLIPS_FRAMES_DATASET_DIR,
    FRAMES_DIR,
)


@track_run(run_name="extract_frames")
def main() -> None:
    mlflow.log_param("clips_dataset_dir", str(CLIPS_DATASET_DIR))
    mlflow.log_param("clips_frames_dataset_dir", str(CLIPS_FRAMES_DATASET_DIR))
    mlflow.log_param("frames_dir", str(FRAMES_DIR))
    mlflow.log_param("n_frames_per_clip", 4)

    dataset = load_from_disk(str(CLIPS_DATASET_DIR))
    mlflow.log_metric("clips_in_dataset", len(dataset))

    dataset = add_frames_column(dataset, FRAMES_DIR)

    dataset.save_to_disk(str(CLIPS_FRAMES_DATASET_DIR))
    mlflow.log_metric("clips_with_frames", len(dataset))


if __name__ == "__main__":
    main()
