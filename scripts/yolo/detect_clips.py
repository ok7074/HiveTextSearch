"""Run the trained YOLO bee detector on every clip's 4 frames and save the counts per clip.

Output: data/detections/, a dataset with one row per clip:
  clip_id, frame_counts (bees per frame), left, center, right (bees per vertical third, summed over the 4 frames),
  count (mean bees per frame), flying (mean bee_flying boxes per frame), label (left/center/right = third with the most bees, overall on a tie, empty with none)
scripts/retrieval/retrieval.py picks this up as the "detector" method. Runs on CPU.

Usage:
    python scripts/yolo/detect_clips.py
    python scripts/yolo/detect_clips.py --weights path/to/best.pt
"""
import argparse
import sys
from collections import Counter
from pathlib import Path

import mlflow
from datasets import Dataset, load_from_disk
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import track_run
from src.detection import DEFAULT_WEIGHTS, DETECTIONS_DIR, detect_clips
from src.frame_selection import CLIPS_FRAMES_DATASET_DIR, FRAMES_DIR


@track_run(run_name="detect_clips")
def main(weights: str) -> None:
    dataset = load_from_disk(str(CLIPS_FRAMES_DATASET_DIR))
    rows = detect_clips(YOLO(weights), dataset, FRAMES_DIR)
    Dataset.from_list(rows).save_to_disk(str(DETECTIONS_DIR))
    labels = Counter(row["label"] for row in rows)
    mlflow.log_metrics({f"clips_{label}": n for label, n in labels.items()})
    print(f"Saved {len(rows)} clips to {DETECTIONS_DIR}; labels: {dict(labels)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="trained YOLO weights (default: the final s960 run)")
    main(weights=parser.parse_args().weights)
