"""Split the bee detection frames into dev/train, dev/val and test by hive. Run once, before train_yolo.py.

Each folder detection/_bee_<hive>/ (images/ + labels/) is one recording at one hive. Only the detection
data is used; pose/ and segmentation/ are left out.

Output, next to the download in data/BeeDataset/:
  dev/train/{images,labels}   hives used to train
  dev/val/{images,labels}     held-out dev hive, scores the Optuna trials
  test/{images,labels}        held-out hives, used once for the final evaluation
  data.yaml                   the splits for Ultralytics

Usage:
    python scripts/yolo/split_bee_dataset.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.yolo_model import BEE_DATASET_DIR, split_by_hive

DETECTION_DIR = (
    BEE_DATASET_DIR / "Labeled dataset for bee detection and direction estimation on beehive landing boards" / "detection"
)
# One hive from each recording date in test, both close to the average bee density.
TEST_HIVES = ["20230609b", "20230711b"]
VAL_HIVES = ["20230609c"]


def hive_of(image: Path) -> str:
    """detection/_bee_20230609a/images/x.jpg -> 20230609a"""
    return image.parent.parent.name.removeprefix("_bee_")


if __name__ == "__main__":
    split_by_hive(DETECTION_DIR, BEE_DATASET_DIR, hive_of, TEST_HIVES, VAL_HIVES)
