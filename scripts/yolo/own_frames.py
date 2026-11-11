"""Temporary: turn the own hive frames in data/yolohivedata into labelled YOLO data for the final training.

Steps, in order (the CVAT check happens by hand between prelabel and add):
  split     15 frames from 15 different videos -> yolohivedata/test, the rest -> yolohivedata/train,
            plus extra frames from data/frames (unused clips, one per clip) so own frames are 8% of train
  prelabel  run the s960 bee detector on both splits; writes <split>/labels/*.txt and cvat_<split>.zip
            (Ultralytics YOLO Detection 1.0, images included) to load into CVAT
  add       with the corrected CVAT export (YOLO 1.1) in data/yolohivedataf/<split>/obj_train_data: copy images
            + labels into BeeDataset (whole videos of train, >= N_VAL frames, to dev/val; the rest to dev/train;
            test to test), and write BeeDataset/data.yaml (classes bee, bee_flying) and data_own_test.yaml
  zip       pack BeeDataset (splits only) and the training code into data/kaggle_final_s960.zip

Usage:
    python scripts/yolo/own_frames.py split|prelabel|add|zip
"""
import argparse
import sys
import zipfile
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.frame_selection import FRAMES_DIR
from src.models.yolo_model import BEE_DATASET_DIR, ROOT
from src.own_frames import add_own_frames, split_own_frames

OWN_DIR = ROOT / "data" / "yolohivedata"
OWN_LABELS_DIR = ROOT / "data" / "yolohivedataf"
# Final model of the s960 Optuna run (Kaggle), the best detector so far.
PRELABEL_WEIGHTS = ROOT / "experiments" / "yolo_runsS960" / "yolo_runs" / "train-11" / "weights" / "best.pt"
KAGGLE_ZIP = ROOT / "data" / "kaggle_final_s960.zip"
N_TEST = 15
OWN_SHARE = 0.08
# ~15% of the 391 own train frames, so own frames weigh about as much in val as in train.
N_VAL = 60


def split() -> None:
    n_public = len(list((BEE_DATASET_DIR / "dev" / "train" / "images").glob("*.jpg")))
    print(split_own_frames(OWN_DIR, FRAMES_DIR, n_public, N_TEST, OWN_SHARE))


def prelabel() -> None:
    from ultralytics import YOLO

    model = YOLO(PRELABEL_WEIGHTS)
    for name in ("train", "test"):
        split_dir = OWN_DIR / name
        labels_dir = split_dir / "labels"
        # Never overwrite: these may already be the hand-corrected labels.
        labels_dir.mkdir(exist_ok=False)
        images = sorted(split_dir.glob("*.jpg"))
        n_boxes = 0
        for result in model.predict([str(i) for i in images], stream=True, verbose=False):
            boxes = result.boxes.xywhn.tolist()
            n_boxes += len(boxes)
            lines = "".join(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for x, y, w, h in boxes)
            (labels_dir / f"{Path(result.path).stem}.txt").write_text(lines)

        with zipfile.ZipFile(OWN_DIR / f"cvat_{name}.zip", "w") as archive:
            archive.writestr("data.yaml", yaml.safe_dump({"path": "./", name: f"images/{name}", "names": {0: "bee"}}))
            for image in images:
                archive.write(image, f"images/{name}/{image.name}", compress_type=zipfile.ZIP_STORED)
                archive.write(labels_dir / f"{image.stem}.txt", f"labels/{name}/{image.stem}.txt")
        print(f"{name}: {len(images)} images, {n_boxes} boxes -> {OWN_DIR / f'cvat_{name}.zip'}")


def add() -> None:
    print(add_own_frames(OWN_DIR, OWN_LABELS_DIR, BEE_DATASET_DIR, N_VAL))


def make_zip() -> None:
    """The splits and yaml files of BeeDataset (not the 6.7 GB download) plus the code train_yolo_final.py needs."""
    if not (BEE_DATASET_DIR / "data_own_test.yaml").exists():
        raise FileNotFoundError("Run the add step first")
    files = [
        *(p for split in ("dev", "test") for p in (BEE_DATASET_DIR / split).rglob("*") if p.suffix in (".jpg", ".txt")),
        *(BEE_DATASET_DIR / name for name in ("data.yaml", "data_own_test.yaml", "own_test.txt")),
        *(ROOT / "src").rglob("*.py"),
        *(ROOT / "scripts" / "yolo").glob("*.py"),
    ]
    with zipfile.ZipFile(KAGGLE_ZIP, "w") as archive:
        for path in files:
            compress = zipfile.ZIP_STORED if path.suffix == ".jpg" else zipfile.ZIP_DEFLATED
            archive.write(path, f"HiveTextSearch/{path.relative_to(ROOT).as_posix()}", compress_type=compress)
    print(f"{len(files)} files -> {KAGGLE_ZIP} ({KAGGLE_ZIP.stat().st_size / 1e9:.1f} GB)")


if __name__ == "__main__":
    steps = {"split": split, "prelabel": prelabel, "add": add, "zip": make_zip}
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("step", choices=steps)
    steps[parser.parse_args().step]()
