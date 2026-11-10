import shutil
from pathlib import Path
from typing import Callable

import yaml
from ultralytics import YOLO
from ultralytics.data.utils import IMG_FORMATS, img2label_paths

ROOT = Path(__file__).resolve().parent.parent.parent
BEE_DATASET_DIR = ROOT / "data" / "BeeDataset"
RUNS_DIR = ROOT / "experiments" / "yolo"
DEFAULT_MODEL = "yolo26n.pt"


def split_by_hive(
    source_dir: str | Path,
    output_dir: str | Path,
    hive_of: Callable[[Path], str],
    test_hives: list[str],
    val_hives: list[str],
) -> Path:
    """
    Copy the YOLO-format images under source_dir into output_dir/dev/train, dev/val and test,
    so that every hive lands in exactly one split.
    Hives in test_hives go to test, hives in val_hives to dev/val, and all other hives to dev/train.
    dev/val is the held-out part of the devset used for tuning; test is only for the final evaluation.
    The original files are left untouched. Writes and returns output_dir/data.yaml.
    """
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    for output in ("dev", "test", "data.yaml"):
        if (output_dir / output).exists():
            raise FileExistsError(f"{output_dir / output} already exists; delete dev/, test/ and data.yaml to re-split")
    if set(test_hives) & set(val_hives):
        raise ValueError(f"Hives in both test and val: {sorted(set(test_hives) & set(val_hives))}")

    images = sorted(p for p in source_dir.rglob("*") if p.suffix[1:].lower() in IMG_FORMATS)
    if not images:
        raise FileNotFoundError(f"No images found under {source_dir}")
    hives = {image: hive_of(image) for image in images}
    found = set(hives.values())
    if set(test_hives + val_hives) - found:
        raise ValueError(f"Hives not in the dataset: {sorted(set(test_hives + val_hives) - found)}; found {sorted(found)}")
    if not found - set(test_hives) - set(val_hives):
        raise ValueError("No hives left for dev/train")

    labels = [Path(p) for p in img2label_paths(images)]
    if not any(label.exists() for label in labels):
        raise FileNotFoundError("No YOLO label files found; expected labels/ next to images/ with matching names")

    counts: dict[str, int] = {}
    for image, label in zip(images, labels):
        hive = hives[image]
        split = "test" if hive in test_hives else "dev/val" if hive in val_hives else "dev/train"
        # A missing label file means a background frame with no bees, which YOLO also reads that way.
        for source, kind in ((image, "images"), (label, "labels")):
            if not source.exists():
                continue
            target = output_dir / split / kind / source.name
            if target.exists():
                raise FileExistsError(f"Two {kind} named {source.name} would land in {split}; file names must be unique")
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
        counts[split] = counts.get(split, 0) + 1

    data_yaml = output_dir / "data.yaml"
    # No "path" key: Ultralytics then resolves the splits relative to this file, so it works on any machine.
    data_yaml.write_text(yaml.safe_dump({
        "train": "dev/train/images",
        "val": "dev/val/images",
        "test": "test/images",
        "names": {0: "bee"},
    }))
    print(f"Split by hive: {counts}, labels missing for {sum(not label.exists() for label in labels)} images")
    return data_yaml


def train(data_yaml: str | Path, model_name: str = DEFAULT_MODEL, **train_args) -> YOLO:
    """Train a YOLO model on data_yaml's train split; validation metrics on its val split end up in model.metrics."""
    model = YOLO(model_name)
    model.train(data=str(data_yaml), project=str(RUNS_DIR), **train_args)
    return model


def evaluate(model: YOLO, data_yaml: str | Path, split: str = "test"):
    """Evaluate a trained model on one split of data_yaml; returns Ultralytics detection metrics."""
    return model.val(data=str(data_yaml), split=split, project=str(RUNS_DIR))
