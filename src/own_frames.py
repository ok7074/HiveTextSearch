"""Own hive frames (data/yolohivedata, corrected labels in data/yolohivedataf) for the final YOLO training:
split, check the corrected labels, add to BeeDataset."""
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

import yaml


def clip_of(frame: Path) -> str:
    """rec2123_seq0150403.jpg -> rec2123_seq01504 (frames are <clip_id><NN>.jpg)."""
    return frame.stem[:-2]


def video_of(frame: Path) -> str:
    """rec2123_seq0150403.jpg -> rec2123_seq015 (clip ids are <video_id><NN>)."""
    return frame.stem[:-4]


def split_own_frames(
    own_dir: Path, frames_dir: Path, n_public_train: int, n_test: int, own_share: float, seed: int = 0,
) -> dict[str, int]:
    """
    Move n_test frames of own_dir, each from a different video, into own_dir/test and the rest into own_dir/train.
    Then copy extra frames from frames_dir into own_dir/train, one per clip and only from clips not already in
    own_dir, until own frames make up own_share of the train split (public + own). Returns the counts.
    """
    if (own_dir / "train").exists() or (own_dir / "test").exists():
        raise FileExistsError(f"{own_dir} is already split; delete train/ and test/ to redo it")
    frames = sorted(own_dir.glob("*.jpg"))
    by_video = defaultdict(list)
    for frame in frames:
        by_video[video_of(frame)].append(frame)
    if len(by_video) < n_test:
        raise ValueError(f"Only {len(by_video)} videos in {own_dir}, need {n_test} for test")

    rng = random.Random(seed)
    test = {rng.choice(by_video[video]) for video in rng.sample(sorted(by_video), n_test)}
    n_train_target = math.ceil(own_share * n_public_train / (1 - own_share))
    n_extra = max(n_train_target - (len(frames) - n_test), 0)

    used_clips = {clip_of(frame) for frame in frames}
    candidates = defaultdict(list)
    for frame in sorted(frames_dir.glob("*.jpg")):
        if clip_of(frame) not in used_clips:
            candidates[clip_of(frame)].append(frame)
    if len(candidates) < n_extra:
        raise ValueError(f"Need {n_extra} extra clips but only {len(candidates)} are unused")
    extra = [rng.choice(candidates[clip]) for clip in rng.sample(sorted(candidates), n_extra)]

    for split in ("train", "test"):
        (own_dir / split).mkdir()
    for frame in frames:
        shutil.move(frame, own_dir / ("test" if frame in test else "train") / frame.name)
    for frame in extra:
        shutil.copy2(frame, own_dir / "train" / frame.name)
    return {"test": n_test, "train_from_folder": len(frames) - n_test, "train_extra": n_extra}


# Class ids of the CVAT export (obj.names). The public BeeDataset only uses 0.
NAMES = {0: "bee", 1: "bee_flying"}


def check_labels(images_dir: Path, labels_dir: Path) -> list[Path]:
    """
    Check labels_dir/*.txt against images_dir/*.jpg and return the images.
    Every label must belong to an image and hold 'class x y w h' lines with a class in NAMES and values in [0, 1].
    An image without a label file is a frame with no bees, which YOLO reads the same way.
    """
    images = sorted(images_dir.glob("*.jpg"))
    labels = sorted(labels_dir.glob("*.txt"))
    if not labels:
        raise FileNotFoundError(f"No labels in {labels_dir}; put the corrected CVAT export there")
    if stray := {label.stem for label in labels} - {image.stem for image in images}:
        raise ValueError(f"Labels in {labels_dir} without an image in {images_dir}: {sorted(stray)[:5]}")
    classes = {str(c) for c in NAMES}
    for label in labels:
        for line in label.read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5 or fields[0] not in classes or not all(0 <= float(v) <= 1 for v in fields[1:]):
                raise ValueError(f"Bad line in {label}: {line!r}")
    return images


def add_own_frames(
    images_dir: Path, labels_dir: Path, bee_dataset_dir: Path, n_val: int, seed: int = 0,
) -> dict[str, int]:
    """
    Copy the own frames (images_dir/<split>/*.jpg, labels from the CVAT export labels_dir/<split>/obj_train_data)
    into BeeDataset: whole videos of the train frames, at least n_val frames, go to dev/val, the rest of train to
    dev/train and test to test. Also writes data.yaml with NAMES, own_test.txt and data_own_test.yaml, which
    evaluates on the own test frames alone. Everything is checked before the first copy, so a bad label leaves
    BeeDataset untouched; images_dir and labels_dir are only read.
    """
    label_dirs = {split: labels_dir / split / "obj_train_data" for split in ("train", "test")}
    checked = {split: check_labels(images_dir / split, label_dirs[split]) for split in label_dirs}

    by_video = defaultdict(list)
    for image in checked["train"]:
        by_video[video_of(image)].append(image)
    val = []
    # Whole videos, so no val frame shares a video with a train frame.
    for video in random.Random(seed).sample(sorted(by_video), len(by_video)):
        if len(val) >= n_val:
            break
        val += by_video[video]
    images = {"train": [i for i in checked["train"] if i not in val], "val": val, "test": checked["test"]}
    sources = {"train": label_dirs["train"], "val": label_dirs["train"], "test": label_dirs["test"]}
    targets = {
        "train": bee_dataset_dir / "dev" / "train", "val": bee_dataset_dir / "dev" / "val", "test": bee_dataset_dir / "test",
    }
    for split, target in targets.items():
        for image in images[split]:
            for kind in ("images", "labels"):
                if (target / kind / (image.stem + (".jpg" if kind == "images" else ".txt"))).exists():
                    raise FileExistsError(f"{image.stem} is already in {target / kind}; own frames were added before")

    for split, target in targets.items():
        for image in images[split]:
            shutil.copy2(image, target / "images" / image.name)
            label = sources[split] / f"{image.stem}.txt"
            if label.exists():
                shutil.copy2(label, target / "labels" / label.name)

    splits = {"train": "dev/train/images", "val": "dev/val/images"}
    (bee_dataset_dir / "data.yaml").write_text(yaml.safe_dump({**splits, "test": "test/images", "names": NAMES}))
    # "./" lines resolve relative to the txt file, so this works wherever the dataset is unpacked.
    (bee_dataset_dir / "own_test.txt").write_text("".join(f"./test/images/{i.name}\n" for i in images["test"]))
    (bee_dataset_dir / "data_own_test.yaml").write_text(yaml.safe_dump({**splits, "test": "own_test.txt", "names": NAMES}))
    return {split: len(images[split]) for split in targets}
