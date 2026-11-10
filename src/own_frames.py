"""Own hive frames (data/yolohivedata) for the final YOLO training: split, check the corrected labels, add to BeeDataset."""
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


def check_labels(split_dir: Path) -> list[Path]:
    """
    Check split_dir/labels/*.txt against split_dir/*.jpg and return the images.
    Every label must belong to an image and hold 'class x y w h' lines with class 0 and values in [0, 1].
    An image without a label file is a frame with no bees, which YOLO reads the same way.
    """
    images = sorted(split_dir.glob("*.jpg"))
    labels = sorted((split_dir / "labels").glob("*.txt"))
    if not labels:
        raise FileNotFoundError(f"No labels in {split_dir / 'labels'}; put the corrected CVAT export there")
    if stray := {label.stem for label in labels} - {image.stem for image in images}:
        raise ValueError(f"Labels without an image in {split_dir}: {sorted(stray)[:5]}")
    for label in labels:
        for line in label.read_text().splitlines():
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 5 or fields[0] != "0" or not all(0 <= float(v) <= 1 for v in fields[1:]):
                raise ValueError(f"Bad line in {label}: {line!r}")
    return images


def add_own_frames(own_dir: Path, bee_dataset_dir: Path) -> dict[str, int]:
    """
    Copy own_dir/train into bee_dataset_dir/dev/train and own_dir/test into bee_dataset_dir/test (images + labels).
    Also writes own_test.txt and data_own_test.yaml, which evaluate on the own test frames alone.
    Everything is checked before the first copy, so a bad label leaves BeeDataset untouched.
    """
    targets = {"train": bee_dataset_dir / "dev" / "train", "test": bee_dataset_dir / "test"}
    images = {split: check_labels(own_dir / split) for split in targets}
    for split, target in targets.items():
        for image in images[split]:
            for kind in ("images", "labels"):
                if (target / kind / (image.stem + (".jpg" if kind == "images" else ".txt"))).exists():
                    raise FileExistsError(f"{image.stem} is already in {target / kind}; own frames were added before")

    for split, target in targets.items():
        for image in images[split]:
            shutil.copy2(image, target / "images" / image.name)
            label = own_dir / split / "labels" / f"{image.stem}.txt"
            if label.exists():
                shutil.copy2(label, target / "labels" / label.name)

    # "./" lines resolve relative to the txt file, so this works wherever the dataset is unpacked.
    (bee_dataset_dir / "own_test.txt").write_text("".join(f"./test/images/{i.name}\n" for i in images["test"]))
    (bee_dataset_dir / "data_own_test.yaml").write_text(yaml.safe_dump({
        "train": "dev/train/images", "val": "dev/val/images", "test": "own_test.txt", "names": {0: "bee"},
    }))
    return {split: len(images[split]) for split in targets}
