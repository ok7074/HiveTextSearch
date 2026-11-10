import pytest
import yaml
from PIL import Image

from src.own_frames import add_own_frames, clip_of, split_own_frames, video_of


def _frames(folder, names):
    folder.mkdir(parents=True, exist_ok=True)
    for name in names:
        Image.new("RGB", (8, 8)).save(folder / f"{name}.jpg")


def test_split_puts_one_frame_per_video_in_test_and_tops_up_train(tmp_path):
    own, frames = tmp_path / "own", tmp_path / "frames"
    # 4 videos (v1..v4), 2 own frames each, all from clip 01.
    _frames(own, [f"recA_v{v}010{f}" for v in range(1, 5) for f in (1, 2)])
    # data/frames holds every clip; clips 01 are the own ones, clips 02..06 are unused.
    _frames(frames, [f"recA_v{v}0{c}0{f}" for v in range(1, 5) for c in range(1, 7) for f in range(1, 5)])

    # target train = ceil(0.5 * 10 / 0.5) = 10 own frames; 8 - 2 test = 6 from the folder, so 4 extra.
    counts = split_own_frames(own, frames, n_public_train=10, n_test=2, own_share=0.5)
    assert counts == {"test": 2, "train_from_folder": 6, "train_extra": 4}

    test, train = sorted((own / "test").glob("*.jpg")), sorted((own / "train").glob("*.jpg"))
    assert len({video_of(f) for f in test}) == 2
    assert len(train) == 10 and not list(own.glob("*.jpg")), "every own frame is moved into a split"
    extra = [f for f in train if not clip_of(f).endswith("01")]
    assert len(extra) == 4 and len({clip_of(f) for f in extra}) == 4, "one extra frame per unused clip"
    with pytest.raises(FileExistsError):
        split_own_frames(own, frames, n_public_train=10, n_test=2, own_share=0.5)


def _own_split(own, split, labels):
    _frames(own / split, labels)
    (own / split / "labels").mkdir()
    for name, text in labels.items():
        if text is not None:
            (own / split / "labels" / f"{name}.txt").write_text(text)


def _bee_dataset(root):
    for split in ("dev/train", "test"):
        for kind in ("images", "labels"):
            (root / split / kind).mkdir(parents=True)


def test_add_copies_images_and_labels_and_writes_own_test_yaml(tmp_path):
    own, bee = tmp_path / "own", tmp_path / "bee"
    _own_split(own, "train", {"a01": "0 0.5 0.5 0.1 0.1\n", "b01": None})  # b01: no bees
    _own_split(own, "test", {"c01": "0 0.2 0.3 0.1 0.1\n"})
    _bee_dataset(bee)

    assert add_own_frames(own, bee) == {"train": 2, "test": 1}
    assert {p.name for p in (bee / "dev/train/images").iterdir()} == {"a01.jpg", "b01.jpg"}
    assert {p.name for p in (bee / "dev/train/labels").iterdir()} == {"a01.txt"}
    assert (bee / "own_test.txt").read_text() == "./test/images/c01.jpg\n"
    assert yaml.safe_load((bee / "data_own_test.yaml").read_text())["test"] == "own_test.txt"
    with pytest.raises(FileExistsError):
        add_own_frames(own, bee)


@pytest.mark.parametrize("bad", ["1 0.5 0.5 0.1 0.1", "0 1.5 0.5 0.1 0.1", "0 0.5 0.5 0.1"])
def test_add_rejects_bad_labels_before_copying_anything(tmp_path, bad):
    own, bee = tmp_path / "own", tmp_path / "bee"
    _own_split(own, "train", {"a01": "0 0.5 0.5 0.1 0.1\n"})
    _own_split(own, "test", {"c01": bad + "\n"})
    _bee_dataset(bee)
    with pytest.raises(ValueError):
        add_own_frames(own, bee)
    assert not list((bee / "dev/train/images").iterdir())
