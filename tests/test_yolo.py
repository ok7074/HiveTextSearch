import pytest
import yaml
from pathlib import Path
from unittest.mock import MagicMock
from PIL import Image

from src import optuna_tuning
from src.models.yolo_model import split_by_hive


def _make_dataset(root: Path, frames: dict[str, list[str]], unlabelled: tuple[str, ...] = ()) -> None:
    """Write images/<hive>_<frame>.jpg and matching labels/*.txt; hive is the part before the first underscore."""
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir(parents=True)
    for hive, names in frames.items():
        for name in names:
            stem = f"{hive}_{name}"
            Image.new("RGB", (8, 8)).save(root / "images" / f"{stem}.jpg")
            if stem not in unlabelled:
                (root / "labels" / f"{stem}.txt").write_text("0 0.5 0.5 0.1 0.1\n")


def _hive(path: Path) -> str:
    return path.stem.split("_")[0]


def test_split_puts_each_hive_in_exactly_one_split(tmp_path):
    _make_dataset(tmp_path, {"h1": ["a", "b"], "h2": ["a"], "h3": ["a", "b"], "h4": ["a"]})
    data_yaml = split_by_hive(tmp_path, _hive, test_hives=["h4"], val_hives=["h3"])

    hives_in = {
        split: {_hive(p) for p in (tmp_path / split / "images").glob("*.jpg")}
        for split in ("dev/train", "dev/val", "test")
    }
    assert hives_in == {"dev/train": {"h1", "h2"}, "dev/val": {"h3"}, "test": {"h4"}}
    for split in ("dev/train", "dev/val", "test"):
        images = {p.stem for p in (tmp_path / split / "images").glob("*.jpg")}
        assert images == {p.stem for p in (tmp_path / split / "labels").glob("*.txt")}
    assert yaml.safe_load(data_yaml.read_text()) == {
        "train": "dev/train/images", "val": "dev/val/images", "test": "test/images", "names": {0: "bee"},
    }
    assert len(list((tmp_path / "images").glob("*.jpg"))) == 6, "originals must stay in place"


def test_split_keeps_unlabelled_frames_as_background(tmp_path):
    _make_dataset(tmp_path, {"h1": ["a", "b"], "h2": ["a"], "h3": ["a"]}, unlabelled=("h1_b",))
    split_by_hive(tmp_path, _hive, test_hives=["h3"], val_hives=["h2"])
    assert (tmp_path / "dev/train/images/h1_b.jpg").exists()
    assert not (tmp_path / "dev/train/labels/h1_b.txt").exists()


@pytest.mark.parametrize("test_hives, val_hives, error", [
    (["h1"], ["h1"], ValueError),            # a hive in both test and val
    (["h9"], ["h2"], ValueError),            # a hive that is not in the dataset
    (["h1"], ["h2", "h3"], ValueError),      # no hive left for dev/train
])
def test_split_rejects_bad_hive_lists(tmp_path, test_hives, val_hives, error):
    _make_dataset(tmp_path, {"h1": ["a"], "h2": ["a"], "h3": ["a"]})
    with pytest.raises(error):
        split_by_hive(tmp_path, _hive, test_hives=test_hives, val_hives=val_hives)
    assert not (tmp_path / "dev").exists() and not (tmp_path / "test").exists()


def test_split_refuses_to_overwrite_an_existing_split(tmp_path):
    _make_dataset(tmp_path, {"h1": ["a"], "h2": ["a"], "h3": ["a"]})
    split_by_hive(tmp_path, _hive, test_hives=["h3"], val_hives=["h2"])
    with pytest.raises(FileExistsError):
        split_by_hive(tmp_path, _hive, test_hives=["h3"], val_hives=["h2"])


def test_split_rejects_duplicate_file_names(tmp_path):
    """Per-hive folders can reuse file names; two of them in one split must not overwrite each other."""
    for hive in ("h1", "h2", "h3"):
        (tmp_path / hive / "images").mkdir(parents=True)
        (tmp_path / hive / "labels").mkdir(parents=True)
        Image.new("RGB", (8, 8)).save(tmp_path / hive / "images" / "frame.jpg")
        (tmp_path / hive / "labels" / "frame.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    with pytest.raises(FileExistsError):
        split_by_hive(tmp_path, lambda p: p.parent.parent.name, test_hives=["h3"], val_hives=[])


def test_tune_returns_the_best_trial(monkeypatch):
    monkeypatch.setattr(optuna_tuning, "mlflow", MagicMock())
    study = optuna_tuning.tune(lambda params: -abs(params["momentum"] - 0.9), n_trials=10)
    assert len(study.trials) == 10
    assert study.best_value == max(t.value for t in study.trials)
    assert set(study.best_params) == {"lr0", "lrf", "momentum", "weight_decay"}
