import numpy as np
import pytest
import torch
from types import SimpleNamespace
from unittest.mock import MagicMock
from datasets import Dataset
from PIL import Image

from src.embeddings import average_frames, embed_frames, save_embeddings
from src.models import clip_image_features, siglip2_image_features


def _write_frames(tmp_path, clips: dict[str, int]) -> Dataset:
    """Write one solid-colour JPG per frame; the red channel encodes clip then frame order."""
    rows = []
    for clip_number, (clip_id, n_frames) in enumerate(clips.items()):
        names = [f"{clip_id}{i + 1:02d}" for i in range(n_frames)]
        for i, name in enumerate(names):
            Image.new("RGB", (8, 8), (clip_number * 40 + i * 5, 0, 0)).save(tmp_path / f"{name}.jpg", quality=100)
        rows.append({"clip_id": clip_id, "frame_names": names})
    return Dataset.from_list(rows)


def _red_channel_features(images) -> torch.Tensor:
    return torch.tensor([[float(image.getpixel((0, 0))[0]), 1.0] for image in images])


def test_embed_frames_keeps_each_frame_with_its_clip_across_batches(tmp_path):
    dataset = _write_frames(tmp_path, {"a": 4, "b": 4, "c": 4})
    result = embed_frames(dataset, tmp_path, _red_channel_features, batch_size=2)

    assert result.shape == (3, 4, 2)
    assert result.dtype == np.float32
    # Lossy JPG can shift a value slightly, so compare with a small tolerance.
    expected = [[clip * 40 + frame * 5 for frame in range(4)] for clip in range(3)]
    assert np.allclose(result[:, :, 0], expected, atol=3)


def test_embed_frames_rejects_clip_with_wrong_frame_count(tmp_path):
    dataset = _write_frames(tmp_path, {"a": 4, "short": 3})
    with pytest.raises(ValueError):
        embed_frames(dataset, tmp_path, _red_channel_features)


def test_average_frames_is_the_normalised_mean():
    frames = np.array([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]], dtype=np.float32)
    assert np.allclose(average_frames(frames), [[2 ** -0.5, 2 ** -0.5]])


def test_save_embeddings_round_trips_without_pickle(tmp_path):
    frames = np.ones((2, 4, 3), dtype=np.float32)
    clips = np.ones((2, 3), dtype=np.float32)
    path = tmp_path / "nested" / "clip.npz"
    save_embeddings(path, ["a", "b"], frames, clips, "some/model")

    with np.load(path, allow_pickle=False) as saved:
        assert saved["clip_ids"].tolist() == ["a", "b"]
        assert saved["frame_embeddings"].shape == (2, 4, 3)
        assert saved["clip_embeddings"].shape == (2, 3)
        assert str(saved["model_id"]) == "some/model"


@pytest.mark.parametrize("image_features", [clip_image_features, siglip2_image_features])
def test_image_features_normalises_pooler_output(image_features):
    """transformers 5 returns an output object from get_image_features, not a tensor."""
    model = MagicMock(device="cpu")
    model.get_image_features.return_value = SimpleNamespace(pooler_output=torch.tensor([[3.0, 4.0]]))
    processor = MagicMock()
    processor.return_value.to.return_value = {}

    assert torch.allclose(image_features(model, processor, ["x"]), torch.tensor([[0.6, 0.8]]))


def test_siglip2_center_crops_images():
    model = MagicMock(device="cpu")
    model.get_image_features.return_value = SimpleNamespace(pooler_output=torch.tensor([[3.0, 4.0]]))
    processor = MagicMock()
    processor.return_value.to.return_value = {}

    siglip2_image_features(model, processor, ["x"])
    assert processor.call_args.kwargs["do_center_crop"] is True
