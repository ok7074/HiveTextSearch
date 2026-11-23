import numpy as np
import pytest
import torch
from datasets import Dataset, load_from_disk
from PIL import Image

from src.embeddings import EMBEDDINGS_DIR, embed_frames
from src.flip_test import is_middle_query, mirror_detections, mirror_query, overlap, rank_correlation
from src.frame_selection import CLIPS_FRAMES_DATASET_DIR, FRAMES_DIR
from src.retrieval import QUERIES_FILE, load_queries

# The spec's table: every side query of queries.txt and its mirror.
SIDE_QUERIES = {
    "Bees clustered on the right side of the board": "Bees clustered on the left side of the board",
    "Left side of the board empty": "Right side of the board empty",
    "Many bees crowded on the left side of the board": "Many bees crowded on the right side of the board",
    "Bees walking on the left side of the board": "Bees walking on the right side of the board",
    "Bees on the right side of a hive with a red stripe": "Bees on the left side of a hive with a red stripe",
    "Bees on the left side near the pole": "Bees on the right side near the pole",
    "More bees on the right than on the left": "More bees on the left than on the right",
    "A few bees on the right side of the board": "A few bees on the left side of the board",
    "Bees flying near the right side of the entrance": "Bees flying near the left side of the entrance",
    "Bees on the left side of the board in sunlight": "Bees on the right side of the board in sunlight",
}
MIDDLE_QUERIES = {
    "Bees bunched in the middle of the board", "A dense group of bees in the middle of the board",
    "Bees resting in the middle of the board", "Bees in the middle of the board next to a black pole",
}


def test_mirror_query_swaps_both_words_at_once_and_keeps_case():
    assert mirror_query("More bees on the right than on the left") == "More bees on the left than on the right"
    assert mirror_query("Left side of the board empty") == "Right side of the board empty"
    assert mirror_query("Bees in the middle of the board") is None
    assert mirror_query("Bees leaving the hive") is None  # "leaving" holds no side word


def test_mirror_query_matches_the_spec_table_and_finds_the_controls():
    queries = [q["query"] for split in ("train", "test") for q in load_queries(QUERIES_FILE, split)]
    mirrored = {q: m for q in queries if (m := mirror_query(q)) is not None}
    assert mirrored == SIDE_QUERIES
    assert {q for q in queries if is_middle_query(q)} == MIDDLE_QUERIES


def test_mirror_detections_swaps_sides_and_mirrors_boxes():
    detections = {
        "left": np.array([5, 0]), "center": np.array([1, 1]), "right": np.array([0, 3]),
        "label": np.array(["left", "center"], dtype=object),
        "box_x": np.array([[0.1, 0.2], [0.5]], dtype=object), "box_y": np.array([[0.4, 0.6], [0.5]], dtype=object),
    }
    mirrored = mirror_detections(detections)
    assert mirrored["left"].tolist() == [0, 3] and mirrored["right"].tolist() == [5, 0]
    assert mirrored["label"].tolist() == ["right", "center"]
    assert np.allclose(mirrored["box_x"][0], [0.9, 0.8]) and np.allclose(mirrored["box_x"][1], [0.5])
    assert mirrored["box_y"] is detections["box_y"]
    assert detections["left"].tolist() == [5, 0]  # the input is left alone


def test_rank_correlation_and_overlap():
    a = np.array([3.0, 2.0, 1.0, np.nan])
    assert rank_correlation(a, a) == pytest.approx(1)
    assert rank_correlation(a, -a, True, False) == pytest.approx(1)  # same order, one ranks fewest first
    assert rank_correlation(np.array([1.0, 2, 3]), np.array([3.0, 2, 1])) == pytest.approx(-1)
    assert overlap([("a", 1), ("b", 1)], [("b", 2), ("c", 2)]) == 1


def test_embed_frames_flip_mirrors_every_frame(tmp_path):
    names = [f"a{i:02d}" for i in range(1, 5)]
    for name in names:
        image = Image.new("RGB", (8, 8), (0, 0, 0))
        image.paste((200, 0, 0), (0, 0, 4, 8))  # left half red
        image.save(tmp_path / f"{name}.jpg", quality=100)
    dataset = Dataset.from_list([{"clip_id": "a", "frame_names": names}])
    left_red = lambda images: torch.tensor([[float(image.getpixel((0, 0))[0])] for image in images])

    assert (embed_frames(dataset, tmp_path, left_red) > 150).all()
    assert (embed_frames(dataset, tmp_path, left_red, flip=True) < 50).all()


@pytest.mark.skipif(not (EMBEDDINGS_DIR / "clip.npz").exists() or not FRAMES_DIR.exists(), reason="needs the real data")
def test_flip_false_reproduces_saved_embeddings_and_flip_true_changes_them():
    from src.models import clip_image_features, load_clip

    try:
        model, processor = load_clip(device_map=None)
    except OSError:
        pytest.skip("CLIP is not in the local Hugging Face cache")
    dataset = load_from_disk(str(CLIPS_FRAMES_DATASET_DIR)).select([0, 1])
    features = lambda images: clip_image_features(model, processor, images)
    with np.load(EMBEDDINGS_DIR / "clip.npz", allow_pickle=False) as saved:
        assert saved["clip_ids"][:2].tolist() == dataset["clip_id"]
        expected = saved["frame_embeddings"][:2]

    assert np.abs(embed_frames(dataset, FRAMES_DIR, features) - expected).max() < 1e-4
    assert np.abs(embed_frames(dataset, FRAMES_DIR, features, flip=True) - expected).max() > 1e-2
