import numpy as np
import pytest
import torch
from datasets import Dataset
from PIL import Image

from src.retrieval import frame_difference, load_queries, motion_descending, motion_scores, text_scores, top_k

QUERIES = """\
Train:
Count:
- many bees on the board
Motion:
1. busy entrance

Test:
Count:
few bees
"""


def test_load_queries_reads_one_split_with_types(tmp_path):
    path = tmp_path / "queries.txt"
    path.write_text(QUERIES)
    assert load_queries(path, "train") == [
        {"query_type": "count", "query": "many bees on the board"},
        {"query_type": "motion", "query": "busy entrance"},
    ]
    assert load_queries(path, "test") == [{"query_type": "count", "query": "few bees"}]


def test_load_queries_rejects_query_without_headings(tmp_path):
    path = tmp_path / "queries.txt"
    path.write_text("Train:\nmany bees\n")
    with pytest.raises(ValueError):
        load_queries(path, "train")


def test_text_scores_are_dot_products_with_the_query():
    clips = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    scores = text_scores("q", clips, lambda texts: torch.tensor([[0.6, 0.8]]))
    assert np.allclose(scores, [0.6, 0.8])


def test_top_k_orders_both_ways_and_keeps_ties_in_clip_order():
    ids = np.array(["a", "b", "c", "d"])
    scores = np.array([0.2, 0.9, 0.2, 0.5])
    assert [c for c, _ in top_k(scores, ids, k=3)] == ["b", "d", "a"]
    assert [c for c, _ in top_k(scores, ids, k=3, descending=False)] == ["a", "c", "d"]


def test_top_k_leaves_out_nan_scores():
    ids = np.array(["a", "b", "c"])
    assert [c for c, _ in top_k(np.array([np.nan, 1.0, 2.0]), ids, k=3)] == ["c", "b"]
    assert [c for c, _ in top_k(np.array([np.nan, 1.0, 2.0]), ids, k=3, descending=False)] == ["b", "c"]


def test_motion_descending_flips_for_quiet_words():
    assert motion_descending("busy entrance")
    assert not motion_descending("Calm, quiet board")


def test_motion_scores_rank_a_changing_clip_above_a_still_one(tmp_path):
    for clip, values in {"still": [100, 100, 100, 100], "moving": [0, 200, 0, 200]}.items():
        for i, value in enumerate(values):
            Image.new("L", (8, 8), value).save(tmp_path / f"{clip}{i + 1:02d}.jpg", quality=100)
    dataset = Dataset.from_list([
        {"clip_id": clip, "frame_names": [f"{clip}{i + 1:02d}" for i in range(4)]} for clip in ("still", "moving")
    ])
    scores = motion_scores(dataset, tmp_path)
    assert scores[0] == pytest.approx(0, abs=1)
    assert scores[1] == pytest.approx(200, abs=3)


def test_frame_difference_does_not_wrap_around_uint8():
    frames = [np.full((2, 2), 10, np.uint8), np.full((2, 2), 5, np.uint8)]
    assert frame_difference(frames) == 5
