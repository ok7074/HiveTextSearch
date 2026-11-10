import numpy as np
import pytest
from datasets import Dataset

from src.detection import clip_label, detector_scores, load_detections, query_position, third_counts

DETECTIONS = {
    "left": np.array([4, 0, 1]),
    "center": np.array([0, 2, 1]),
    "right": np.array([0, 0, 1]),
    "count": np.array([1.0, 0.5, 0.75]),
    "label": np.array(["left", "center", "overall"]),
}


def test_third_counts_splits_the_width_in_three():
    assert third_counts([10, 40, 50, 99, 100], width=100).tolist() == [1, 2, 2]
    assert third_counts([], width=100).tolist() == [0, 0, 0]


def test_clip_label_picks_the_busiest_third_or_overall_or_empty():
    assert clip_label([1, 5, 2]) == "center"
    assert clip_label([3, 0, 3]) == "overall"
    assert clip_label([0, 0, 0]) == "empty"


def test_load_detections_reorders_rows_to_match_clip_ids(tmp_path):
    rows = [{"clip_id": c, "left": n, "center": 0, "right": 0, "count": n / 4, "label": "left"} for c, n in (("b", 2), ("a", 1))]
    Dataset.from_list(rows).save_to_disk(str(tmp_path))
    detections = load_detections(np.array(["a", "b"]), tmp_path)
    assert detections["left"].tolist() == [1, 2]
    with pytest.raises(ValueError):
        load_detections(np.array(["a", "c"]), tmp_path)


def test_query_position_maps_synonyms():
    assert query_position("Bees in the middle") == "center"
    assert query_position("crowded on the left side") == "left"
    assert query_position("many bees") is None


def test_detector_filters_by_label_and_ranks_by_that_third():
    scores, descending = detector_scores("bees on the left", "position", DETECTIONS)
    assert descending
    assert scores[0] == 4 and np.isnan(scores[1:]).all()


def test_detector_ranks_count_queries_by_mean_count_and_skips_other_types():
    scores, descending = detector_scores("few bees", "count", DETECTIONS)
    assert not descending and scores.tolist() == [1.0, 0.5, 0.75]
    assert detector_scores("busy entrance", "motion", DETECTIONS) is None


WITH_EMPTY_CLIP = {
    "left": np.array([4, 0, 0]), "center": np.array([0, 2, 0]), "right": np.array([0, 3, 0]),
    "count": np.array([1.0, 1.25, 0.0]), "label": np.array(["left", "right", "empty"]),
}


def test_handful_ranks_fewest_bees_first_but_not_empty_clips():
    """R1 + R3: "handful" is a quiet word, and few-words leave out clips with no bees; "empty" keeps them."""
    scores, descending = detector_scores("Only a handful of bees on the board", "count", WITH_EMPTY_CLIP)
    assert not descending and scores[:2].tolist() == [1.0, 1.25] and np.isnan(scores[2])
    scores, descending = detector_scores("Almost empty landing board", "count", WITH_EMPTY_CLIP)
    assert not descending and scores.tolist() == [1.0, 1.25, 0.0]


def test_empty_side_ranks_clips_with_bees_elsewhere_by_fewest_on_that_side():
    """R2: "left side empty" is not "left-heavy clips"; clip 1 (no bees left, 5 elsewhere) must be allowed."""
    scores, descending = detector_scores("Left side of the board empty", "position", WITH_EMPTY_CLIP)
    assert not descending
    assert scores[1] == 0 and np.isnan(scores[0]) and np.isnan(scores[2])


def test_position_query_without_a_side_is_left_to_clip():
    """R4: no left/middle/right word, so the detector does not rank position queries."""
    assert detector_scores("Bees evenly spread along the board", "position", WITH_EMPTY_CLIP) is None
    assert detector_scores("Bees clustered on the right side of the board", "position", WITH_EMPTY_CLIP) is not None
