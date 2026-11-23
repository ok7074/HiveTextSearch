import numpy as np
import pytest
from datasets import Dataset

from src.detection import (
    board_lines, clip_label, detector_scores, load_detections, query_band, query_position, query_spread,
    spread_scores, third_counts, two_groups,
)

DETECTIONS = {
    "left": np.array([4, 0, 1]),
    "center": np.array([0, 2, 1]),
    "right": np.array([0, 0, 1]),
    "count": np.array([1.0, 0.5, 0.75]),
    "flying": np.array([0.0, 0.5, 0.25]),
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
    rows = [
        {"clip_id": c, "left": n, "center": 0, "right": 0, "count": n / 4, "flying": 0.0, "label": "left",
         "box_x": [0.1] * n, "box_y": [0.5] * n, "box_cls": [0] * n}
        for c, n in (("vid01", 2), ("vid02", 1))
    ]
    Dataset.from_list(rows).save_to_disk(str(tmp_path / "new"))
    detections = load_detections(np.array(["vid02", "vid01"]), tmp_path / "new")
    assert detections["left"].tolist() == [1, 2]
    assert detections["board_middle"].tolist() == [0.5, 0.5]
    with pytest.raises(ValueError):
        load_detections(np.array(["vid02", "c"]), tmp_path / "new")
    old = [{k: v for k, v in row.items() if not k.startswith("box_")} for row in rows]
    Dataset.from_list(old).save_to_disk(str(tmp_path / "old"))
    with pytest.raises(ValueError, match="rerun"):  # saved before box positions were stored
        load_detections(np.array(["vid02", "vid01"]), tmp_path / "old")


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
    assert detector_scores("busy entrance", "colour", DETECTIONS) is None


def test_motion_queries_rank_by_flying_bees_and_flip_for_quiet_words():
    scores, descending = detector_scores("Many bees flying in front of the hive", "motion", DETECTIONS)
    assert descending and scores.tolist() == [0.0, 0.5, 0.25]
    _, descending = detector_scores("Bees sitting still on the board", "motion", DETECTIONS)
    assert not descending


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


def test_general_queries_are_ranked_by_total_count():
    scores, descending = detector_scores("Bees in bright sunlight", "general", WITH_EMPTY_CLIP)
    assert descending and scores.tolist() == [1.0, 1.25, 0.0]
    scores, descending = detector_scores("Quiet entrance in shade", "general", WITH_EMPTY_CLIP)
    assert not descending


def test_position_query_naming_no_place_or_spread_is_left_to_clip():
    assert detector_scores("Bees near the pole", "position", WITH_EMPTY_CLIP) is None


def test_query_band_and_spread_words():
    assert query_band("A bee on the wooden board above the entrance") == "above"
    assert query_band("Bees at the lower edge of the board") == "below"
    assert query_band("Bees on the left") is None
    assert query_spread("Bees evenly spread along the board") == "even"
    assert query_spread("Two separate groups of bees on the board") == "groups"
    assert query_spread("A dense group of bees in the middle") is None


def test_board_lines_put_the_entrance_at_the_top_edge_of_the_sitting_bees_per_video():
    entrance, middle = board_lines(
        np.array(["vidA01", "vidA02", "vidB01"]),
        [[0.4, 0.9], [0.5], []],  # 0.9 is a flying bee, so it does not count
        [[0, 1], [0], []],
    )
    # sitting heights 0.4 and 0.5: Q1 0.425, median 0.45, entrance 0.425 - 1.5 x 0.025
    assert entrance[0] == entrance[1] == pytest.approx(0.3875) and middle[0] == pytest.approx(0.45)
    assert entrance[2] == pytest.approx(0.3875)  # no sitting bees: median over the other videos


# Clip 0: 4 bees above the entrance on the left; clip 1: 4 under the board middle on the right (2 flying); clip 2: none.
BOXES = {
    "box_x": [[0.1] * 4, [0.9] * 4, []],
    "box_y": [[0.3] * 4, [0.6] * 4, []],
    "box_cls": [[0] * 4, [0, 0, 1, 1], []],
    "entrance": np.array([0.4, 0.4, 0.4]),
    "board_middle": np.array([0.45, 0.45, 0.45]),
}


def test_band_queries_count_bees_above_or_below_the_entrance_line():
    scores, descending = detector_scores("A bee on the wooden board above the entrance", "position", BOXES)
    assert descending and scores[0] == 1.0 and np.isnan(scores[1:]).all()
    scores, _ = detector_scores("Bees at the lower edge of the board", "position", BOXES)
    assert scores[1] == 0.5 and np.isnan(scores[[0, 2]]).all()  # only the 2 sitting bees count
    assert detector_scores("Bees above dry grass", "general", {**BOXES, "count": np.zeros(3)})[0].tolist() == [0, 0, 0]


def test_position_motion_counts_flying_or_sitting_bees_on_the_named_side():
    scores, descending = detector_scores("Bees flying near the right side of the entrance", "position-motion", BOXES)
    assert descending and scores[1] == 0.5 and np.isnan(scores[[0, 2]]).all()
    scores, descending = detector_scores("Bees resting on the left side", "position-motion", BOXES)
    assert descending and scores[0] == 1.0 and np.isnan(scores[[1, 2]]).all()


def spread_clip(columns):
    """Boxes of one clip with the given sitting bees per column (6 columns), as box_x / box_cls lists."""
    x = [(i + 0.5) / 6 for i, n in enumerate(columns) for _ in range(n)]
    return x, [0] * len(x)


def test_spread_scores_even_and_two_groups():
    clips = [spread_clip(c) for c in ([4, 4, 4, 4, 4, 4], [8, 8, 0, 0, 6, 3], [24, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 1])]
    boxes = {"box_x": [x for x, _ in clips], "box_cls": [c for _, c in clips]}
    even = spread_scores(boxes, "even")
    assert even[0] == pytest.approx(1.0) and even[2] < even[1] < even[0] and np.isnan(even[3])  # clip 3: too few bees
    groups = spread_scores(boxes, "groups")
    assert groups[1] == 6 and np.isnan(groups[[0, 2, 3]]).all()  # the smaller group's peak
    assert np.isnan(two_groups(np.array([8, 5, 3, 8, 0, 0])))  # the gap (3) is over a fifth of the busiest (8)
