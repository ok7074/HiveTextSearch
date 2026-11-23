"""Left/right test: do the methods notice which side of the board the bees are on?

A method that understands sides must change its answer when the query says "right" instead of "left" (test A), and
when the frames are mirrored (test B). One that ignores sides gives the same answer both times.
"""
import re

import numpy as np
import pandas as pd

from .detection import query_position

_SIDES = {"left": "right", "right": "left"}
_SIDE_WORD = re.compile(r"\b(left|right)\b", re.IGNORECASE)


def mirror_query(query: str) -> str | None:
    """
    The query with every "left" and "right" swapped in one pass, so "more on the right than on the left" becomes
    "more on the left than on the right"; a capital first letter stays capital. None when the query names no side.
    """
    if not _SIDE_WORD.search(query):
        return None

    def swap(match: re.Match) -> str:
        word = _SIDES[match.group(0).lower()]
        return word.capitalize() if match.group(0)[0].isupper() else word

    return _SIDE_WORD.sub(swap, query)


def is_middle_query(query: str) -> bool:
    """The negative control: a query about the middle of the board, which a mirror leaves where it is."""
    return query_position(query) == "center"


def mirror_detections(detections: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    The detections of the mirrored frames, without running YOLO again: left and right counts and labels swap and
    every box's x centre (a share of the frame width) becomes 1 - x. Heights and board lines do not change.
    """
    mirrored = dict(detections)
    mirrored["left"], mirrored["right"] = detections["right"], detections["left"]
    mirrored["label"] = np.array([_SIDES.get(label, label) for label in detections["label"]], dtype=object)
    mirrored["box_x"] = np.array([1 - np.asarray(x, dtype=float) for x in detections["box_x"]], dtype=object)
    return mirrored


def rank_correlation(a: np.ndarray, b: np.ndarray, a_descending: bool = True, b_descending: bool = True) -> float:
    """
    Spearman correlation between two rankings of the same clips. Scores are first turned so that higher is better
    (a method ranking fewest first is negated); clips a method left out (NaN) share the bottom rank.
    """
    ranks = [
        pd.Series(scores if descending else -scores).rank(na_option="bottom")
        for scores, descending in ((np.asarray(a, dtype=float), a_descending), (np.asarray(b, dtype=float), b_descending))
    ]
    return float(ranks[0].corr(ranks[1]))


def overlap(top_a: list[tuple[str, float]], top_b: list[tuple[str, float]]) -> int:
    """Number of clips two top_k lists share."""
    return len({clip for clip, _ in top_a} & {clip for clip, _ in top_b})
