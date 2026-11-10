import re
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from datasets import Dataset

from .frame_selection import DATA_DIR, N_FRAMES

QUERIES_FILE = DATA_DIR.parent / "queries.txt"
RUNS_DIR = DATA_DIR / "runs"
SPLITS = ("train", "test")
TOP_K = 10
# shortcut: plain word match, so "not busy" still ranks the busiest clips first; switch to per-query rules if that matters.
QUIET_WORDS = {"calm", "still", "quiet", "resting", "rest", "idle", "few", "empty", "slow", "inactive", "little", "no"}


def load_queries(path: str | Path, split: str) -> list[dict]:
    """
    Read the queries of one split from a text file laid out as:

        Train:
        Count:
        many bees on the board
        Motion:
        busy entrance
        Test:
        ...

    A line ending in ":" is a heading: Train/Test start a split, anything else names the query type.
    Blank lines are skipped and leading "-", "*" or "1." are removed.
    Returns [{"query_type": ..., "query": ...}] in file order, query types in lower case.
    """
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
    queries, current_split, query_type = [], None, None
    for raw in Path(path).read_text(encoding="utf-8-sig").splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", raw).strip()
        if not line:
            continue
        if line.endswith(":"):
            heading = line[:-1].strip().lower()
            if heading in SPLITS:
                current_split, query_type = heading, None
            else:
                query_type = heading
            continue
        if current_split is None or query_type is None:
            raise ValueError(f"Query outside a Train/Test heading and a query-type heading: {raw!r}")
        if current_split == split:
            queries.append({"query_type": query_type, "query": line})
    if not queries:
        raise ValueError(f"No {split} queries found in {path}")
    return queries


def is_motion_query(query_type: str) -> bool:
    return "motion" in query_type


def text_scores(query: str, clip_embeddings: np.ndarray, text_features: Callable) -> np.ndarray:
    """Cosine similarity between the query and every clip; both sides are unit length, so a dot product suffices."""
    return clip_embeddings @ text_features([query]).float().cpu().numpy()[0]


def frame_difference(frames: list[np.ndarray]) -> float:
    """Mean absolute pixel change between consecutive grayscale frames (0-255 scale)."""
    frames = [frame.astype(np.float32) for frame in frames]
    return float(np.mean([np.abs(b - a).mean() for a, b in zip(frames, frames[1:])]))


def motion_scores(dataset: Dataset, frames_dir: Path) -> np.ndarray:
    """One frame-difference score per clip, in dataset order, from its saved JPG frames."""
    scores = []
    for names in dataset["frame_names"]:
        if len(names) != N_FRAMES:
            raise ValueError(f"Expected {N_FRAMES} frames, got {names}")
        frames = [cv2.imread(str(frames_dir / f"{name}.jpg"), cv2.IMREAD_GRAYSCALE) for name in names]
        if any(frame is None for frame in frames):
            raise FileNotFoundError(f"Missing or unreadable frame among {names} in {frames_dir}")
        scores.append(frame_difference(frames))
    return np.array(scores, dtype=np.float32)


def motion_descending(query: str) -> bool:
    """True when the query asks for activity (most movement first), False for quiet words like "calm"."""
    return not set(re.findall(r"[a-z]+", query.lower())) & QUIET_WORDS


def top_k(scores: np.ndarray, clip_ids: np.ndarray, k: int = TOP_K, descending: bool = True) -> list[tuple[str, float]]:
    """The k best (clip_id, score) pairs; ties keep clip order so reruns give the same list. NaN scores are left out."""
    order = np.argsort(-scores if descending else scores, kind="stable")
    order = order[~np.isnan(scores[order])][:k]
    return [(str(clip_ids[i]), float(scores[i])) for i in order]
