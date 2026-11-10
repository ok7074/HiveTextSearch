import re
from pathlib import Path

import numpy as np
from datasets import Dataset

from .frame_selection import DATA_DIR, N_FRAMES
from .retrieval import motion_descending

DETECTIONS_DIR = DATA_DIR / "detections"
DEFAULT_WEIGHTS = DATA_DIR.parent / "experiments" / "yolo_runs" / "train-11" / "weights" / "best.pt"
THIRDS = ("left", "center", "right")
POSITION_WORDS = {"left": "left", "center": "center", "centre": "center", "middle": "center", "right": "right"}


def third_counts(x_centers, width: int) -> np.ndarray:
    """Number of boxes in each vertical third (left, center, right), from the boxes' x centres in pixels."""
    thirds = np.minimum((np.asarray(x_centers, dtype=float) * 3 // width).astype(int), 2)
    return np.bincount(thirds, minlength=3)


def clip_label(counts) -> str:
    """The third with the most boxes; "overall" on a tie, "empty" with no boxes at all."""
    counts = np.asarray(counts)
    if counts.sum() == 0:
        return "empty"
    best = np.flatnonzero(counts == counts.max())
    return THIRDS[best[0]] if len(best) == 1 else "overall"


def detect_clips(model, dataset: Dataset, frames_dir: Path) -> list[dict]:
    """
    Run a YOLO model on every clip's saved frames and return one row per clip:
      clip_id, frame_counts (boxes per frame), left/center/right (boxes per third summed over the frames),
      count (mean boxes per frame) and label (see clip_label).
    """
    rows = []
    for number, (clip_id, names) in enumerate(zip(dataset["clip_id"], dataset["frame_names"]), 1):
        if len(names) != N_FRAMES:
            raise ValueError(f"Expected {N_FRAMES} frames, got {names}")
        paths = [frames_dir / f"{name}.jpg" for name in names]
        if missing := [str(p) for p in paths if not p.exists()]:
            raise FileNotFoundError(f"Missing frames: {missing}")
        results = model.predict([str(p) for p in paths], device="cpu", verbose=False)
        per_frame = [third_counts(r.boxes.xywh[:, 0].cpu().numpy(), r.orig_shape[1]) for r in results]
        total = np.sum(per_frame, axis=0)
        rows.append({
            "clip_id": clip_id,
            "frame_counts": [int(c.sum()) for c in per_frame],
            **{third: int(n) for third, n in zip(THIRDS, total)},
            "count": float(total.sum() / N_FRAMES),
            "label": clip_label(total),
        })
        if number % 100 == 0:
            print(f"{number}/{len(dataset)} clips", flush=True)
    return rows


def query_position(query: str) -> str | None:
    """The third a query asks for ("middle" and "centre" count as center), or None."""
    return next((POSITION_WORDS[w] for w in re.findall(r"[a-z]+", query.lower()) if w in POSITION_WORDS), None)


def detector_scores(query: str, query_type: str, detections: dict[str, np.ndarray]) -> tuple[np.ndarray, bool] | None:
    """
    Detector ranking for count and position queries; None (not applicable) for every other query type.
    A query naming a third keeps only clips with that label, ranked by the boxes in that third;
    excluded clips get NaN, which top_k drops. Otherwise all clips are ranked by their mean count.
    Quiet words such as "few" or "empty" put the fewest bees first.
    """
    if "count" not in query_type and "position" not in query_type:
        return None
    position = query_position(query)
    if position is None:
        return detections["count"].astype(np.float32), motion_descending(query)
    scores = np.where(detections["label"] == position, detections[position], np.nan).astype(np.float32)
    return scores, motion_descending(query)
