import re
from pathlib import Path

import numpy as np
from datasets import Dataset, load_from_disk

from .frame_selection import DATA_DIR, N_FRAMES
from .retrieval import motion_descending

DETECTIONS_DIR = DATA_DIR / "detections"
DEFAULT_WEIGHTS = DATA_DIR.parent / "experiments" / "yolo_runsFinal" / "experiments" / "yolo" / "final_s960" / "weights" / "best.pt"
THIRDS = ("left", "center", "right")
POSITION_WORDS = {"left": "left", "center": "center", "centre": "center", "middle": "center", "right": "right"}
# "A handful of bees" means at least one bee, unlike "empty" or "no bees".
FEW_WORDS = {"few", "handful", "couple"}
DETECTOR_TYPES = ("count", "position", "general", "motion")
FLYING_CLASS = 1  # bee_flying in the final model's data.yaml


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
      count (mean boxes per frame), flying (mean bee_flying boxes per frame) and label (see clip_label).
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
        flying = sum(int((r.boxes.cls == FLYING_CLASS).sum()) for r in results)
        rows.append({
            "clip_id": clip_id,
            "frame_counts": [int(c.sum()) for c in per_frame],
            **{third: int(n) for third, n in zip(THIRDS, total)},
            "count": float(total.sum() / N_FRAMES),
            "flying": float(flying / N_FRAMES),
            "label": clip_label(total),
        })
        if number % 100 == 0:
            print(f"{number}/{len(dataset)} clips", flush=True)
    return rows


def load_detections(clip_ids: np.ndarray, detections_dir: Path = DETECTIONS_DIR) -> dict[str, np.ndarray]:
    """YOLO counts and labels per clip, reordered so row i belongs to clip_ids[i]."""
    detections = load_from_disk(str(detections_dir)).to_pandas().set_index("clip_id")
    if set(detections.index) != set(clip_ids):
        raise ValueError(f"{detections_dir} lists different clips than the embeddings; rerun scripts/yolo/detect_clips.py")
    detections = detections.loc[clip_ids]
    return {column: detections[column].to_numpy() for column in ("left", "center", "right", "count", "flying", "label")}


def query_position(query: str) -> str | None:
    """The third a query asks for ("middle" and "centre" count as center), or None."""
    return next((POSITION_WORDS[w] for w in re.findall(r"[a-z]+", query.lower()) if w in POSITION_WORDS), None)


def detector_scores(query: str, query_type: str, detections: dict[str, np.ndarray]) -> tuple[np.ndarray, bool] | None:
    """
    Detector ranking for count, position, general and motion queries; None (not applicable) otherwise. Excluded clips
    get NaN, which top_k drops. Quiet words such as "few" or "empty" put the fewest bees first.
      - motion query: all clips by mean bee_flying boxes per frame (fewest first for quiet words such as "still")
      - names a third, busy ("bees on the left"): clips with that label, ranked by the boxes in that third
      - names a third, quiet ("left side empty"): clips with bees in the other thirds, fewest in that third first
      - count or general query without a third: all clips by mean count; "few"/"handful"/"couple" leave out clips
        with no bees. For general queries this is a measured test: a count mostly encodes which hive it is.
      - position query without a third: None, since a total count says nothing about where bees sit
    """
    if not any(kind in query_type for kind in DETECTOR_TYPES):
        return None
    position, descending = query_position(query), motion_descending(query)
    if "motion" in query_type:
        return detections["flying"].astype(np.float32), descending
    if position is None:
        if "position" in query_type:
            return None
        scores = detections["count"].astype(np.float32)
        if set(re.findall(r"[a-z]+", query.lower())) & FEW_WORDS:
            scores = np.where(scores > 0, scores, np.nan).astype(np.float32)
        return scores, descending
    if not descending:
        elsewhere = sum(detections[third] for third in THIRDS) - detections[position]
        return np.where(elsewhere > 0, detections[position], np.nan).astype(np.float32), False
    return np.where(detections["label"] == position, detections[position], np.nan).astype(np.float32), True
