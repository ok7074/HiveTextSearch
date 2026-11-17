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
BEE_CLASS = 0
BAND_WORDS = {
    "above": "above", "upper": "above", "top": "above", "over": "above",
    "below": "below", "lower": "below", "bottom": "below", "under": "below", "beneath": "below",
}
# shortcut: one margin (~58 px, about a bee length) for every hive; tune per hive if the top clips look wrong.
LOWER_EDGE_MARGIN = 0.03
SPREAD_COLUMNS = 6
SPREAD_MIN_BEES = 3  # mean sitting bees per frame a clip needs before "spread" or "groups" means anything
BOX_COLUMNS = ("box_x", "box_y", "box_cls")


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
      count (mean boxes per frame), flying (mean bee_flying boxes per frame), label (see clip_label) and
      box_x, box_y, box_cls (every box of the 4 frames: centre as a share of frame width / height, and class).
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
            "box_x": [float(v) for r in results for v in r.boxes.xywhn[:, 0].cpu().numpy()],
            "box_y": [float(v) for r in results for v in r.boxes.xywhn[:, 1].cpu().numpy()],
            "box_cls": [int(v) for r in results for v in r.boxes.cls.cpu().numpy()],
        })
        if number % 100 == 0:
            print(f"{number}/{len(dataset)} clips", flush=True)
    return rows


def load_detections(clip_ids: np.ndarray, detections_dir: Path = DETECTIONS_DIR) -> dict[str, np.ndarray]:
    """YOLO counts, labels, boxes and the video's board lines per clip, reordered so row i belongs to clip_ids[i]."""
    detections = load_from_disk(str(detections_dir)).to_pandas().set_index("clip_id")
    if set(detections.index) != set(clip_ids):
        raise ValueError(f"{detections_dir} lists different clips than the embeddings; rerun scripts/yolo/detect_clips.py")
    if missing := [c for c in BOX_COLUMNS if c not in detections]:
        raise ValueError(f"{detections_dir} has no {missing}; rerun scripts/yolo/detect_clips.py")
    detections = detections.loc[clip_ids]
    columns = ("left", "center", "right", "count", "flying", "label", *BOX_COLUMNS)
    loaded = {column: detections[column].to_numpy() for column in columns}
    loaded["entrance"], loaded["board_middle"] = board_lines(np.asarray(clip_ids), loaded["box_y"], loaded["box_cls"])
    return loaded


def board_lines(clip_ids: np.ndarray, box_y, box_cls) -> tuple[np.ndarray, np.ndarray]:
    """
    Per clip, two heights (share of frame height) from the sitting bees in all clips of its video, which crowd the
    landing board just below the entrance:
      - entrance: top edge of that crowd, Q1 - 1.5 x (median - Q1); quartiles ignore the few boxes far above in the mesh
      - board_middle: the median
    The video is the clip id without its 2-digit clip number. A video without sitting bees gets the median over videos.
    """
    videos = np.array([c[:-2] for c in clip_ids])
    sitting = [np.asarray(y, dtype=float)[np.asarray(c) == BEE_CLASS] for y, c in zip(box_y, box_cls)]
    per_video = {}
    for video in set(videos):
        heights = np.concatenate([sitting[i] for i in np.flatnonzero(videos == video)])
        if len(heights):
            q1, median = np.quantile(heights, [0.25, 0.5])
            per_video[video] = (q1 - 1.5 * (median - q1), median)
    fallback = tuple(np.median(list(per_video.values()), axis=0)) if per_video else (0.45, 0.5)
    lines = np.array([per_video.get(v, fallback) for v in videos], dtype=float)
    return lines[:, 0], lines[:, 1]


def section_counts(detections: dict, third: str | None = None, band: str | None = None,
                   cls: int | None = None) -> np.ndarray:
    """
    Mean boxes per frame in a third (left/center/right), a band, or both. above = above the video's entrance edge;
    below = more than LOWER_EDGE_MARGIN under the board middle (the lower edge of the board and lower).
    """
    counts = []
    rows = zip(detections["box_x"], detections["box_y"], detections["box_cls"],
               detections["entrance"], detections["board_middle"])
    for x, y, c, entrance, middle in rows:
        x, y, c = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(c)
        keep = np.ones(len(x), dtype=bool)
        if third is not None:
            keep &= np.minimum((x * 3).astype(int), 2) == THIRDS.index(third)
        if band == "above":
            keep &= y < entrance
        elif band == "below":
            keep &= y > middle + LOWER_EDGE_MARGIN
        if cls is not None:
            keep &= c == cls
        counts.append(keep.sum() / N_FRAMES)
    return np.array(counts, dtype=np.float32)


def spread_scores(detections: dict, spread: str) -> np.ndarray:
    """
    How the sitting bees spread along the board, from their counts in SPREAD_COLUMNS columns; higher is better.
    Clips with fewer than SPREAD_MIN_BEES sitting bees per frame get NaN.
      - even: 1 / (1 + coefficient of variation of the column counts); 1 = the same number in every column
      - groups: see two_groups
    """
    scores = []
    for x, c in zip(detections["box_x"], detections["box_cls"]):
        x = np.asarray(x, dtype=float)[np.asarray(c) == BEE_CLASS]
        columns = np.bincount(np.minimum((x * SPREAD_COLUMNS).astype(int), SPREAD_COLUMNS - 1), minlength=SPREAD_COLUMNS)
        if columns.sum() < SPREAD_MIN_BEES * N_FRAMES:
            scores.append(np.nan)
        elif spread == "even":
            scores.append(1 / (1 + columns.std() / columns.mean()))
        else:
            scores.append(two_groups(columns))
    return np.array(scores, dtype=np.float32)


def two_groups(columns: np.ndarray) -> float:
    """
    Exactly two runs of busy columns (at least half the busiest) with a column of at most a fifth of the busiest
    between them: the smaller run's peak. Anything else: NaN.
    """
    busy = columns >= 0.5 * columns.max()
    starts = np.flatnonzero(busy & ~np.r_[False, busy[:-1]])
    ends = np.flatnonzero(busy & ~np.r_[busy[1:], False])
    if len(starts) != 2 or columns[ends[0] + 1:starts[1]].min() > 0.2 * columns.max():
        return np.nan
    return float(min(columns[starts[i]:ends[i] + 1].max() for i in range(2)))


def query_position(query: str) -> str | None:
    """The third a query asks for ("middle" and "centre" count as center), or None."""
    return next((POSITION_WORDS[w] for w in re.findall(r"[a-z]+", query.lower()) if w in POSITION_WORDS), None)


def query_band(query: str) -> str | None:
    """"above" or "below" the entrance line when the query names one ("above the entrance", "lower edge"), or None."""
    return next((BAND_WORDS[w] for w in re.findall(r"[a-z]+", query.lower()) if w in BAND_WORDS), None)


def query_spread(query: str) -> str | None:
    """"even" for evenly spread bees, "groups" for two separate groups or clusters, or None."""
    words = set(re.findall(r"[a-z]+", query.lower()))
    if words & {"evenly", "even", "spread"}:
        return "even"
    if words & {"group", "groups", "clusters"} and words & {"two", "separate"}:
        return "groups"
    return None


def detector_scores(query: str, query_type: str, detections: dict[str, np.ndarray]) -> tuple[np.ndarray, bool] | None:
    """
    Detector ranking for count, position, general and motion queries; None (not applicable) otherwise. Excluded clips
    get NaN, which top_k drops. Quiet words such as "few" or "empty" put the fewest bees first.
      - position-motion query naming a place (third and/or band): boxes in that place only; flying bees for busy
        words, sitting bees for quiet words ("resting in the middle"); most first, clips with none there left out
      - motion query: all clips by mean bee_flying boxes per frame (fewest first for quiet words such as "still")
      - position query naming a band ("above the entrance", "lower edge"): sitting bees above the video's entrance /
        on the lower part of the board (inside the named third too, if any); clips with none there are left out
      - names a third, busy ("bees on the left"): clips with that label, ranked by the boxes in that third
      - names a third, quiet ("left side empty"): clips with bees in the other thirds, fewest in that third first
      - count or general query without a third: all clips by mean count; "few"/"handful"/"couple" leave out clips
        with no bees. For general queries this is a measured test: a count mostly encodes which hive it is.
      - position query about spread ("evenly spread", "two separate groups"): see spread_scores
      - position query naming no place and no spread: None, since a total count says nothing about where bees sit
    Band and spread words are only read in position queries, so "flying low over the grass" stays a motion query.
    """
    if not any(kind in query_type for kind in DETECTOR_TYPES):
        return None
    position, descending = query_position(query), motion_descending(query)
    is_position = "position" in query_type
    band = query_band(query) if is_position else None
    if "motion" in query_type:
        if is_position and (position or band):
            scores = section_counts(detections, position, band, FLYING_CLASS if descending else BEE_CLASS)
            return np.where(scores > 0, scores, np.nan).astype(np.float32), True
        return detections["flying"].astype(np.float32), descending
    if band is not None:
        scores = section_counts(detections, position, band, BEE_CLASS)  # bees sitting there, not flying past
        return (np.where(scores > 0, scores, np.nan).astype(np.float32) if descending else scores), descending
    if position is None:
        if is_position:
            spread = query_spread(query)
            return (spread_scores(detections, spread), True) if spread else None
        scores = detections["count"].astype(np.float32)
        if set(re.findall(r"[a-z]+", query.lower())) & FEW_WORDS:
            scores = np.where(scores > 0, scores, np.nan).astype(np.float32)
        return scores, descending
    if not descending:
        elsewhere = sum(detections[third] for third in THIRDS) - detections[position]
        return np.where(elsewhere > 0, detections[position], np.nan).astype(np.float32), False
    return np.where(detections["label"] == position, detections[position], np.nan).astype(np.float32), True
