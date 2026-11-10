"""Run every search method over the train or test queries and save each method's top 10 clips.

Methods:
  clip, siglip2  every query: cosine similarity between the query and each clip's averaged embedding
  motion         motion queries only: frame difference across each clip's 4 frames,
                 most movement first, or least first for quiet words such as "calm"
  detector       count and position queries only, once scripts/yolo/detect_clips.py has run: YOLO bee counts;
                 a query naming left/center/middle/right keeps clips with that label, ranked by bees in that third
                 (or, with a quiet word like "empty", clips with bees elsewhere, fewest on that side first);
                 count queries rank all clips by mean count (fewest first for quiet words such as "few");
                 position queries without a side word are left to the other methods (see src/detection.py)

Each method's top 10 holds at most one clip per source video, so neighbouring clips cannot fill it.

Output: data/runs/<split>/, a dataset with one row per query, method and rank:
  split, query_type, query, method, rank, clip_id, score
Nothing is printed about which method found which clip, so the clips can later be rated blind.
A later method (detector, tracking, combined score) is one more entry in the methods dict below.

Reads queries.txt (see src/retrieval.load_queries for the layout), data/embeddings/{clip,siglip2}.npz
and, the first time only, data/frames/ to build data/embeddings/motion.npz. Runs on CPU.

Each run is tracked in MLflow under the experiment "HiveTextSearch".

Usage:
    python scripts/retrieval/retrieval.py --split train
    python scripts/retrieval/retrieval.py --split test --queries path/to/queries.txt
"""
import argparse
import sys
import mlflow
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import Dataset, load_from_disk
from src import track_run
from src.detection import DETECTIONS_DIR, detector_scores, load_detections
from src.embeddings import EMBEDDINGS_DIR
from src.frame_selection import CLIPS_FRAMES_DATASET_DIR, FRAMES_DIR
from src.models import clip_text_features, load_clip, load_siglip2, siglip2_text_features
from src.retrieval import (
    QUERIES_FILE, RUNS_DIR, SPLITS, TOP_K,
    is_motion_query, load_queries, motion_descending, motion_scores, text_scores, top_k,
)

TEXT_MODELS = {
    "clip": (load_clip, clip_text_features),
    "siglip2": (load_siglip2, siglip2_text_features),
}
MOTION_FILE = EMBEDDINGS_DIR / "motion.npz"


def load_clip_embeddings(name: str) -> tuple[np.ndarray, np.ndarray]:
    with np.load(EMBEDDINGS_DIR / f"{name}.npz", allow_pickle=False) as saved:
        return saved["clip_ids"], saved["clip_embeddings"]


def load_motion_scores() -> tuple[np.ndarray, np.ndarray]:
    """Frame-difference scores do not depend on the query, so they are computed once and saved."""
    if not MOTION_FILE.exists():
        dataset = load_from_disk(str(CLIPS_FRAMES_DATASET_DIR))
        print("Computing motion scores from the saved frames (first run only)...", flush=True)
        np.savez(MOTION_FILE, clip_ids=np.array(dataset["clip_id"]), motion_scores=motion_scores(dataset, FRAMES_DIR))
    with np.load(MOTION_FILE, allow_pickle=False) as saved:
        return saved["clip_ids"], saved["motion_scores"]


def text_method(load_model, text_features, clip_embeddings: np.ndarray):
    # device_map=None keeps the model on CPU, which is fast enough for one query at a time.
    model, processor = load_model(device_map=None)
    return lambda q: (text_scores(q["query"], clip_embeddings, lambda texts: text_features(model, processor, texts)), True)


def load_methods() -> tuple[np.ndarray, dict]:
    """
    Every method maps a query dict to (one score per clip aligned with clip_ids, rank highest first?),
    or to None when it does not apply to that query type.
    """
    clip_ids, methods = None, {}
    for name, (load_model, text_features) in TEXT_MODELS.items():
        ids, embeddings = load_clip_embeddings(name)
        clip_ids = ids if clip_ids is None else clip_ids
        if not np.array_equal(ids, clip_ids):
            raise ValueError(f"{name}.npz lists different clips than the other embedding files")
        methods[name] = text_method(load_model, text_features, embeddings)
    motion_ids, motion = load_motion_scores()
    if not np.array_equal(motion_ids, clip_ids):
        raise ValueError(f"{MOTION_FILE} lists different clips than the embeddings; delete it to rebuild")
    methods["motion"] = lambda q: (motion, motion_descending(q["query"])) if is_motion_query(q["query_type"]) else None
    if DETECTIONS_DIR.exists():
        detections = load_detections(clip_ids)
        methods["detector"] = lambda q: detector_scores(q["query"], q["query_type"], detections)
    else:
        print(f"No {DETECTIONS_DIR} yet: run scripts/yolo/detect_clips.py to add the detector method", flush=True)
    return clip_ids, methods


@track_run(run_name="retrieval")
def main(split: str, queries_file: str) -> None:
    queries = load_queries(queries_file, split)
    clip_ids, methods = load_methods()

    rows = []
    for number, q in enumerate(queries, 1):
        for name, method in methods.items():
            if (result := method(q)) is None:
                continue
            scores, descending = result
            for rank, (clip_id, score) in enumerate(top_k(scores, clip_ids, TOP_K, descending), 1):
                rows.append({"split": split, **q, "method": name, "rank": rank, "clip_id": clip_id, "score": score})
        print(f"{number}/{len(queries)} [{q['query_type']}] {q['query']}", flush=True)

    output_dir = RUNS_DIR / split
    Dataset.from_list(rows).save_to_disk(str(output_dir))
    mlflow.log_params({"top_k": TOP_K, "n_queries": len(queries), "methods": ",".join(methods)})
    mlflow.log_metric("rows_saved", len(rows))
    print(f"Saved {len(rows)} rows to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--queries", default=str(QUERIES_FILE), help="query file (default: queries.txt in the project)")
    args = parser.parse_args()
    main(split=args.split, queries_file=args.queries)
