"""Compute mAP@10 and nDCG@10 per method and query type from your ratings.

Reads data/runs/<split> (each method's ranked top 10) and experiments/ratings_<split>.csv (your 0/1/2 ratings)
and writes results/<split>_metrics.json:
  summary    method -> query type -> n_queries, map, map_std, ndcg, ndcg_std
  per_query  AP and nDCG for every query and method, to show the spread
Relevant = rating 1 or 2. Clips left unrated count as 0; their number is printed and saved as unrated_clips.
A method only appears for the query types it ran on (motion: motion queries; detector: count, position
and general; rrf: all).

Usage:
    python scripts/scoring/scoring.py
    python scripts/scoring/scoring.py --split test
"""
import argparse
import json
import sys
from pathlib import Path

import mlflow
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src import track_run
from src.evaluation import RATINGS_DIR, RESULTS_DIR, read_ratings, score
from src.retrieval import RUNS_DIR, SPLITS


@track_run(run_name="scoring")
def main(split: str) -> None:
    runs = load_from_disk(str(RUNS_DIR / split)).to_pandas()
    metrics = score(runs, read_ratings(RATINGS_DIR / f"ratings_{split}.csv"))

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{split}_metrics.json"
    path.write_text(json.dumps({"split": split, **metrics}, indent=2), encoding="utf-8")
    for method, by_type in metrics["summary"].items():
        for query_type, m in by_type.items():
            print(f"{method:9} {query_type:10} n={m['n_queries']:2}  mAP {m['map']}  nDCG {m['ndcg']}")
            mlflow.log_metrics({f"{method}_{query_type}_map": m["map"] or 0, f"{method}_{query_type}_ndcg": m["ndcg"] or 0})
    print(f"Unrated clips counted as 0: {metrics['unrated_clips']}. Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=SPLITS, default="train")
    main(split=parser.parse_args().split)
