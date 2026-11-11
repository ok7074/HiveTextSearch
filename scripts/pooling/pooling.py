"""Merge every method's top 10 into one blind rating sheet per query.

Reads data/runs/<split> (from scripts/retrieval/retrieval.py) and writes experiments/ratings_<split>.csv:
  query_type, query, clip_id, rating
One row per query and clip, even when several methods found the same clip. Clips are shuffled within each
query, and method and rank are left out, so you rate without knowing who found a clip.
Fill the rating column with 0 (not relevant), 1 (partly) or 2 (clearly), then run scripts/scoring/scoring.py.

Running it again after adding a method (e.g. the detector) keeps every rating you already wrote
and only appends the new clips at the bottom. Commit the CSV after each rating session: it is your only copy.

Usage:
    python scripts/pooling/pooling.py
    python scripts/pooling/pooling.py --split test
"""
import argparse
import sys
from pathlib import Path

from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.evaluation import RATINGS_DIR, pool, read_ratings, write_ratings
from src.retrieval import RUNS_DIR, SPLITS


def main(split: str) -> None:
    runs = load_from_disk(str(RUNS_DIR / split)).to_pandas()
    path = RATINGS_DIR / f"ratings_{split}.csv"
    existing = read_ratings(path) if path.exists() else None
    ratings = pool(runs, existing)
    added = len(ratings) - (0 if existing is None else len(existing))
    write_ratings(ratings, path)
    print(f"{path}: {len(ratings)} clips to rate over {ratings['query'].nunique()} queries, {added} new")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=SPLITS, default="train")
    main(split=parser.parse_args().split)
