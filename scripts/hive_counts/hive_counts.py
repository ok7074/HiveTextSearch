"""Count how many of each method's top 10 clips come from each hive, per split and query.

Reads data/runs/{train,test} (scripts/retrieval/retrieval.py) and writes one CSV row per split, query and method:
  split, query_type, query, method, orange, red-stripe, n
Prints a summary per split and method: the mean share of the top 10 taken by its majority hive, and the number of
queries whose whole list comes from one hive. For comparison, all clips split 1,125 orange / 315 red-stripe
(78% / 22%): a method picking in that ratio shows no hive shortcut; one always at 10/0 does.

Usage:
    python scripts/hive_counts/hive_counts.py --out results/hive_counts.csv
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.retrieval import HIVES, RUNS_DIR, SPLITS, hive_counts


def summary(counts: pd.DataFrame) -> pd.DataFrame:
    """Per split and method: mean majority-hive share of the list and queries with every clip from one hive."""
    hives = counts[list(HIVES.values())]
    counts = counts.assign(majority_share=hives.max(axis=1) / counts["n"], one_hive=hives.min(axis=1) == 0)
    return counts.groupby(["split", "method"]).agg(
        n_queries=("query", "size"), majority_share=("majority_share", "mean"), all_one_hive=("one_hive", "sum"),
    ).round(3).reset_index()


def main(out: str) -> None:
    runs = pd.concat([load_from_disk(str(RUNS_DIR / split)).to_pandas() for split in SPLITS])
    counts = hive_counts(runs)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(out, index=False)
    print(summary(counts).to_string(index=False))
    print(f"Saved {len(counts)} rows to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True, help="CSV to write the per-query counts to")
    main(out=parser.parse_args().out)
