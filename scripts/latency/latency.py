"""Time how long each method takes to answer a query, over 200 queries per method.

Uses the train and test queries from queries.txt, cycling through the ones a method applies to
(motion: motion queries; detector: count, position, general and motion queries; rrf: all, timed including every
method it fuses) until it has answered --n of them.
One timed answer = score every clip + pick the top 10; loading the models is not timed, and the first call
per method (a warm-up) is not counted. Writes p50 and p99 in milliseconds to results/latency.json.
Motion is near 0 ms because its scores are precomputed and do not depend on the query text.

Usage:
    python scripts/latency/latency.py
    python scripts/latency/latency.py --n 500
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import mlflow
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.retrieval.retrieval import load_methods
from src import track_run
from src.evaluation import RESULTS_DIR
from src.retrieval import QUERIES_FILE, TOP_K, load_queries, top_k


@track_run(run_name="latency")
def main(queries_file: str, n: int) -> None:
    queries = load_queries(queries_file, "train") + load_queries(queries_file, "test")
    clip_ids, methods = load_methods()

    latency = {}
    for name, method in methods.items():
        applicable = [q for q in queries if method(q) is not None]  # also warms the method up
        if not applicable:
            print(f"{name}: no queries of its type, skipped")
            continue
        times = []
        for q in itertools.islice(itertools.cycle(applicable), n):
            start = time.perf_counter()
            scores, descending = method(q)
            top_k(scores, clip_ids, TOP_K, descending)
            times.append((time.perf_counter() - start) * 1000)
        latency[name] = {
            "n": n, "distinct_queries": len(applicable),
            "p50_ms": round(float(np.percentile(times, 50)), 3), "p99_ms": round(float(np.percentile(times, 99)), 3),
        }
        mlflow.log_metrics({f"{name}_p50_ms": latency[name]["p50_ms"], f"{name}_p99_ms": latency[name]["p99_ms"]})
        print(f"{name:9} p50 {latency[name]['p50_ms']:8.2f} ms   p99 {latency[name]['p99_ms']:8.2f} ms", flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / "latency.json"
    path.write_text(json.dumps(latency, indent=2), encoding="utf-8")
    print(f"Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--queries", default=str(QUERIES_FILE), help="query file (default: queries.txt in the project)")
    parser.add_argument("--n", type=int, default=200, help="timed queries per method")
    args = parser.parse_args()
    main(queries_file=args.queries, n=args.n)
