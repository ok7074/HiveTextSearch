import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .frame_selection import DATA_DIR
from .retrieval import TOP_K

RATINGS_DIR = DATA_DIR.parent / "experiments"
RESULTS_DIR = DATA_DIR.parent / "results"
FAILURES_FILE = RATINGS_DIR / "failures.csv"
KEY = ["query_type", "query", "clip_id"]
RATINGS = (0, 1, 2)
FAILURE_CATEGORIES = (
    "hive shortcut", "wrong side or position", "negation or comparison", "wrong count",
    "flying or blurry bee", "lighting or shadow", "other",
)
FAILURE_COLUMNS = ["split", "query_type", "query", "clip_id", "method", "category", "note", "time"]


def write_ratings(ratings: pd.DataFrame, path: str | Path) -> None:
    """Write the ratings CSV via a copy that is swapped in, so a failed write never leaves it half-erased."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    ratings.to_csv(tmp, index=False)
    os.replace(tmp, path)


def fully_rated_queries(runs: pd.DataFrame, ratings: pd.DataFrame) -> set[tuple[str, str]]:
    """
    (query_type, query) pairs whose clips are all rated: every clip any method returned is in the sheet with a rating.
    A rerun that adds a method brings unpooled clips, which locks the query again until they are pooled and rated.
    """
    merged = runs[KEY].drop_duplicates().merge(ratings[KEY + ["rating"]], on=KEY, how="left")
    done = merged.groupby(["query_type", "query"])["rating"].apply(lambda r: r.notna().all())
    return set(done[done].index)


def append_failure(row: dict, path: str | Path = FAILURES_FILE) -> None:
    """Append one failure note (FAILURE_COLUMNS; time is filled in) to the failures CSV, creating it if needed."""
    path = Path(path)
    if row.get("category") not in FAILURE_CATEGORIES:
        raise ValueError(f"category must be one of {FAILURE_CATEGORIES}, got {row.get('category')!r}")
    record = pd.DataFrame([{**{c: row.get(c, "") for c in FAILURE_COLUMNS}, "time": datetime.now().isoformat(timespec="seconds")}])
    path.parent.mkdir(parents=True, exist_ok=True)
    record.to_csv(path, mode="a", header=not path.exists(), index=False)


def pool(runs: pd.DataFrame, existing: pd.DataFrame | None = None, seed: int = 0) -> pd.DataFrame:
    """
    One row per query and clip found by any method (duplicates across methods removed), with an empty rating.
    Clips are shuffled within each query, and method and rank are left out, so the rater cannot tell who found what.
    Rows already in existing keep their place and rating; only clips not rated before are appended.
    """
    new = runs[KEY].drop_duplicates()
    if existing is not None:
        new = new.merge(existing[KEY], how="left", indicator=True).query("_merge == 'left_only'").drop(columns="_merge")
    query_order = {q: i for i, q in enumerate(runs["query"].drop_duplicates())}
    new = new.sample(frac=1, random_state=seed).sort_values("query", key=lambda s: s.map(query_order), kind="stable")
    new = new.assign(rating=pd.Series(pd.NA, index=new.index, dtype="Int64"))
    return new.reset_index(drop=True) if existing is None else pd.concat([existing, new], ignore_index=True)


def read_ratings(path: str | Path) -> pd.DataFrame:
    """Read a ratings CSV (comma or semicolon, as Excel may save it); ratings must be 0, 1, 2 or empty."""
    ratings = pd.read_csv(path, sep=None, engine="python", dtype={c: str for c in KEY})
    if missing := set(KEY + ["rating"]) - set(ratings.columns):
        raise ValueError(f"{path} is missing columns {sorted(missing)}")
    numeric = pd.to_numeric(ratings["rating"], errors="coerce")
    if (bad := ratings["rating"].notna() & ~numeric.isin(RATINGS)).any():
        raise ValueError(f"Ratings must be 0, 1, 2 or empty; bad rows:\n{ratings[bad]}")
    if (dupes := ratings.duplicated(KEY)).any():
        raise ValueError(f"Clips rated twice for the same query:\n{ratings[dupes]}")
    return ratings.assign(rating=numeric.astype("Int64"))  # Int64 keeps empty cells and saves 2, not 2.0


def average_precision(gains, n_relevant: int) -> float:
    """Mean of the precision at each relevant position (gain >= 1), divided by n_relevant; NaN if nothing is relevant."""
    if n_relevant == 0:
        return float("nan")
    relevant = np.asarray(gains) >= 1
    precision = np.cumsum(relevant) / np.arange(1, len(relevant) + 1)
    return float((precision * relevant).sum() / n_relevant)


def dcg(gains) -> float:
    gains = np.asarray(gains, dtype=float)
    return float((gains / np.log2(np.arange(2, len(gains) + 2))).sum())


def ndcg(gains, ideal_gains) -> float:
    """DCG of the ranked gains over the DCG of the best possible order; NaN if the ideal is 0."""
    ideal = dcg(ideal_gains)
    return dcg(gains) / ideal if ideal else float("nan")


def score(runs: pd.DataFrame, ratings: pd.DataFrame, k: int = TOP_K) -> dict:
    """
    AP@k and nDCG@k for every query and method, plus their mean and spread per method and query type.
    The pool of a query = every clip rated for it. Relevant = rating >= 1. AP is divided by the relevant
    clips in the pool (at most k); the nDCG ideal is the k best ratings in the pool. Unrated clips count as 0.
    """
    runs = runs.merge(ratings[KEY + ["rating"]], on=KEY, how="left")
    unrated = int(runs.drop_duplicates(KEY)["rating"].isna().sum())
    runs["rating"] = runs["rating"].fillna(0)
    pools = ratings.fillna({"rating": 0}).groupby(["query_type", "query"])["rating"].apply(list)

    per_query = []
    for (query_type, query, method), hits in runs.groupby(["query_type", "query", "method"], sort=False):
        gains = hits.sort_values("rank")["rating"].head(k).tolist()
        pool_ratings = pools.get((query_type, query), [])
        n_relevant = min(sum(r >= 1 for r in pool_ratings), k)
        per_query.append({
            "query_type": query_type, "query": query, "method": method,
            "ap": average_precision(gains, n_relevant),
            "ndcg": ndcg(gains, sorted(pool_ratings, reverse=True)[:k]),
        })

    per_query = pd.DataFrame(per_query)
    summary = {}
    # No "all" row: methods run on different query types, so an overall mean would not compare like with like.
    for (method, query_type), group in per_query.groupby(["method", "query_type"]):
        summary.setdefault(method, {})[query_type] = {
            "n_queries": int(group["ap"].notna().sum()),
            "map": _round(group["ap"].mean()),
            "map_std": _round(group["ap"].std()),
            "ndcg": _round(group["ndcg"].mean()),
            "ndcg_std": _round(group["ndcg"].std()),
        }
    return {
        "k": k,
        "unrated_clips": unrated,
        "summary": summary,
        "per_query": [{key: _round(v) if isinstance(v, float) else v for key, v in row.items()} for row in per_query.to_dict("records")],
    }


def _round(value: float) -> float | None:
    """JSON has no NaN, so undefined values become null."""
    return None if pd.isna(value) else round(float(value), 4)
