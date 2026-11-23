"""Left/right test: can CLIP, SigLIP 2 and the detector tell left from right?

Queries: every query naming "left" or "right" (side queries) and, as a negative control, every query about the
middle of the board. q' is the query with left and right swapped (q' = q for the middle controls).
Methods: clip, siglip2 and detector (the positive control: it reads sides from box positions).

  Test A (side queries)  q vs q' on the normal clips: Spearman rho of the two score vectors and how many clips the two
                         top 10s share. Ignoring sides gives rho ~ 1 and overlap ~ 10; the detector should be near 0.
  Test B (all)           q on the normal vs the mirrored clips (data/embeddings/{clip,siglip2}_flipped.npz from
                         create_embeddings.py --flip; the detector's boxes are mirrored in code). Same two measures.
                         The detector must stay near 1 / 10 on the middle controls. CLIP and SigLIP 2 do not: the
                         mirror alone reshuffles their lists (~2/10 kept on middle queries, rho ~0.73), so for them
                         test B measures mirror sensitivity and test A is the left/right measure (see NOTE).
  AP vs AP_flipped       optional, needs --pool and ratings (not run: test A already answers the question).
                         A mirrored clip shows bees on the left exactly when the original shows them on the right,
                         so the flipped top 10 for q is judged with ratings of the original clips for q'.
                         Understands sides: AP_flipped ~ AP. Ignores sides: AP_flipped drops.

--pool writes experiments/ratings_flip_<split>.csv (query column = q', the query being judged) with every clip of the
flipped top 10s, copying ratings that already exist in ratings_<split>.csv. Rate the empty cells 0/1/2 in the app
(Rate tab, split "flip_<split>") or in the CSV. These ratings never enter the main scores.

Without --pool it writes results/flip_test_<split>.json, one row per query and method:
  rho_a, overlap_a, rho_b, overlap_b, ap, ap_flipped (unrated clips count as 0; their number is printed and saved)
and prints the means per method, side queries and middle controls apart. Runs on CPU.

Usage:
    python scripts/flip_test/flip_test.py --split train --pool
    python scripts/flip_test/flip_test.py --split train
"""
import argparse
import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.retrieval.retrieval import TEXT_MODELS, load_clip_embeddings
from src import track_run
from src.detection import detector_scores, load_detections
from src.embeddings import EMBEDDINGS_DIR
from src.evaluation import KEY, RATINGS_DIR, RESULTS_DIR, pool, read_ratings, score, write_ratings
from src.flip_test import is_middle_query, mirror_detections, mirror_query, overlap, rank_correlation
from src.retrieval import QUERIES_FILE, RUNS_DIR, SPLITS, TOP_K, load_queries, text_scores, top_k

METHODS = ("clip", "siglip2", "detector")
# The detector's test A should be near 0 shared clips and its test B on middle queries near 10; otherwise suspect a bug.
DETECTOR_MAX_OVERLAP_A, CONTROL_MIN_OVERLAP_B = 3, 7
NOTE = (
    "Test A (swap left/right in the query) is the left/right measure. For clip and siglip2, test B (mirrored frames) "
    "mostly measures mirror sensitivity: their lists reshuffle even on middle queries, which a mirror leaves in place. "
    "Checked: the Kaggle flipped embeddings match a local recompute within 1e-6. AP_flipped was not rated."
)


def flip_queries(split: str) -> list[dict]:
    """Side queries with their mirror, then the middle controls (mirrored = the query itself)."""
    items = []
    for q in load_queries(QUERIES_FILE, split):
        if (mirrored := mirror_query(q["query"])) is not None:
            items.append({**q, "mirrored": mirrored, "control": False})
        elif is_middle_query(q["query"]):
            items.append({**q, "mirrored": q["query"], "control": True})
    return items


def load_scorers() -> tuple[np.ndarray, dict, bool]:
    """
    scorers[method][variant] maps (query, query_type) to (scores, descending) or None; variant is "original" or
    "flipped". The flipped variant is missing when the *_flipped.npz files are not there yet.
    """
    clip_ids, scorers, have_flipped = None, {}, True
    for name, (load_model, text_features) in TEXT_MODELS.items():
        ids, embeddings = load_clip_embeddings(name)
        clip_ids = ids if clip_ids is None else clip_ids
        model, processor = load_model(device_map=None)
        text = lambda texts, m=model, p=processor, f=text_features: f(m, p, texts)
        variants = {"original": embeddings}
        flipped_path = EMBEDDINGS_DIR / f"{name}_flipped.npz"
        if flipped_path.exists():
            with np.load(flipped_path, allow_pickle=False) as saved:
                if not np.array_equal(saved["clip_ids"], clip_ids):
                    raise ValueError(f"{flipped_path} lists different clips than {name}.npz")
                variants["flipped"] = saved["clip_embeddings"]
        else:
            have_flipped = False
        scorers[name] = {
            variant: (lambda query, query_type, e=emb, t=text: (text_scores(query, e, t), True))
            for variant, emb in variants.items()
        }
    detections = load_detections(clip_ids)
    scorers["detector"] = {
        variant: (lambda query, query_type, d=d: detector_scores(query, query_type, d))
        for variant, d in (("original", detections), ("flipped", mirror_detections(detections)))
    }
    if not have_flipped:
        print("No *_flipped.npz in data/embeddings yet: test B runs for the detector only; no AP_flipped.", flush=True)
        for name in TEXT_MODELS:
            scorers[name].pop("flipped", None)
    return clip_ids, scorers, have_flipped


def run_tests(items: list[dict], clip_ids: np.ndarray, scorers: dict) -> tuple[list[dict], list[dict]]:
    """Test A and B per query and method, plus the flipped top 10s as run rows judged under q'."""
    rows, flipped_runs = [], []
    for item in items:
        q, q_type = item["query"], item["query_type"]
        for method in METHODS:
            if (original := scorers[method]["original"](q, q_type)) is None:
                continue
            row = {key: item[key] for key in ("query_type", "query", "mirrored", "control")} | {"method": method}
            top = top_k(original[0], clip_ids, TOP_K, original[1])
            if not item["control"]:
                mirrored = scorers[method]["original"](item["mirrored"], q_type)
                row["rho_a"] = rank_correlation(original[0], mirrored[0], original[1], mirrored[1])
                row["overlap_a"] = overlap(top, top_k(mirrored[0], clip_ids, TOP_K, mirrored[1]))
            if "flipped" in scorers[method]:
                flipped = scorers[method]["flipped"](q, q_type)
                flipped_top = top_k(flipped[0], clip_ids, TOP_K, flipped[1])
                row["rho_b"] = rank_correlation(original[0], flipped[0], original[1], flipped[1])
                row["overlap_b"] = overlap(top, flipped_top)
                flipped_runs += [
                    {"query_type": q_type, "query": item["mirrored"], "method": method, "rank": rank, "clip_id": clip, "score": s}
                    for rank, (clip, s) in enumerate(flipped_top, 1)
                ]
            rows.append(row)
        print(f"[{q_type}] {q}", flush=True)
    return rows, flipped_runs


def pool_ratings(split: str, flipped_runs: pd.DataFrame) -> None:
    """Add the flipped top 10s to the flip rating sheet and copy every rating ratings_<split>.csv already has."""
    path = RATINGS_DIR / f"ratings_flip_{split}.csv"
    existing = read_ratings(path) if path.exists() else None
    sheet = pool(flipped_runs, existing)
    main = read_ratings(RATINGS_DIR / f"ratings_{split}.csv").rename(columns={"rating": "main_rating"})
    sheet = sheet.merge(main[KEY + ["main_rating"]], on=KEY, how="left")
    sheet["rating"] = sheet["rating"].fillna(sheet.pop("main_rating"))
    write_ratings(sheet, path)
    print(f"{path}: {len(sheet)} clips, {int(sheet['rating'].isna().sum())} still to rate")


def add_ap(split: str, rows: pd.DataFrame, flipped_runs: pd.DataFrame) -> tuple[pd.DataFrame, int | None]:
    """
    Adds ap (normal top 10, main ratings) and ap_flipped (flipped top 10 judged under q', main + flip ratings) with
    the main scoring's rule and denominator. Also returns the number of unrated flipped clips (counted as 0).
    """
    main = read_ratings(RATINGS_DIR / f"ratings_{split}.csv")
    runs = load_from_disk(str(RUNS_DIR / split)).to_pandas()
    runs = runs[runs["method"].isin(METHODS) & runs["query"].isin(rows["query"])]
    ap = pd.DataFrame(score(runs, main)["per_query"])[["query", "method", "ap"]]
    rows = rows.merge(ap, on=["query", "method"], how="left")

    flip_path = RATINGS_DIR / f"ratings_flip_{split}.csv"
    if flipped_runs.empty:
        return rows, None
    if not flip_path.exists():
        print(f"No {flip_path.name} yet: run with --pool and rate it first; AP_flipped skipped.")
        return rows, None
    ratings = pd.concat([main, read_ratings(flip_path)]).drop_duplicates(KEY)
    flipped = score(flipped_runs, ratings)
    ap_flipped = pd.DataFrame(flipped["per_query"]).rename(columns={"query": "mirrored", "ap": "ap_flipped"})
    return rows.merge(ap_flipped[["mirrored", "method", "ap_flipped"]], on=["mirrored", "method"], how="left"), flipped["unrated_clips"]


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in ("overlap_a", "overlap_b", "ap", "ap_flipped") if c in rows]
    rows = rows.assign(queries=np.where(rows["control"], "middle", "side"))
    return rows.groupby(["queries", "method"])[columns].mean().round(3)


def sanity_check(rows: pd.DataFrame) -> None:
    """
    Warn when the detector, the method whose answers are known, misbehaves; that points at a bug in the mirroring.
    For CLIP and SigLIP 2 the middle-query test B overlap is printed as their mirror sensitivity.
    """
    side, middle = rows[~rows["control"]], rows[rows["control"]]
    detector_a = side.loc[side["method"] == "detector", "overlap_a"].mean()
    if detector_a > DETECTOR_MAX_OVERLAP_A:
        print(f"WARNING: detector test A overlap is {detector_a:.1f}/10 on side queries, expected near 0. Stop and check.")
    if "overlap_b" not in middle or middle.empty:
        return
    by_method = middle.groupby("method")["overlap_b"].mean()
    if by_method.get("detector", 10) < CONTROL_MIN_OVERLAP_B:
        print(f"WARNING: detector test B overlap on middle queries is {by_method['detector']:.1f}/10, expected near 10.")
    others = by_method.drop("detector", errors="ignore").dropna()
    if not others.empty:
        print("Mirror sensitivity (clips kept on middle queries after mirroring, of 10): "
              + ", ".join(f"{method} {value:.1f}" for method, value in others.items()))


@track_run(run_name="flip_test")
def main(split: str, pool_only: bool) -> None:
    items = flip_queries(split)
    clip_ids, scorers, have_flipped = load_scorers()
    rows, flipped_runs = run_tests(items, clip_ids, scorers)
    rows, flipped_runs = pd.DataFrame(rows), pd.DataFrame(flipped_runs)
    mlflow.log_params({"n_side_queries": int((~rows.drop_duplicates("query")["control"]).sum()), "have_flipped": have_flipped})

    if pool_only:
        if not have_flipped:
            raise FileNotFoundError("Pooling needs the flipped embeddings: run create_embeddings.py --flip first")
        pool_ratings(split, flipped_runs)
        return

    unrated = None
    if have_flipped:
        rows, unrated = add_ap(split, rows, flipped_runs)
    else:
        rows = add_ap(split, rows, flipped_runs.iloc[0:0])[0]  # AP only; flipped runs are incomplete
    sanity_check(rows)
    summary = summarise(rows)
    print(summary.to_string())
    for (queries, method), values in summary.iterrows():
        mlflow.log_metrics({f"{method}_{queries}_{name}": float(v) for name, v in values.items() if pd.notna(v)})

    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"flip_test_{split}.json"
    records = rows.round(4).astype(object).where(rows.notna(), None).to_dict("records")
    path.write_text(json.dumps({"split": split, "k": TOP_K, "note": NOTE, "unrated_clips": unrated, "rows": records}, indent=2), encoding="utf-8")
    print(f"Unrated flipped clips counted as 0: {unrated}. Saved {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--pool", action="store_true", help="write the flip rating sheet instead of scoring")
    args = parser.parse_args()
    main(split=args.split, pool_only=args.pool)
