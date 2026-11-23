"""Experiment: how RRF and weighted RRF scores change as the method weights change, on the existing ratings.

Sweeps, one weight changed at a time, the others at today's values (src/retrieval.RRF_WEIGHTS):
  A  all methods: clip 0-0.7, siglip2 0.55-1.45, detector 0.8-4.0 (keeps detector > siglip2 > clip)
  B  no CLIP:     siglip2, motion, detector each 0.25-3.0
References: plain RRF (all weights 1) with and without CLIP.

Clips never rated for a query are skipped: each fused ranking keeps only the rated clips (condensed lists,
Sakai 2007), so a weighting is not punished for finding clips nobody judged. unrated_in_top10 says how many
unrated clips the full top 10 would have had.

Output per split in results/rrf_sweep/<split>/: per_query.csv, per_type.csv, total.csv and one plot per
swept method (sweep_A_<method>.png, sweep_B_<method>.png).
Sweeping on test and picking the best is tuning on test: use the test plots to understand, choose on train.

To undo: delete scripts/rrf_sweep/ and results/rrf_sweep/ (nothing else uses them).

Usage:
    python scripts/rrf_sweep/rrf_sweep.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.retrieval.retrieval import load_methods
from src.evaluation import RATINGS_DIR, average_precision, ndcg, read_ratings
from src.retrieval import QUERIES_FILE, RRF_WEIGHTS, SPLITS, TOP_K, load_queries, rrf, top_k

OUT_DIR = Path(__file__).parent.parent.parent / "results" / "rrf_sweep"
SINGLE_TYPES = ["count", "position", "motion", "general"]
SWEEPS = {
    "A": {"methods": ["clip", "siglip2", "motion", "detector"],
          "vary": {"clip": np.arange(0, 0.71, 0.1), "siglip2": np.arange(0.55, 1.46, 0.1),
                   "detector": np.arange(0.8, 4.01, 0.2)}},
    "B": {"methods": ["siglip2", "motion", "detector"],
          "vary": {m: np.arange(0.25, 3.01, 0.25) for m in ("siglip2", "motion", "detector")}},
}


def evaluate(results: dict, weights: dict, pool: pd.Series, clip_ids: np.ndarray) -> tuple[float, float, int]:
    """AP@10, nDCG@10 and unrated clips in the full top 10 for one query and one weighting."""
    names = [n for n in weights if n in results]
    fused = rrf([results[n] for n in names], weights=[weights[n] for n in names])
    unrated = sum(c not in pool.index for c, _ in top_k(fused, clip_ids, TOP_K, True))
    rated = np.where(np.isin(clip_ids, pool.index), fused, np.nan)
    gains = [int(pool[c]) for c, _ in top_k(rated, clip_ids, TOP_K, True)]
    n_relevant = min(int((pool >= 1).sum()), TOP_K)
    return average_precision(gains, n_relevant), ndcg(gains, sorted(pool, reverse=True)[:TOP_K]), unrated


def run_split(split: str, clip_ids: np.ndarray, singles: dict) -> pd.DataFrame:
    ratings = read_ratings(RATINGS_DIR / f"ratings_{split}.csv")
    rows = []
    for q in load_queries(QUERIES_FILE, split):
        results = {n: r for n, m in singles.items() if (r := m(q)) is not None}  # each method scored once
        pool = ratings[(ratings["query_type"] == q["query_type"]) & (ratings["query"] == q["query"])]
        pool = pool.set_index("clip_id")["rating"].astype(int)
        configs = [("reference", "plain RRF", "-", 1.0, dict.fromkeys(SWEEPS["A"]["methods"], 1.0)),
                   ("reference", "plain RRF, no CLIP", "-", 1.0, dict.fromkeys(SWEEPS["B"]["methods"], 1.0)),
                   ("reference", "today's weights", "-", 1.0, dict(RRF_WEIGHTS))]
        for sweep, spec in SWEEPS.items():
            for method, values in spec["vary"].items():
                for w in values:
                    weights = {m: RRF_WEIGHTS[m] for m in spec["methods"]} | {method: round(float(w), 2)}
                    configs.append((sweep, f"sweep {sweep}", method, round(float(w), 2), weights))
        for sweep, label, method, weight, weights in configs:
            ap, nd, unrated = evaluate(results, weights, pool, clip_ids)
            rows.append({"sweep": sweep, "label": label, "varied": method, "weight": weight, **q,
                         "ap": ap, "ndcg": nd, "unrated_in_top10": unrated})
    return pd.DataFrame(rows)


def check_against_scoring(per_query: pd.DataFrame, split: str) -> None:
    """Today's weights and plain RRF must give exactly the numbers scoring.py saved (their top 10s are all rated)."""
    saved = pd.DataFrame(json.loads((OUT_DIR.parent / f"{split}_metrics.json").read_text())["per_query"])
    for label, method in (("today's weights", "rrf_weighted"), ("plain RRF", "rrf")):
        ours = per_query[per_query["label"] == label].set_index(["query_type", "query"])
        theirs = saved[saved["method"] == method].set_index(["query_type", "query"]).loc[ours.index]
        assert ours["unrated_in_top10"].sum() == 0, f"{label}: unrated clips in a top 10 that should be fully rated"
        assert np.allclose(ours["ap"], theirs["ap"], atol=1e-4, equal_nan=True), f"{label}: AP differs from scoring.py"
        assert np.allclose(ours["ndcg"], theirs["ndcg"], atol=1e-4, equal_nan=True), f"{label}: nDCG differs"


def plot(per_type: pd.DataFrame, total: pd.DataFrame, sweep: str, method: str, out: Path, split: str) -> None:
    lines = total[(total["sweep"] == sweep) & (total["varied"] == method)]
    by_type = per_type[(per_type["sweep"] == sweep) & (per_type["varied"] == method)]
    reference = total[total["label"] == ("plain RRF" if sweep == "A" else "plain RRF, no CLIP")].iloc[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, metric, name in zip(axes, ("ap", "ndcg"), ("mAP@10", "nDCG@10")):
        ax.plot(lines["weight"], lines[metric], color="black", linewidth=2.5, marker="o", label="all queries")
        for query_type in SINGLE_TYPES:
            part = by_type[by_type["query_type"] == query_type]
            if len(part):
                ax.plot(part["weight"], part[metric], marker=".", linewidth=1.2, label=f"{query_type} only")
        ax.axvline(RRF_WEIGHTS[method], color="grey", linestyle="--", linewidth=1, label=f"today's weight ({RRF_WEIGHTS[method]})")
        ax.axhline(reference[metric], color="grey", linestyle=":", linewidth=1.2, label=f"{reference['label']} (all queries)")
        ax.set(title=name, xlabel=f"{method} weight", ylim=(0, 1.05))
        ax.grid(alpha=0.3)
    axes[1].legend(fontsize=8, loc="lower right")
    others = ", ".join(f"{m} {RRF_WEIGHTS[m]}" for m in SWEEPS[sweep]["methods"] if m != method)
    fig.suptitle(f"{split}: weighted RRF {'(all methods)' if sweep == 'A' else 'without CLIP'}, "
                 f"varying {method} (others: {others})", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / f"sweep_{sweep}_{method}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    clip_ids, methods = load_methods()
    singles = {n: m for n, m in methods.items() if n not in ("rrf", "rrf_weighted")}
    for split in SPLITS:
        out = OUT_DIR / split
        out.mkdir(parents=True, exist_ok=True)
        per_query = run_split(split, clip_ids, singles)
        check_against_scoring(per_query, split)
        keys = ["sweep", "label", "varied", "weight"]
        per_type = per_query.groupby([*keys, "query_type"], sort=False)[["ap", "ndcg", "unrated_in_top10"]].mean().reset_index()
        total = per_query.groupby(keys, sort=False).agg(
            ap=("ap", "mean"), ndcg=("ndcg", "mean"), unrated_in_top10=("unrated_in_top10", "sum"),
            n_queries=("query", "size")).reset_index()
        per_query.to_csv(out / "per_query.csv", index=False)
        per_type.round(4).to_csv(out / "per_type.csv", index=False)
        total.round(4).to_csv(out / "total.csv", index=False)
        for sweep, spec in SWEEPS.items():
            for method in spec["vary"]:
                plot(per_type, total, sweep, method, out, split)

        print(f"\n== {split} ({per_query['query'].nunique()} queries): matches scoring.py at today's weights")
        print(total[total["sweep"] == "reference"][["label", "ap", "ndcg"]].round(3).to_string(index=False))
        for (sweep, method), part in total[total["sweep"] != "reference"].groupby(["sweep", "varied"], sort=False):
            best = part.loc[part["ap"].idxmax()]
            print(f"sweep {sweep} {method:9} best mAP {best['ap']:.3f} (nDCG {best['ndcg']:.3f}) at {best['weight']}; "
                  f"range {part['ap'].min():.3f}-{part['ap'].max():.3f}; unrated clips skipped: "
                  f"{int(part['unrated_in_top10'].min())}-{int(part['unrated_in_top10'].max())} per weighting")
        print(f"Saved to {out}")


if __name__ == "__main__":
    main()
