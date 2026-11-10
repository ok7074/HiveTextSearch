import pandas as pd
import pytest

from src.evaluation import average_precision, ndcg, pool, read_ratings, score

RUNS = pd.DataFrame([
    {"query_type": "count", "query": "many bees", "method": m, "rank": r, "clip_id": c}
    for m, clips in {"clip": ["a", "b"], "siglip2": ["b", "c"]}.items()
    for r, c in enumerate(clips, 1)
])


def test_pool_removes_duplicates_and_hides_method_and_rank():
    pooled = pool(RUNS)
    assert sorted(pooled["clip_id"]) == ["a", "b", "c"]
    assert list(pooled.columns) == ["query_type", "query", "clip_id", "rating"]


def test_pool_keeps_existing_ratings_and_appends_only_new_clips(tmp_path):
    path = tmp_path / "ratings.csv"
    pool(RUNS).assign(rating=[2, 0, 1]).to_csv(path, index=False)
    existing = read_ratings(path)
    more = pd.concat([RUNS, RUNS.iloc[[0]].assign(method="detector", clip_id="d")])
    pooled = pool(more, existing)
    assert pooled.iloc[:3].equals(existing)
    assert pooled.iloc[3]["clip_id"] == "d" and pd.isna(pooled.iloc[3]["rating"])


def test_read_ratings_rejects_values_other_than_0_1_2(tmp_path):
    path = tmp_path / "ratings.csv"
    pool(RUNS).assign(rating=[2, 3, None]).to_csv(path, index=False)
    with pytest.raises(ValueError):
        read_ratings(path)


def test_read_ratings_accepts_semicolons_from_excel(tmp_path):
    path = tmp_path / "ratings.csv"
    pool(RUNS).assign(rating=[2, 0, None]).to_csv(path, index=False, sep=";")
    assert read_ratings(path)["rating"].tolist()[:2] == [2, 0]


def test_average_precision_and_ndcg_by_hand():
    assert average_precision([2, 0, 1], n_relevant=2) == pytest.approx((1 + 2 / 3) / 2)
    assert average_precision([0, 0], n_relevant=0) != average_precision([0, 0], n_relevant=0)  # NaN
    assert ndcg([2, 1], [2, 1]) == pytest.approx(1)
    assert ndcg([1, 2], [2, 1]) < 1


def test_score_uses_the_whole_pool_for_the_ideal():
    ratings = pd.DataFrame({"query_type": "count", "query": "many bees", "clip_id": ["a", "b", "c"], "rating": [0, 1, 2]})
    summary = score(RUNS, ratings, k=2)["summary"]
    assert summary["clip"]["count"]["map"] == pytest.approx(0.25)  # b relevant at rank 2 of 2 relevant: (1/2) / 2
    assert summary["siglip2"]["count"]["map"] == pytest.approx(1.0)
    assert summary["siglip2"]["all"]["ndcg"] < 1  # b (1) before c (2)
