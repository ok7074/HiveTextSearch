"""Offline search page and evaluation helper for the hive clips, built with Gradio.

Tabs:
  Search            type a query, pick its type and a method, see the top 10 clips; click a thumbnail to play it
  Rate (blind)      rate the pooled clips 0/1/2 one at a time, without seeing which method found them
  Failure analysis  fully rated queries only: worst queries first, every method's top 10 side by side with your
                    ratings, and a form that saves failure notes to experiments/failures.csv
Queries that still have unrated clips stay locked in Failure analysis, so your ratings stay blind.

Runs on CPU without internet: the models load from the local Hugging Face cache.
Reads data/runs/<split> (scripts/retrieval/retrieval.py) and experiments/ratings_<split>.csv (scripts/pooling/pooling.py).

Usage:
    python scripts/app/app.py
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

import gradio as gr
import pandas as pd
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.retrieval.retrieval import load_methods
from src.evaluation import (
    FAILURE_CATEGORIES, RATINGS_DIR, append_failure, fully_rated_queries, read_ratings, score, write_ratings,
)
from src.frame_selection import DATA_DIR, FRAMES_DIR
from src.retrieval import RUNS_DIR, SPLITS, TOP_K, top_k

CLIPS_DIR = DATA_DIR / "clips"
QUERY_TYPES = ["general", "count", "position", "motion"]
MARKS = {2: "✅2", 1: "🟡1", 0: "❌0"}
WHOLE_QUERY = "whole query"


def thumbnail(clip_id: str) -> str:
    return str(FRAMES_DIR / f"{clip_id}02.jpg")


def video(clip_id: str) -> str:
    return str(CLIPS_DIR / f"{clip_id}.mp4")


def ratings_path(split: str) -> Path:
    return RATINGS_DIR / f"ratings_{split}.csv"


def play(clip_ids: list[str], event: gr.SelectData) -> str:
    return video(clip_ids[event.index])


# Factories, so each button or gallery in a loop keeps its own value instead of the loop's last one.
def player_for(method: str):
    def handler(ids: dict, event: gr.SelectData) -> str:
        return video(ids[method][event.index])
    return handler


def rater(value: int):
    def handler(split: str, current: dict | None, skipped: list[int]):
        return rate(split, current, value, skipped)
    return handler


# --- Search ---------------------------------------------------------------------------------------------------

def search(query: str, query_type: str, method: str):
    if not query.strip():
        return [], [], "Type a query."
    start = time.perf_counter()
    result = METHODS[method]({"query": query.strip(), "query_type": query_type})
    if result is None:
        return [], [], f"**{method}** does not apply to {query_type} queries; pick another method or type."
    hits = top_k(result[0], CLIP_IDS, TOP_K, result[1])
    ms = (time.perf_counter() - start) * 1000
    gallery = [(thumbnail(c), f"#{rank} {c}  score {s:.3f}") for rank, (c, s) in enumerate(hits, 1)]
    return gallery, [c for c, _ in hits], f"{len(hits)} clips in {ms:.0f} ms"


# --- Rate (blind) ---------------------------------------------------------------------------------------------

def next_unrated(split: str, skipped: list[int]):
    """The first unrated row not skipped in this session; shows only the query and the clip."""
    path = ratings_path(split)
    if not path.exists():
        return f"No {path.name} yet: run `python scripts/pooling/pooling.py --split {split}` first.", None, None
    ratings = read_ratings(path)
    todo = [i for i in ratings.index[ratings["rating"].isna()] if i not in skipped]
    left = int(ratings["rating"].isna().sum())
    if left == 0:
        return f"All {len(ratings)} clips of {split} are rated.", None, None
    if not todo:
        return f"{left} left, all skipped in this session; switch split and back to see them again.", None, None
    row = ratings.loc[todo[0]]
    text = f"### {row['query']}\n*{row['query_type']} query* · {left} clips left to rate in {split}"
    return text, video(row["clip_id"]), {"row": int(todo[0]), "clip_id": row["clip_id"]}


def rate(split: str, current: dict | None, value: int, skipped: list[int]):
    if current is not None:
        path = ratings_path(split)
        ratings = read_ratings(path)
        # pooling only appends rows, so the row number still points at the same clip; check anyway.
        if ratings.loc[current["row"], "clip_id"] != current["clip_id"]:
            raise gr.Error("The ratings file changed underneath the app; reopen the tab.")
        ratings.loc[current["row"], "rating"] = value
        write_ratings(ratings, path)
    return next_unrated(split, skipped)


def skip(split: str, current: dict | None, skipped: list[int]):
    skipped = skipped + ([current["row"]] if current else [])
    return *next_unrated(split, skipped), skipped


# --- Failure analysis -----------------------------------------------------------------------------------------

def load_split(split: str):
    """Runs, ratings and the fully rated queries of a split, or a message why there is nothing to show."""
    if not (RUNS_DIR / split).exists():
        return None, None, set(), f"No runs for {split}: run scripts/retrieval/retrieval.py --split {split}."
    runs = load_from_disk(str(RUNS_DIR / split)).to_pandas()
    if not ratings_path(split).exists():
        return runs, None, set(), f"No ratings for {split} yet: pool and rate first."
    ratings = read_ratings(ratings_path(split))
    return runs, ratings, fully_rated_queries(runs, ratings), ""


def overview(split: str):
    runs, ratings, done, message = load_split(split)
    if ratings is None:
        return pd.DataFrame(), gr.Dropdown(choices=[], value=None), message
    unrated = runs[["query_type", "query", "clip_id"]].drop_duplicates().merge(ratings, how="left")
    unrated = unrated[unrated["rating"].isna()].groupby("query").size()
    locked = "\n".join(f"- 🔒 {q}: rate {n} more clips first" for q, n in unrated.items())

    per_query = pd.DataFrame(score(runs, ratings)["per_query"])
    per_query = per_query[[key in done for key in zip(per_query["query_type"], per_query["query"])]]
    if per_query.empty:
        return pd.DataFrame(), gr.Dropdown(choices=[], value=None), locked or "Nothing rated yet."
    table = per_query.pivot(index=["query_type", "query"], columns="method", values=["ap", "ndcg"])
    table.columns = [f"{method} {'AP' if metric == 'ap' else 'nDCG'}" for metric, method in table.columns]
    sort_by = "rrf AP" if "rrf AP" in table.columns else table.columns[0]
    table = table.sort_values(sort_by, na_position="first").reset_index()
    choices = [(f"[{t}] {q}", f"{t}|{q}") for t, q in zip(table["query_type"], table["query"])]
    return table.round(3), gr.Dropdown(choices=choices, value=choices[0][1]), locked


def side_by_side(split: str, key: str | None):
    """One gallery per method with your rating on every clip, plus each method's AP and nDCG."""
    empty = [[]] * len(METHOD_NAMES) + [""] * len(METHOD_NAMES)
    if not key:
        return *empty, {}, gr.Dropdown(choices=[WHOLE_QUERY], value=WHOLE_QUERY)
    query_type, query = key.split("|", 1)
    runs, ratings, done, _ = load_split(split)
    if (query_type, query) not in done:
        return *empty, {}, gr.Dropdown(choices=[WHOLE_QUERY], value=WHOLE_QUERY)
    runs = runs[(runs["query_type"] == query_type) & (runs["query"] == query)]
    rating_of = ratings[(ratings["query_type"] == query_type) & (ratings["query"] == query)].set_index("clip_id")["rating"]
    metrics = {m["method"]: m for m in score(runs, ratings)["per_query"]}

    galleries, headers, ids = [], [], {}
    for method in METHOD_NAMES:
        hits = runs[runs["method"] == method].sort_values("rank")
        ids[method] = hits["clip_id"].tolist()
        galleries.append([(thumbnail(c), f"#{r} {c}  {MARKS[int(rating_of[c])]}") for r, c in zip(hits["rank"], hits["clip_id"])])
        m = metrics.get(method)
        headers.append(f"**{method}**  AP {m['ap']} · nDCG {m['ndcg']}" if m else f"**{method}**: not run for this query")
    clips = [WHOLE_QUERY] + sorted(runs["clip_id"].unique())
    return *galleries, *headers, ids, gr.Dropdown(choices=clips, value=WHOLE_QUERY)


def save_failure(split: str, key: str | None, clip: str, method: str, category: str, note: str) -> str:
    if not key:
        return "Pick a query first."
    query_type, query = key.split("|", 1)
    append_failure({
        "split": split, "query_type": query_type, "query": query, "clip_id": "" if clip == WHOLE_QUERY else clip,
        "method": method, "category": category, "note": note.strip(),
    })
    return f"Saved to experiments/failures.csv: {category} · {query} · {clip}"


# --- Layout ---------------------------------------------------------------------------------------------------

def build() -> gr.Blocks:
    with gr.Blocks(title="Hive clip search") as demo:
        gr.Markdown("# Hive clip search")
        with gr.Tab("Search"):
            with gr.Row():
                query = gr.Textbox(label="Query", placeholder="e.g. Bees sitting still on the board", scale=4)
                query_type = gr.Dropdown(QUERY_TYPES, value="general", label="Query type")
                method = gr.Dropdown(METHOD_NAMES, value="rrf", label="Method")
            go = gr.Button("Search", variant="primary")
            status = gr.Markdown()
            with gr.Row():
                results = gr.Gallery(label="Top 10 (click to play)", columns=5, height=520, scale=3)
                player = gr.Video(label="Clip", autoplay=True, loop=True, scale=2)
            found = gr.State([])
            for trigger in (go.click, query.submit):
                trigger(search, [query, query_type, method], [results, found, status])
            results.select(play, found, player)

        with gr.Tab("Rate (blind)"):
            gr.Markdown("0 = not relevant · 1 = partly · 2 = clearly. Saved after every click.")
            rate_split = gr.Radio(list(SPLITS), value="train", label="Split")
            prompt = gr.Markdown()
            clip_video = gr.Video(autoplay=True, loop=True, height=480)
            current, skipped = gr.State(None), gr.State([])
            with gr.Row():
                buttons = [gr.Button(f"{v}", variant="primary") for v in (0, 1, 2)]
                skip_button = gr.Button("Skip")
            outputs = [prompt, clip_video, current]
            rate_split.change(lambda s: (*next_unrated(s, []), []), rate_split, [*outputs, skipped])
            for value, button in zip((0, 1, 2), buttons):
                button.click(rater(value), [rate_split, current, skipped], outputs)
            skip_button.click(skip, [rate_split, current, skipped], [*outputs, skipped])
            demo.load(lambda s: next_unrated(s, []), rate_split, outputs)

        with gr.Tab("Failure analysis"):
            fa_split = gr.Radio(list(SPLITS), value="train", label="Split")
            refresh = gr.Button("Refresh after rating")
            worst = gr.Dataframe(label="Fully rated queries, worst rrf AP first", interactive=False)
            locked = gr.Markdown()
            picked = gr.Dropdown(label="Query", choices=[])
            headers, galleries = [], []
            with gr.Row():
                for name in METHOD_NAMES:
                    with gr.Column(min_width=160):
                        headers.append(gr.Markdown(f"**{name}**"))
                        galleries.append(gr.Gallery(columns=2, height=600, show_label=False))
            fa_player = gr.Video(label="Clip", autoplay=True, loop=True, height=420)
            method_ids = gr.State({})
            with gr.Row():
                fail_clip = gr.Dropdown([WHOLE_QUERY], value=WHOLE_QUERY, label="Clip")
                fail_method = gr.Dropdown(["all"] + METHOD_NAMES, value="rrf", label="Method")
                fail_category = gr.Dropdown(list(FAILURE_CATEGORIES), value=FAILURE_CATEGORIES[0], label="Category")
            fail_note = gr.Textbox(label="Why did it fail?", lines=2)
            save = gr.Button("Save failure note")
            saved = gr.Markdown()

            for trigger in (fa_split.change, refresh.click, demo.load):
                trigger(overview, fa_split, [worst, picked, locked])
            picked.change(side_by_side, [fa_split, picked], [*galleries, *headers, method_ids, fail_clip])
            for name, gallery in zip(METHOD_NAMES, galleries):
                gallery.select(player_for(name), method_ids, fa_player)
            save.click(save_failure, [fa_split, picked, fail_clip, fail_method, fail_category, fail_note], saved)
    return demo


CLIP_IDS, METHODS = load_methods()
METHOD_NAMES = list(METHODS)

if __name__ == "__main__":
    build().launch(allowed_paths=[str(DATA_DIR)])
