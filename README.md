# HiveTextSearch

Text search over 2-second clips of honeybee hive entrance videos. Given a natural-language query ("many bees", "bees on the left", "busy entrance"), the system ranks ~1,440 short clips by relevance using vision-language models and a bee detector.

Presentation deadline: **25 November 2026** — 15 minutes + Q&A, at most 3 slides.

---

## What it does

- Cuts 96 unlabeled hive entrance videos (30 s, 1080×1920 @ 60 fps) into 2-second clips (~1,440 total)
- Samples 4 evenly-spaced frames per clip and computes one embedding per clip (average of its 4 frames)
- At query time, embeds the text query and ranks all clips by cosine similarity
- Compares retrieval methods:
  | Method | Query types | Description | Status |
  |--------|-------------|-------------|--------|
  | CLIP | all | OpenAI `clip-vit-base-patch32`, zero-shot | done |
  | SigLIP 2 | all | Google `siglip2-base-patch16-224`, zero-shot | done |
  | Motion | motion | Frame-difference score over the 4 frames (baseline) | done |
  | Detector | count, position, general, motion | YOLO26s @ 960 fine-tuned on bee data (classes bee, bee_flying); bee counts per vertical third, flying bees for motion | done |
  | Combined (RRF) | all | Reciprocal rank fusion (k = 60) of every method that applies to the query | done |
  | Combined (tuned RRF) | all | RRF with weights clip 0.5, SigLIP 2 0.75, motion 1, detector 1.5 (chosen on train queries) | done |
- Evaluates with **mAP@10** and **nDCG@10** on hand-annotated relevance labels (0 / 1 / 2) across four query types (count, position, motion, general description) and the six pairs of them
- Reports latency at p50 and p99 over 200 queries per method

No vector database or cloud service needed — at ~1,440 clips, a NumPy matrix multiply searches everything in milliseconds.

---

## Repository layout

Reusable functions live in `src/<name>.py`; each runnable step is `scripts/<name>/<name>.py`.

```
HiveTextSearch/
├── src/
│   ├── __init__.py            # activates MLflow autologging on import
│   ├── tracking.py            # mlflow.autolog() + @track_run decorator
│   ├── video_format.py, clips_and_frames.py, frame_selection.py   # data preparation
│   ├── embeddings.py          # embed the 4 frames per clip, average, save .npz
│   ├── optuna_tuning.py       # Optuna search for YOLO hyperparameters
│   ├── retrieval.py           # query file parsing, cosine scores, motion score, top k
│   ├── detection.py           # YOLO counts per third, clip labels, detector ranking
│   ├── evaluation.py          # blind pooling, rating file checks, mAP / nDCG
│   └── models/                # clip_model.py, siglip2_model.py, yolo_model.py
├── scripts/
│   ├── data_prep/             # videos -> 2 s clips -> 4 frames per clip
│   ├── clip_siglip_embeddings/# CLIP and SigLIP 2 clip embeddings
│   ├── yolo/                  # split bee data by hive, train YOLO, detect bees in all clips
│   ├── retrieval/             # run every method over the train or test queries
│   ├── pooling/               # build the blind rating sheet
│   ├── scoring/               # mAP@10 / nDCG@10 from the ratings
│   └── latency/               # p50 / p99 per method
├── tests/                     # pytest
├── queries.txt                # Train:/Test: then Count:/Position:/Motion:/General: headings
├── experiments/               # YOLO runs, ratings_<split>.csv (git-ignored outputs)
├── results/                   # metrics and latency JSON
├── data/                      # dataset, frames, embeddings, detections, runs (git-ignored)
└── README.md
```

---

## Setup

**Requirements:** Python 3.11+, a GPU is helpful for indexing but not required for search.

```bash
# 1. Activate the virtual environment
HiveTextSearch\Scripts\Activate.ps1        # Windows PowerShell
# source HiveTextSearch/bin/activate       # macOS / Linux

# 2. Install the exact versions this project was run with
pip install -r requirements.txt
```

The model code needs transformers 5 (`get_image_features(...).pooler_output`).

**Dataset** — download once from Hugging Face (gated, requires an access token):

```python
from datasets import load_dataset
ds = load_dataset("mrkschtr/thehiveisreal", token="<your_hf_token>")
ds.save_to_disk("data/thehiveisreal")
```

---

## Usage

Full pipeline, after data preparation and embeddings:

```bash
python scripts/yolo/detect_clips.py                   # YOLO on all clips -> data/detections/
python scripts/retrieval/retrieval.py --split train   # top 10 per method -> data/runs/train/
python scripts/pooling/pooling.py                     # blind sheet -> experiments/ratings_train.csv
# fill the rating column with 0 / 1 / 2
python scripts/scoring/scoring.py --split train       # -> results/train_metrics.json
# repeat retrieval, pooling and rating with --split test, then:
python scripts/scoring/scoring.py --split test        # -> results/test_metrics.json
python scripts/latency/latency.py                     # -> results/latency.json
```

Search from Python:

```python
import numpy as np
from src.models import load_clip, clip_text_features

model, processor = load_clip()
embeddings = np.load("data/embeddings/clip.npz")

def search(query: str, top_k: int = 10):
    query_embedding = clip_text_features(model, processor, [query]).cpu().numpy()[0]
    scores = embeddings["clip_embeddings"] @ query_embedding  # cosine similarity: all vectors have length 1
    best = np.argsort(-scores)[:top_k]
    return [(str(embeddings["clip_ids"][i]), float(scores[i])) for i in best]
```

For SigLIP 2, use `load_siglip2`, `siglip2_text_features` and `data/embeddings/siglip2.npz`.

View MLflow experiment runs:
```bash
mlflow ui          # open http://127.0.0.1:5000
```

---

## Methods

### CLIP and SigLIP 2
Both models embed images and text into the same vector space. Cosine similarity between the query embedding and each clip embedding gives a relevance score.

- CLIP is trained with a **softmax** contrastive loss: each image is pushed towards its own caption and away from every other caption in the batch
- SigLIP 2 is trained with a **sigmoid** loss: each image–text pair is scored on its own as a match or not, with text formatted as `"This is a photo of {label}."`
- For a fair comparison both models get the same prompt, `"This is a photo of {query}."`, and the same 224×224 center crop
- For search, both rank clips by cosine similarity between embeddings; the loss only shapes training

### Bee detector
YOLO26s (960 px) fine-tuned on the [Sledevič et al. (2025)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11856287/) public bee dataset plus hand-labelled frames from our own videos, with two classes (`bee`, `bee_flying`), split by hive so test hives are never seen in training (9 Optuna trials, then a final run, weights `final_s960`). Held-out public test hives: mAP50-95 0.560, mAP50 0.847, precision 0.927, recall 0.769. Own test frames: mAP50-95 0.438, mAP50 0.785, precision 0.858, recall 0.726.

Per clip, every box of the 4 frames is stored with its position and class, plus the boxes per vertical third (left / center / right). The clip's label is the busiest third, `overall` on a tie, `empty` with no bees. Each video gets an entrance height and a board middle from where its sitting bees crowd (they sit on the landing board just below the entrance). At query time:
- a count or position query naming left, center (also "middle", "centre") or right keeps only clips with that label, ranked by the bees in that third
- a quiet query naming a third ("left side empty") keeps clips with bees elsewhere, fewest in that third first
- a position query naming up or down ("above the entrance", "lower edge of the board") ranks clips by the sitting bees above the entrance / on the lower part of the board
- a position query about spread ranks "evenly spread" by how equal the bee counts are over 6 columns of the board, and "two separate groups" by two busy column runs with a near-empty gap between them
- a position-motion query naming a place counts only that place: flying bees for busy words, sitting bees for quiet words ("resting in the middle")
- count and general queries without a third rank all clips by mean bees per frame; quiet words ("few", "no", "empty", …) put the fewest first
- motion queries rank all clips by mean `bee_flying` boxes per frame (fewest first for quiet words such as "still")

Limitations: the detector was trained on landscape images, while the clips are portrait. It also mistakes the blurry mesh above some entrances for bees, the same spots in every clip of a hive, which inflates "above the entrance" counts (kept on purpose, so the ratings show it). In two videos (rec2125_seq016, rec2125_seq017) these mesh boxes push the estimated entrance off the top of the frame, so they never count as "above".

### Motion baseline
Frame-difference score: mean pixel change between consecutive frames of a clip. Busiest first, or stillest first for quiet words ("calm", "still", …). Used as the baseline and fallback for motion queries.

### Combined score (RRF)
Reciprocal rank fusion (Cormack et al., 2009): each clip scores the sum over methods of `weight / (60 + rank)`, using every method that applies to the query. `rrf` uses equal weights; `rrf_tuned` uses clip 0.5, SigLIP 2 0.75, motion 1, detector 1.5, chosen on the train queries and frozen before test. `rrf_tuned_no_clip` is `rrf_tuned` without CLIP, same weights; it was added after the first test scoring, once a weight sweep showed CLIP adds little, so its test numbers are not as blind as the others'. `rrf_siglip_boost` is `rrf_tuned_no_clip` with SigLIP 2 raised from 0.75 to 1, also chosen after test scoring.

---

## Evaluation

- **Query types:** count ("many bees"), position ("bees on the left"), motion ("busy entrance"), general description, plus the six pairs of them (e.g. count-position: "many bees crowded on the left side")
- **58 queries total**: 10 per single type (2 development `Train:` queries, 8 test) and 3 per pair (1 development, 2 test), so 14 train and 44 test
- **Near-duplicates:** each method's top 10 holds at most one clip per source video, so neighbouring clips cannot fill it
- **Pooling:** top 10 of each method merged per query, duplicates removed, shuffled, method and rank hidden; rated blind in a CSV (0 = not relevant, 1 = partly, 2 = clearly relevant). Re-pooling after adding a method keeps existing ratings
- **Metrics:** mAP@10 and nDCG@10 (mean and spread) per method × query type. Relevant = rating ≥ 1; the AP denominator is the relevant clips in the pool (at most 10); the nDCG ideal is the best 10 ratings in the pool; unrated clips count as 0
- **Latency:** p50 / p99 over 200 queries per method, model loading excluded
- **No model-made labels:** all relevance judgements are human-annotated

### Results (test split)
mAP@10 / nDCG@10 per single query type (8 test queries each; all 1,743 pooled clips rated). Full numbers, including the six pairs (2 queries each, so noisy), in `results/test_metrics.json`.

| Method | Count | General | Motion | Position |
|--------|-------|---------|--------|----------|
| CLIP | 0.57 / 0.44 | 0.57 / 0.60 | 0.59 / 0.48 | 0.27 / 0.35 |
| SigLIP 2 | 0.50 / 0.47 | **0.86 / 0.78** | 0.66 / 0.56 | 0.43 / 0.44 |
| Motion | – | – | 0.59 / 0.54 | – |
| Detector | **0.81 / 0.75** | 0.75 / 0.69 | 0.69 / 0.59 | 0.70 / 0.64 |
| RRF | 0.67 / 0.64 | 0.70 / 0.72 | 0.64 / 0.54 | 0.44 / 0.44 |
| Tuned RRF | 0.77 / 0.72 | 0.76 / 0.71 | 0.69 / 0.60 | 0.67 / 0.63 |
| Tuned RRF, no CLIP | 0.77 / 0.71 | 0.76 / 0.70 | **0.70 / 0.61** | **0.71 / 0.66** |
| SigLIP-boosted RRF (no CLIP) | 0.73 / 0.69 | 0.77 / 0.71 | 0.68 / 0.59 | **0.71 / 0.66** |

**Note on position:** the detector's up/down, spread and position-motion rules were added *after* the first test scoring, which had covered only 3 position queries with the detector (detector 0.63 / 0.55; tuned RRF 0.53 / 0.52). The new clips they found were rated blind like all others, and the tuned RRF weights were not changed. The first scoring is kept in `results/test_metrics_frozen.json`. CLIP and SigLIP 2 position scores dropped although their top 10s did not change: the new clips added relevant ones to the pool, which raises the bar for every method.

Latency per query (p50 / p99, `results/latency.json`): CLIP 20 / 27 ms, SigLIP 2 63 / 78 ms, motion 0.1 / 0.2 ms, detector 0.1 / 46 ms (detections precomputed; the p99 is the spread rules looping over boxes), RRF 87 / 179 ms, tuned RRF 89 / 150 ms, tuned RRF without CLIP 68 / 118 ms, SigLIP-boosted RRF 69 / 143 ms.

### Annotation rules

Each clip is rated against the query text only, by watching the whole 2-second clip: **2** = clearly relevant,
**1** = partly, **0** = not. The rule for the current query type is shown under the query in the app's Rate tab
(`RATING_RULES` in `src/evaluation.py`, the same text as below).

**Count** — judge how much of the board is covered.
- *Busy wording* (packed, crowded, dense, many): 2 = covered almost edge to edge · 1 = many bees but clear empty patches · 0 = scattered bees or mostly empty. Example: "Landing board packed with bees".
- *Quiet wording* (few, handful, almost empty, single): the mirror — 2 = mostly empty (an empty board too) · 1 = medium coverage · 0 = well covered. Example: "Only a handful of bees on the board".
- *Medium wording* ("a medium number, not crowded"): 2 = clearly in between · 1 = borderline · 0 = packed or empty.

**Position** — judge only the named place and ignore the rest of the board.
- *Bees on / clustered on a side*: 2 = a clear dense group there, fewer bees elsewhere · 1 = bees there but spread over the board too, no clear group · 0 = that place empty or the group sits elsewhere. Example: "Bees clustered on the right side of the board".
- *A side empty*: 2 = few to no bees there (an empty board too) · 1 = some bees, clearly fewer than elsewhere · 0 = as busy as the rest. Example: "Left side of the board empty".
- *Other places* (mesh grid, upper board, lower edge, two groups, evenly spread): 2 = clearly as described · 1 = partly · 0 = not.

**Motion**
- *Busy wording* (flying, taking off, landing, traffic, walking, crawling): count the bees doing it — 2 = several at once · 1 = one or two, or only partly that motion · 0 = none during the clip. Example: "Many bees flying in front of the hive".
- *Flying low over the grass*: due to constraints on depth perception, bees flying over the grass are counted as partly (1).
- *Calm wording* (still, quiet, little movement): judge the whole clip — 2 = calm throughout · 1 = calm with some activity · 0 = busy. Bees must be on the board for anything above 0. Example: "Bees sitting still on the board".

**General** — is the described thing there? 2 = clearly (that hive, sunlight, shade, plants, pole …) · 0 = not · 1 only when it is partly or unclearly visible. Hive descriptions are 2 or 0, since each fits exactly one hive. Example: "Hive with an orange front and a white toothed entrance".

**Pairs of types** — judge each part with its own type's rule. 2 = both parts clearly true · 1 = one part true and the other partly true or unclear · 0 = neither, or one part plainly wrong. Per pair:
- *count-position*: count only inside the named place — 2 = that place densely covered (for "a few": only a few there) · 1 = a fair group with gaps, or the whole board busy without that side standing out · 0 = otherwise.
- *count-motion*: count the bees doing the motion — 2 = the motion with the asked number of bees · 1 = the motion with fewer or more bees than asked · 0 = the motion is missing.
- *position-motion*: judge the motion inside the named place — 2 = that motion clearly happening there · 1 = happening but mostly elsewhere, or only partly there · 0 = no such motion there.
- *count-general, position-general, motion-general*: the general part decides first; the wrong scene is 0. If it is right, the other part is graded with its own rule.

---

## Key references

- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP)
- Tschannen et al. (2025). *SigLIP 2*, arXiv 2502.14786
- Paiss et al. (2023). *Teaching CLIP to Count to Ten*
- Kamath et al. (2023). *What's "up" with vision-language models?*
- Sledevič et al. (2025). Bee detection dataset, PLOS ONE
