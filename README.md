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
python scripts/scoring/scoring.py                     # -> results/train_metrics.json
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
YOLO26n (640 px) fine-tuned on the [Sledevič et al. (2025)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11856287/) public bee dataset, split by hive so test hives are never seen in training (10 Optuna trials × 10 epochs, then 60 epochs). Held-out test hives: mAP50-95 0.613, mAP50 0.871, precision 0.824, recall 0.846. A larger YOLO26s (960 px) run is prepared in `scripts/yolo/train_yolo_s960.py`.

Per clip, the boxes in each vertical third (left / center / right) are summed over the 4 frames. The clip's label is the busiest third, `overall` on a tie, `empty` with no bees. At query time:
- a count or position query naming left, center (also "middle", "centre") or right keeps only clips with that label, ranked by the bees in that third
- other count and position queries rank all clips by mean bees per frame; quiet words ("few", "no", "empty", …) put the fewest first
- motion and general queries do not use the detector

Limitation: the detector was trained on landscape images, while the clips are portrait.

### Motion baseline
Frame-difference score: mean pixel change between consecutive frames of a clip. Busiest first, or stillest first for quiet words ("calm", "still", …). Used as the baseline and fallback for motion queries.

### Combined score (planned)
`final = w × image–text score + (1 − w) × detector score`, with `w` tuned on development queries only and reported on held-out test queries.

---

## Evaluation

- **Query types:** count ("many bees"), position ("bees on the left"), motion ("busy entrance"), general description, plus the six pairs of them (e.g. count-position: "many bees crowded on the left side")
- **58 queries total**: 10 per single type (2 development `Train:` queries, 8 test) and 3 per pair (1 development, 2 test), so 14 train and 44 test
- **Near-duplicates:** each method's top 10 holds at most one clip per source video, so neighbouring clips cannot fill it
- **Pooling:** top 10 of each method merged per query, duplicates removed, shuffled, method and rank hidden; rated blind in a CSV (0 = not relevant, 1 = partly, 2 = clearly relevant). Re-pooling after adding a method keeps existing ratings
- **Metrics:** mAP@10 and nDCG@10 (mean and spread) per method × query type. Relevant = rating ≥ 1; the AP denominator is the relevant clips in the pool (at most 10); the nDCG ideal is the best 10 ratings in the pool; unrated clips count as 0
- **Latency:** p50 / p99 over 200 queries per method, model loading excluded
- **No model-made labels:** all relevance judgements are human-annotated

---

## Key references

- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP)
- Tschannen et al. (2025). *SigLIP 2*, arXiv 2502.14786
- Paiss et al. (2023). *Teaching CLIP to Count to Ten*
- Kamath et al. (2023). *What's "up" with vision-language models?*
- Sledevič et al. (2025). Bee detection dataset, PLOS ONE
