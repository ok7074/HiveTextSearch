# HiveTextSearch

Text search over 2-second clips of honeybee hive entrance videos. Given a natural-language query ("many bees", "bees on the left", "busy entrance"), the system ranks ~1,440 short clips by relevance using vision-language models and a bee detector.

Presentation deadline: **25 November 2026** — 15 minutes + Q&A, at most 3 slides.

---

## What it does

- Cuts 96 unlabeled hive entrance videos (30 s, 1080×1920 @ 60 fps) into 2-second clips (~1,440 total)
- Samples 4 evenly-spaced frames per clip and computes one embedding per clip (average of its 4 frames)
- At query time, embeds the text query and ranks all clips by cosine similarity
- Compares four retrieval methods:
  | Method | Description |
  |--------|-------------|
  | CLIP | OpenAI `clip-vit-base-patch32`, zero-shot |
  | SigLIP 2 | Google `siglip2-base-patch16-224`, zero-shot |
  | Bee detector | YOLO fine-tuned on bee data; counts per region (left / middle / right) |
  | Combined | Weighted mix of image–text score and detector score |
- Evaluates with **mAP** and **nDCG** on hand-annotated relevance labels (0 / 1 / 2) across four query types: count, position, motion, general description
- Reports latency at p50 and p99

No vector database or cloud service needed — at ~1,440 clips, a NumPy matrix multiply searches everything in milliseconds.

---

## Repository layout

```
HiveTextSearch/
├── src/
│   ├── __init__.py          # activates MLflow autologging on import
│   ├── tracking.py          # mlflow.autolog() + @track_run decorator
│   └── models/
│       ├── __init__.py
│       ├── clip_model.py    # load_clip, get_similarity_scores, get_image/text_features
│       └── siglip2_model.py # load_siglip2, get_similarity_scores, get_image/text_features
├── scripts/                 # one-off processing scripts (frame extraction, embedding, etc.)
├── notebooks/               # exploratory notebooks
├── experiments/
│   └── logs/                # experiment outputs
├── data/                    # dataset (git-ignored)
└── README.md
```

---

## Setup

**Requirements:** Python 3.11+, a GPU is helpful for indexing but not required for search.

```bash
# 1. Activate the virtual environment
HiveTextSearch\Scripts\Activate.ps1        # Windows PowerShell
# source HiveTextSearch/bin/activate       # macOS / Linux

# 2. Install dependencies
pip install numpy opencv-python datasets transformers torch Pillow mlflow hf_xet accelerate pytest ultralytics optuna
```

**Dataset** — download once from Hugging Face (gated, requires an access token):

```python
from datasets import load_dataset
ds = load_dataset("mrkschtr/thehiveisreal", token="<your_hf_token>")
ds.save_to_disk("data/thehiveisreal")
```

---

## Usage

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
- For search, both rank clips by cosine similarity between embeddings; the loss only shapes training

### Bee detector
YOLO fine-tuned on the [Sledevič et al. (2025)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11856287/) public bee dataset. Counts bees per frame region (left / middle / right third), then maps query keywords to count signals via simple rules.

### Combined score
`final = w × image–text score + (1 − w) × detector score`, with `w` tuned on development queries only and reported on held-out test queries.

### Baseline
Frame-difference motion score: average pixel change across the 4 frames per clip. Used as a sanity check and fallback if the detector does not reach acceptable quality.

---

## Evaluation

- **Query types:** count ("many bees"), position ("bees on the left"), motion ("busy entrance"), general description
- **40 queries total**, 10 per type; 2 per type held out as development queries
- **Pooling:** top 20 of each method merged per query; relevance labeled blind (0 = not relevant, 1 = partly, 2 = clearly relevant)
- **Metrics:** mAP and nDCG per method × query type; p50 / p99 latency over 200+ queries
- **No model-made labels:** all relevance judgements are human-annotated

---

## Key references

- Radford et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision* (CLIP)
- Tschannen et al. (2025). *SigLIP 2*, arXiv 2502.14786
- Paiss et al. (2023). *Teaching CLIP to Count to Ten*
- Kamath et al. (2023). *What's "up" with vision-language models?*
- Sledevič et al. (2025). Bee detection dataset, PLOS ONE
