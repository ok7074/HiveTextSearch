"""Create CLIP and SigLIP2 embeddings for all frames, averaged into one embedding per clip.

Output (one file per model):
  data/embeddings/clip.npz
  data/embeddings/siglip2.npz

Each file holds these arrays, aligned by row:
  clip_ids          (n_clips,)
  frame_embeddings  (n_clips, 4, embedding_dim)  L2-normalised
  clip_embeddings   (n_clips, embedding_dim)     mean of the 4 frames, L2-normalised
  model_id          the Hugging Face model the embeddings came from

With --flip every frame is mirrored left to right first and the files are saved as
data/embeddings/{clip,siglip2}_flipped.npz instead (the originals are never overwritten). Both models resize the
short side to 224 and center-crop a 224 square; that crop is symmetric, so it does not interfere with the flip.

Reads data/clips_frames_dataset/ and data/frames/. Intended for a GPU runtime.

Each run is tracked in MLflow under the experiment "HiveTextSearch".

Usage:
    python scripts/clip_siglip_embeddings/create_embeddings.py
    python scripts/clip_siglip_embeddings/create_embeddings.py --flip
"""
import argparse
import sys
import mlflow
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from datasets import load_from_disk
from src import track_run
from src.embeddings import EMBEDDINGS_DIR, average_frames, embed_frames, save_embeddings
from src.frame_selection import CLIPS_FRAMES_DATASET_DIR, FRAMES_DIR
from src.models import clip_image_features, load_clip, load_siglip2, siglip2_image_features

MODELS = {
    "clip": (load_clip, clip_image_features),
    "siglip2": (load_siglip2, siglip2_image_features),
}


@track_run(run_name="create_embeddings")
def main(flip: bool = False) -> None:
    dataset = load_from_disk(str(CLIPS_FRAMES_DATASET_DIR))
    clip_ids = list(dataset["clip_id"])
    mlflow.log_metric("clips_in_dataset", len(clip_ids))

    for name, (load_model, image_features) in MODELS.items():
        model, processor = load_model()
        print(f"{name}: {model.name_or_path} on {model.device}", flush=True)
        mlflow.log_param(f"{name}_model_id", model.name_or_path)
        mlflow.log_param(f"{name}_device", str(model.device))

        frame_embeddings = embed_frames(
            dataset, FRAMES_DIR, lambda images: image_features(model, processor, images), flip=flip
        )
        clip_embeddings = average_frames(frame_embeddings)
        save_embeddings(
            EMBEDDINGS_DIR / f"{name}{'_flipped' if flip else ''}.npz", clip_ids, frame_embeddings, clip_embeddings, model.name_or_path
        )
        mlflow.log_metric(f"{name}_embedding_dim", clip_embeddings.shape[1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--flip", action="store_true", help="mirror every frame and save *_flipped.npz")
    main(flip=parser.parse_args().flip)
