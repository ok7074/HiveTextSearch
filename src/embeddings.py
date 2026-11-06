import numpy as np
from pathlib import Path
from typing import Callable
from datasets import Dataset
from PIL import Image

from .frame_selection import DATA_DIR, N_FRAMES

EMBEDDINGS_DIR = DATA_DIR / "embeddings"


def embed_frames(
    dataset: Dataset,
    frames_dir: Path,
    image_features: Callable,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Embed every frame of every clip, keeping dataset order.
    batch_size counts clips, so each model call receives batch_size * N_FRAMES images.
    Returns an array of shape (n_clips, N_FRAMES, embedding_dim).
    """
    # Paths come from frame_names because the stored frame_paths are absolute Windows paths.
    all_names = list(dataset["frame_names"])
    batches: list[np.ndarray] = []

    for start in range(0, len(all_names), batch_size):
        clips = all_names[start:start + batch_size]
        if any(len(names) != N_FRAMES for names in clips):
            raise ValueError(f"Expected {N_FRAMES} frames per clip in the batch at row {start}")
        images = [Image.open(frames_dir / f"{name}.jpg").convert("RGB") for names in clips for name in names]
        features = image_features(images).float().cpu().numpy()
        batches.append(features.reshape(len(clips), N_FRAMES, -1))
        print(f"{start + len(clips)}/{len(all_names)} clips", flush=True)

    return np.concatenate(batches)


def average_frames(frame_embeddings: np.ndarray) -> np.ndarray:
    """Average each clip's frame embeddings into one L2-normalised clip embedding."""
    mean = frame_embeddings.mean(axis=1)
    # The mean of unit vectors is shorter than 1, and cosine search expects unit length.
    return mean / np.linalg.norm(mean, axis=-1, keepdims=True)


def save_embeddings(
    path: Path,
    clip_ids: list[str],
    frame_embeddings: np.ndarray,
    clip_embeddings: np.ndarray,
    model_id: str,
) -> None:
    """Save one model's embeddings to a single .npz file, aligned by row with clip_ids."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        clip_ids=np.array(clip_ids),
        frame_embeddings=frame_embeddings,
        clip_embeddings=clip_embeddings,
        model_id=np.array(model_id),
    )
