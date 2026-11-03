import torch
from transformers import AutoModel, AutoProcessor

DEFAULT_MODEL_ID = "google/siglip2-base-patch16-224"
_TEXT_MAX_LENGTH = 64


def load_siglip2(model_id: str = DEFAULT_MODEL_ID, device_map: str = "auto") -> tuple:
    model = AutoModel.from_pretrained(model_id, attn_implementation="sdpa", device_map=device_map)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def get_similarity_scores(model, processor, images, texts: list[str]) -> torch.Tensor:
    """Returns sigmoid probabilities of shape (n_images, n_texts).

    SigLIP2 uses sigmoid (independent per label) rather than softmax.
    Texts are auto-formatted as 'This is a photo of {label}.' to match training.
    """
    formatted = [f"This is a photo of {t}." for t in texts]
    inputs = processor(
        text=formatted,
        images=images,
        padding="max_length",
        max_length=_TEXT_MAX_LENGTH,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return torch.sigmoid(outputs.logits_per_image)


def get_image_features(model, processor, images) -> torch.Tensor:
    """Returns L2-normalised image embeddings."""
    inputs = processor(images=images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features / features.norm(dim=-1, keepdim=True)


def get_text_features(model, processor, texts: list[str]) -> torch.Tensor:
    """Returns L2-normalised text embeddings.

    Processor applies lowercasing and fixed padding/truncation (max_length=64) automatically.
    """
    inputs = processor(
        text=texts,
        padding="max_length",
        max_length=_TEXT_MAX_LENGTH,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        features = model.get_text_features(**inputs)
    return features / features.norm(dim=-1, keepdim=True)
