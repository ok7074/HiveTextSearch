import torch
from transformers import AutoModel, AutoProcessor

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"


def load_clip(model_id: str = DEFAULT_MODEL_ID, device_map: str = "auto") -> tuple:
    model = AutoModel.from_pretrained(model_id, attn_implementation="sdpa", device_map=device_map)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def get_similarity_scores(model, processor, images, texts: list[str]) -> torch.Tensor:
    """Returns raw similarity logits of shape (n_images, n_texts). Use for zero-shot classification only; for retrieval use get_image_features + get_text_features."""
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits_per_image


def get_image_features(model, processor, images) -> torch.Tensor:
    """Returns L2-normalised image embeddings."""
    inputs = processor(images=images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        features = model.get_image_features(**inputs).pooler_output
    return features / features.norm(dim=-1, keepdim=True)


def get_text_features(model, processor, texts: list[str]) -> torch.Tensor:
    """Returns L2-normalised text embeddings."""
    inputs = processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(model.device)
    with torch.no_grad():
        features = model.get_text_features(**inputs).pooler_output
    return features / features.norm(dim=-1, keepdim=True)
