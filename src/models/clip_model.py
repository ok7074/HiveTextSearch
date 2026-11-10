import torch
from transformers import AutoModel, AutoProcessor

from .siglip2_model import PROMPT

DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"


def load_clip(model_id: str = DEFAULT_MODEL_ID, device_map: str = "auto") -> tuple:
    model = AutoModel.from_pretrained(model_id, attn_implementation="sdpa", device_map=device_map)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def get_image_features(model, processor, images) -> torch.Tensor:
    """Returns L2-normalised image embeddings."""
    inputs = processor(images=images, return_tensors="pt").to(model.device)
    with torch.no_grad():
        features = model.get_image_features(**inputs).pooler_output
    return features / features.norm(dim=-1, keepdim=True)


def get_text_features(model, processor, texts: list[str]) -> torch.Tensor:
    """Returns L2-normalised text embeddings. Texts are wrapped in the same prompt as SigLIP 2's."""
    inputs = processor(
        text=[PROMPT.format(t) for t in texts], return_tensors="pt", padding=True, truncation=True
    ).to(model.device)
    with torch.no_grad():
        features = model.get_text_features(**inputs).pooler_output
    return features / features.norm(dim=-1, keepdim=True)
