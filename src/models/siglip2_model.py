import torch
from transformers import AutoModel, AutoProcessor

DEFAULT_MODEL_ID = "google/siglip2-base-patch16-224"
_TEXT_MAX_LENGTH = 64
PROMPT = "This is a photo of {}."  # the text format SigLIP2 was trained with; CLIP uses it too for a fair comparison
# Matches CLIP's preprocessing so both models see the same pixels of a portrait frame.
_CENTER_CROP = {"do_center_crop": True, "size": {"shortest_edge": 224}, "crop_size": {"height": 224, "width": 224}}


def load_siglip2(model_id: str = DEFAULT_MODEL_ID, device_map: str = "auto") -> tuple:
    model = AutoModel.from_pretrained(model_id, attn_implementation="sdpa", device_map=device_map)
    processor = AutoProcessor.from_pretrained(model_id)
    model.eval()
    return model, processor


def get_image_features(model, processor, images) -> torch.Tensor:
    """Returns L2-normalised image embeddings."""
    inputs = processor(images=images, return_tensors="pt", **_CENTER_CROP).to(model.device)
    with torch.no_grad():
        features = model.get_image_features(**inputs).pooler_output
    return features / features.norm(dim=-1, keepdim=True)


def get_text_features(model, processor, texts: list[str]) -> torch.Tensor:
    """Returns L2-normalised text embeddings.

    Texts are auto-formatted as 'This is a photo of {query}.' to match training.
    """
    inputs = processor(
        text=[PROMPT.format(t) for t in texts],
        padding="max_length",
        max_length=_TEXT_MAX_LENGTH,
        truncation=True,
        return_tensors="pt",
    ).to(model.device)
    with torch.no_grad():
        features = model.get_text_features(**inputs).pooler_output
    return features / features.norm(dim=-1, keepdim=True)
