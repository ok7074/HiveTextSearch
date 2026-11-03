import torch
from unittest.mock import MagicMock
from src.models.clip_model import get_similarity_scores as clip_scores
from src.models.siglip2_model import get_similarity_scores as siglip2_scores


def _make_mocks(logits: torch.Tensor):
    outputs = MagicMock()
    outputs.logits_per_image = logits

    model = MagicMock()
    model.device = "cpu"
    model.return_value = outputs

    inputs_dict = {"dummy": torch.zeros(1)}
    inputs_obj = MagicMock()
    inputs_obj.to.return_value = inputs_dict

    processor = MagicMock()
    processor.return_value = inputs_obj

    return model, processor


def test_clip_returns_raw_logits_not_softmax():
    """CLIP scores must not be softmax-normalised: with one query, softmax always returns 1.0."""
    logits = torch.tensor([[2.0, 0.5]])
    model, processor = _make_mocks(logits)
    scores = clip_scores(model, processor, images=["x"], texts=["a", "b"])
    assert scores.shape == (1, 2)
    assert not torch.isclose(scores[0].sum(), torch.tensor(1.0), atol=0.01), (
        "CLIP get_similarity_scores applied softmax — scores must be raw logits for retrieval"
    )


def test_siglip2_scores_are_sigmoid_not_softmax():
    """SigLIP2 scores must be independent sigmoid values in (0, 1), not softmax-normalised."""
    logits = torch.tensor([[2.0, -1.0]])
    model, processor = _make_mocks(logits)
    scores = siglip2_scores(model, processor, images=["x"], texts=["a", "b"])
    assert scores.shape == (1, 2)
    assert (scores >= 0).all() and (scores <= 1).all(), "SigLIP2 scores must be in [0, 1]"
    assert not torch.isclose(scores[0].sum(), torch.tensor(1.0), atol=0.01), (
        "SigLIP2 scores sum to 1 — sigmoid was replaced with softmax"
    )
