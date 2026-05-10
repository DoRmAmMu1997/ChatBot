import pytest

torch = pytest.importorskip("torch")

from src.chatbot.config import tiny_config
from src.chatbot.model import TransformerChatModel


def test_tiny_model_forward_returns_logits_and_loss():
    config = tiny_config(vocab_size=32)
    model = TransformerChatModel(config)
    x = torch.randint(0, config.vocab_size, (2, 8))
    y = torch.randint(0, config.vocab_size, (2, 8))

    logits, loss = model(x, y)

    assert logits.shape == (2, 8, config.vocab_size)
    assert loss is not None
    assert torch.isfinite(loss)


def test_kv_cache_path_returns_one_cache_per_layer():
    config = tiny_config(vocab_size=32)
    model = TransformerChatModel(config)
    x = torch.randint(0, config.vocab_size, (1, 4))

    logits, _, cache = model(x, use_cache=True)

    assert logits.shape == (1, 4, config.vocab_size)
    assert len(cache) == config.n_layer
    assert cache[0][0].shape[2] == 4


def test_generation_supports_top_p_and_beam_search():
    torch.manual_seed(0)
    config = tiny_config(vocab_size=32)
    model = TransformerChatModel(config)
    prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

    sampled = model.generate(prompt, max_new_tokens=2, top_k=8, top_p=0.9, temperature=0.8)
    beams = model.generate(prompt, max_new_tokens=2, num_beams=2, do_sample=False)

    assert sampled.shape == (1, 5)
    assert beams.shape == (1, 5)
