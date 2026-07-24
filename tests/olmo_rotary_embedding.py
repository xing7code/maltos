from __future__ import annotations

import torch

from models.olmo import OlmoConfig, OlmoForCausalLM, OlmoRotaryEmbedding, _apply_olmo_rotary


def test_olmo_rope_uses_rotate_half_pairing() -> None:
    rotary = OlmoRotaryEmbedding(head_dim=4, max_position_embeddings=4, base=10_000.0)
    cos, sin = rotary(1, 2)
    x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

    actual = _apply_olmo_rotary(x, cos, sin)
    expected = x * cos + torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]]) * sin

    torch.testing.assert_close(actual, expected)


def test_olmo_rope_duplicates_frequency_halves() -> None:
    rotary = OlmoRotaryEmbedding(head_dim=8, max_position_embeddings=4, base=10_000.0)
    cos, sin = rotary(1, 2)

    torch.testing.assert_close(cos[..., :4], cos[..., 4:])
    torch.testing.assert_close(sin[..., :4], sin[..., 4:])


def test_olmo_model_materializes_packed_rope_once_per_forward() -> None:
    model = OlmoForCausalLM(
        OlmoConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            max_position_embeddings=8,
        )
    )
    counters = {"model": 0, "layers": 0}
    handles = [
        model.rotary_emb.register_forward_pre_hook(
            lambda *_: counters.__setitem__("model", counters["model"] + 1)
        )
    ]
    handles.extend(
        layer.self_attn.rotary_emb.register_forward_pre_hook(
            lambda *_: counters.__setitem__("layers", counters["layers"] + 1)
        )
        for layer in model.layers
    )
    try:
        input_ids = torch.randint(0, 32, (1, 8))
        model({"input_ids": input_ids, "position_ids": torch.arange(8).unsqueeze(0)})
    finally:
        for handle in handles:
            handle.remove()

    assert counters == {"model": 1, "layers": 0}


def test_shared_packed_rope_matches_per_layer_materialization() -> None:
    model = OlmoForCausalLM(
        OlmoConfig(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            max_position_embeddings=8,
        )
    ).eval()
    input_ids = torch.randint(0, 32, (1, 8))
    position_ids = torch.tensor([[0, 1, 2, 0, 1, 2, 3, 4]])

    with torch.no_grad():
        expected = model({"input_ids": input_ids, "position_ids": position_ids})
        # ``rotary=None`` exercises the original per-attention materialization
        # path, providing a direct numerical oracle for the shared cache.
        x = model.embed_tokens(input_ids)
        for layer in model.layers:
            x = layer(x, position_ids=position_ids, rotary=None)
        actual = model.lm_head(model.norm(x))

    torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    test_olmo_rope_uses_rotate_half_pairing()
    test_olmo_rope_duplicates_frequency_halves()
    test_olmo_model_materializes_packed_rope_once_per_forward()
    test_shared_packed_rope_matches_per_layer_materialization()
    print("PASS")
