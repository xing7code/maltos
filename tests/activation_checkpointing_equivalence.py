from __future__ import annotations

import torch

from models.activation_checkpointing import ActivationCheckpointConfig
from models.llama import LlamaConfig, LlamaForCausalLM
from models.olmo import OlmoConfig, OlmoForCausalLM
from tests.helpers import causal_lm_batch


def _assert_checkpoint_gradients_match(model_cls, config_cls) -> None:
    torch.manual_seed(1234)
    plain = model_cls(
        config_cls(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            max_position_embeddings=16,
            activation_checkpointing=ActivationCheckpointConfig(enabled=False),
        )
    )
    checkpointed = model_cls(
        config_cls(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=3,
            num_attention_heads=4,
            max_position_embeddings=16,
            activation_checkpointing=ActivationCheckpointConfig(enabled=True, every_n_layers=1),
        )
    )
    checkpointed.load_state_dict(plain.state_dict())
    input_ids = torch.randint(0, 64, (2, 8))
    batch = {"input_ids": input_ids, "labels": causal_lm_batch(input_ids)[1]}

    plain_loss = plain(batch)
    checkpointed_loss = checkpointed(batch)
    plain_loss.backward()
    checkpointed_loss.backward()

    torch.testing.assert_close(plain_loss, checkpointed_loss)
    for (plain_name, plain_param), (checkpointed_name, checkpointed_param) in zip(
        plain.named_parameters(), checkpointed.named_parameters(), strict=True
    ):
        assert plain_name == checkpointed_name
        torch.testing.assert_close(plain_param.grad, checkpointed_param.grad, rtol=1e-5, atol=1e-6)


def test_llama_checkpoint_gradients_match() -> None:
    _assert_checkpoint_gradients_match(LlamaForCausalLM, LlamaConfig)


def test_olmo_checkpoint_gradients_match() -> None:
    _assert_checkpoint_gradients_match(OlmoForCausalLM, OlmoConfig)


if __name__ == "__main__":
    test_llama_checkpoint_gradients_match()
    test_olmo_checkpoint_gradients_match()
    print("PASS")
