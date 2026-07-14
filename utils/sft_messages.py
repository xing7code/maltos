from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EncodedSFTExample:
    token_ids: list[int]
    supervised_mask: list[int]


def encode_sft_example(
    row: dict[str, Any],
    *,
    tokenizer,
    messages_column: str,
    prompt_column: str,
    completion_column: str,
    apply_chat_template: bool,
    assistant_only_loss: bool,
    append_eos: bool,
) -> EncodedSFTExample | None:
    messages = extract_messages(row, messages_column)
    if messages is None:
        messages = messages_from_prompt_completion(
            row.get(prompt_column),
            row.get(completion_column),
        )
    if messages is None:
        return None
    if assistant_only_loss and not has_supervised_assistant_turn(messages):
        return None
    if apply_chat_template:
        return encode_messages_with_chat_template(
            messages,
            tokenizer=tokenizer,
            assistant_only_loss=assistant_only_loss,
        )
    return encode_messages_plain(
        messages,
        tokenizer=tokenizer,
        assistant_only_loss=assistant_only_loss,
        append_eos=append_eos,
    )


def extract_messages(row: dict[str, Any], messages_column: str) -> list[dict[str, str]] | None:
    if messages_column not in row:
        return None
    raw_messages = _maybe_json_load(row.get(messages_column))
    if not isinstance(raw_messages, list):
        return None
    messages: list[dict[str, str]] = []
    for raw_message in raw_messages:
        if not isinstance(raw_message, dict):
            raise ValueError(f"message must be a mapping, got {type(raw_message)!r}")
        role = raw_message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"message role must be a non-empty string, got {role!r}")
        content = normalize_message_content(raw_message.get("content"))
        if content:
            messages.append({"role": role, "content": content})
    return messages or None


def messages_from_prompt_completion(prompt: Any, completion: Any) -> list[dict[str, str]] | None:
    if not isinstance(prompt, str) or not isinstance(completion, str):
        return None
    if not prompt or not completion:
        return None
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": completion},
    ]


def has_supervised_assistant_turn(messages: list[dict[str, str]]) -> bool:
    return any(message["role"] == "assistant" and bool(message["content"]) for message in messages)


def encode_messages_with_chat_template(
    messages: list[dict[str, str]],
    *,
    tokenizer,
    assistant_only_loss: bool,
) -> EncodedSFTExample:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            "tokenizer does not define a chat_template; "
            "rerun with --no-apply-chat-template or choose a chat tokenizer"
        )
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
        return_assistant_tokens_mask=assistant_only_loss,
    )
    token_ids = [int(token_id) for token_id in encoded["input_ids"]]
    if not assistant_only_loss:
        return EncodedSFTExample(
            token_ids=token_ids,
            supervised_mask=[1] * len(token_ids),
        )
    assistant_masks = encoded.get("assistant_masks")
    if assistant_masks is None:
        raise ValueError(
            "chat template did not return assistant_masks; "
            "the tokenizer template likely needs {% generation %} support"
        )
    supervised_mask = [int(mask) for mask in assistant_masks]
    if supervised_mask and not any(supervised_mask):
        raise ValueError(
            "chat template returned assistant_masks but all values are 0; "
            "the tokenizer template likely needs {% generation %} blocks around assistant content"
        )
    return EncodedSFTExample(
        token_ids=token_ids,
        supervised_mask=supervised_mask,
    )


def encode_messages_plain(
    messages: list[dict[str, str]],
    *,
    tokenizer,
    assistant_only_loss: bool,
    append_eos: bool,
) -> EncodedSFTExample:
    """Fallback, try to use apply_chat_template instead!"""
    token_ids: list[int] = []
    supervised_mask: list[int] = []
    eos_id = tokenizer.eos_token_id
    for message in messages:
        role = message["role"]
        content = message["content"]
        role_token_ids = [int(token_id) for token_id in tokenizer.encode(f"{role}: ", add_special_tokens=False)]
        content_token_ids = [int(token_id) for token_id in tokenizer.encode(content, add_special_tokens=False)]
        eos_token_ids = [int(eos_id)] if append_eos and eos_id is not None else []

        token_ids.extend(role_token_ids)
        token_ids.extend(content_token_ids)
        token_ids.extend(eos_token_ids)

        if not assistant_only_loss:
            mask_value = [1] * (len(role_token_ids) + len(content_token_ids) + len(eos_token_ids))
            supervised_mask.extend(mask_value)
            continue

        if role == "assistant":
            supervised_mask.extend([0] * len(role_token_ids))
            supervised_mask.extend([1] * len(content_token_ids))
            supervised_mask.extend([1] * len(eos_token_ids))
            continue

        mask_value = [0] * (len(role_token_ids) + len(content_token_ids) + len(eos_token_ids))
        supervised_mask.extend(mask_value)

    return EncodedSFTExample(token_ids=token_ids, supervised_mask=supervised_mask)


def normalize_message_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        text = content.get("text")
        if isinstance(text, str):
            return text
        raise ValueError(f"unsupported message content mapping: {content!r}")
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
                continue
            if not isinstance(part, dict):
                raise ValueError(f"unsupported message content part: {part!r}")
            part_type = part.get("type")
            text = part.get("text")
            if isinstance(text, str) and (part_type is None or part_type == "text"):
                text_parts.append(text)
                continue
            raise ValueError(f"unsupported non-text message content part: {part!r}")
        return "".join(text_parts)
    raise ValueError(f"unsupported message content type={type(content)!r}")


def _maybe_json_load(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    stripped = value.strip()
    if not stripped or stripped[0] not in "[{":
        return value
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        return value
