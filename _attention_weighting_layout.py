"""Private tokenizer-layout support for the Krea2 attention-weighted encoder."""

import hashlib
import json
import numbers
import os
import re
from decimal import Decimal, InvalidOperation

import torch
from transformers import Qwen2TokenizerFast


KREA2_CONDITIONING_WIDTH = 12 * 2560
KREA2_TOKEN_KEY = "qwen3vl_4b"
WEIGHT_METADATA_KEY = "krea2_weighted_phrase_layout"

KREA2_TEMPLATE_PREFIX = (
    "<|im_start|>system\n"
    "Describe the image by detailing the color, shape, size, texture, quantity, "
    "text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n"
)
KREA2_TEMPLATE_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
TOKENIZER_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "comfy",
    "text_encoders",
    "qwen25_tokenizer",
)

WEIGHT_PATTERN = re.compile(
    r"\((?P<phrase>[^()]+):"
    r"(?P<weight>[+-]?(?:\d+(?:\.\d*)?|\.\d+))\)",
    re.UNICODE,
)

_tokenizer = None


def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = Qwen2TokenizerFast.from_pretrained(TOKENIZER_PATH)
    return _tokenizer


def _decode_piece(tokenizer, token_id):
    value = tokenizer.decode(
        [token_id],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    return value.replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t")


def _parse_weighted_phrases(text):
    if not isinstance(text, str):
        raise ValueError("text must be a string")

    clean_parts = []
    spans = []
    cursor = 0
    clean_length = 0
    for match in WEIGHT_PATTERN.finditer(text):
        phrase = match.group("phrase")
        if not phrase or not phrase.strip():
            raise ValueError("A weighted phrase cannot be empty.")
        raw_weight = match.group("weight")
        try:
            decimal_weight = Decimal(raw_weight)
        except InvalidOperation as error:
            raise ValueError(
                f"Invalid weight {raw_weight!r} for phrase {phrase!r}."
            ) from error
        if not decimal_weight.is_finite():
            raise ValueError(f"Weight for phrase {phrase!r} must be finite.")
        if decimal_weight < 0:
            raise ValueError(
                f"Weight for phrase {phrase!r} must be zero or greater. "
                "Use a value between 0 and 1 to reduce it."
            )

        prefix = text[cursor : match.start()]
        clean_parts.append(prefix)
        clean_length += len(prefix)
        start = clean_length
        clean_parts.append(phrase)
        clean_length += len(phrase)
        spans.append(
            {
                "phrase": phrase,
                "weight": float(decimal_weight),
                "weight_text": raw_weight,
                "start": start,
                "end": clean_length,
                "source_start": match.start(),
                "source_end": match.end(),
                "source": match.group(0),
            }
        )
        cursor = match.end()

    clean_parts.append(text[cursor:])
    return "".join(clean_parts), spans


def _build_layout(text):
    clean_text, spans = _parse_weighted_phrases(text)
    tokenizer = _get_tokenizer()
    rendered = KREA2_TEMPLATE_PREFIX + clean_text + KREA2_TEMPLATE_SUFFIX
    encoded = tokenizer(
        rendered,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    full_ids = [int(value) for value in encoded["input_ids"]]
    offsets = encoded["offset_mapping"]
    prefix_ids = tokenizer(
        KREA2_TEMPLATE_PREFIX,
        add_special_tokens=False,
    )["input_ids"]
    suffix_ids = tokenizer(
        KREA2_TEMPLATE_SUFFIX,
        add_special_tokens=False,
    )["input_ids"]
    prefix_count = len(prefix_ids)
    suffix_count = len(suffix_ids)
    if full_ids[:prefix_count] != prefix_ids or full_ids[-suffix_count:] != suffix_ids:
        raise ValueError("The Qwen tokenizer did not preserve Krea2's template boundaries.")

    retained_ids = full_ids[prefix_count:]
    retained_offsets = offsets[prefix_count:]
    content_count = len(retained_ids) - suffix_count
    prompt_offset = len(KREA2_TEMPLATE_PREFIX)
    rows = []
    for index in range(content_count):
        absolute_start, absolute_end = retained_offsets[index]
        rows.append(
            {
                "index": index,
                "token_id": retained_ids[index],
                "piece": _decode_piece(tokenizer, retained_ids[index]),
                "start": max(0, int(absolute_start) - prompt_offset),
                "end": min(len(clean_text), int(absolute_end) - prompt_offset),
                "weight": 1.0,
            }
        )

    owner_by_row = {}
    weights = [1.0] * len(retained_ids)
    for span_index, span in enumerate(spans):
        indices = [
            row["index"]
            for row in rows
            if row["start"] < span["end"] and row["end"] > span["start"]
        ]
        if not indices:
            raise ValueError(
                f"Weighted phrase {span['phrase']!r} did not overlap a Qwen token row."
            )
        for index in indices:
            previous = owner_by_row.get(index)
            if previous is not None:
                raise ValueError(
                    "Two weighted phrases resolve to the same Qwen token row: "
                    f"{spans[previous]['source']!r} and {span['source']!r}. "
                    "Separate them with ordinary unweighted text."
                )
            owner_by_row[index] = span_index
            weights[index] = span["weight"]
            rows[index]["weight"] = span["weight"]
        span["row_indices"] = indices
        span["token_ids"] = [rows[index]["token_id"] for index in indices]
        span["pieces"] = [rows[index]["piece"] for index in indices]

    payload = json.dumps(
        {
            "full_ids": full_ids,
            "spans": [
                {
                    "start": span["start"],
                    "end": span["end"],
                    "weight": span["weight_text"],
                }
                for span in spans
            ],
        },
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return {
        "version": 1,
        "fingerprint": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "source_text": text,
        "clean_text": clean_text,
        "full_token_ids": full_ids,
        "stripped_prefix_rows": prefix_count,
        "retained_suffix_rows": suffix_count,
        "token_count": len(retained_ids),
        "content_count": content_count,
        "retained_token_ids": retained_ids,
        "row_weights": weights,
        "rows": rows,
        "weighted_spans": spans,
        "insertion_or_deletion": False,
    }


def _single_token_batch(tokens, label):
    if not isinstance(tokens, dict) or set(tokens) != {KREA2_TOKEN_KEY}:
        keys = sorted(tokens) if isinstance(tokens, dict) else []
        raise ValueError(
            f"{label} must come from a Krea2 CLIP tokenizer with the single key "
            f"{KREA2_TOKEN_KEY!r}; received {keys}."
        )
    batches = tokens[KREA2_TOKEN_KEY]
    if not isinstance(batches, list) or len(batches) != 1:
        raise ValueError(f"{label} must resolve to exactly one Qwen token batch.")
    batch = batches[0]
    if not isinstance(batch, list):
        raise ValueError(f"{label} Qwen token batch is malformed.")
    return batch


def _integer_token_ids(batch, label):
    output = []
    for index, item in enumerate(batch):
        if not isinstance(item, (tuple, list)) or not item:
            raise ValueError(f"{label} token entry {index} is malformed.")
        token_id = item[0]
        if not isinstance(token_id, numbers.Integral):
            raise ValueError(
                f"{label} token entry {index} is not a text token. This node accepts "
                "Krea2 text conditioning without visual or embedding tokens."
            )
        output.append(int(token_id))
    return output


def _validate_conditioning_tensor(tensor, token_count, label):
    if not torch.is_tensor(tensor) or tensor.ndim != 3:
        raise ValueError(f"{label} must be a three-dimensional conditioning tensor.")
    if tensor.shape[-1] != KREA2_CONDITIONING_WIDTH:
        raise ValueError(
            f"{label} width must be 12 x 2560 = {KREA2_CONDITIONING_WIDTH}; "
            f"received {tensor.shape[-1]}. Load the text encoder as Krea2."
        )
    if tensor.shape[1] != token_count:
        raise ValueError(
            f"{label} has {tensor.shape[1]} token rows, but the verified Krea2 "
            f"layout has {token_count}."
        )


__all__ = [
    "KREA2_CONDITIONING_WIDTH",
    "WEIGHT_METADATA_KEY",
    "_build_layout",
    "_integer_token_ids",
    "_single_token_batch",
    "_validate_conditioning_tensor",
]
