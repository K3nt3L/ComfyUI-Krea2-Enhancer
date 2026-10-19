"""Exact phrase weighting in Krea2's shared image-to-text attention route.

This deliberately does not scale, copy, delete, average, or replace text rows.
For an annotated phrase with weight ``w``, it adds ``log(w)`` to the attention
logits only where an image query reads one of that phrase's text keys.  Thus the
selected key receives exactly ``w`` times its original attention odds before
the normal softmax renormalization.
"""

import json
import math

import torch

from ._attention_weighting_layout import (
    KREA2_CONDITIONING_WIDTH,
    WEIGHT_METADATA_KEY,
    _build_layout,
    _integer_token_ids,
    _single_token_batch,
    _validate_conditioning_tensor,
)


ATTENTION_METADATA_KEY = "krea2_phrase_attention_weight_layout"
ZERO_WEIGHT_LOGIT_BIAS = -80.0
HEAD_DIM_ALIGNMENT = 8


def _validate_krea2_model(model):
    if model is None:
        raise ValueError("The MODEL input is missing.")
    try:
        diffusion_model = model.get_model_object("diffusion_model")
    except Exception as error:
        raise ValueError(
            "Could not inspect the connected MODEL as a ComfyUI ModelPatcher."
        ) from error

    required = ("txtfusion", "txtmlp", "blocks", "txtlayers", "txtdim")
    missing = [name for name in required if not hasattr(diffusion_model, name)]
    if missing:
        raise ValueError(
            "The connected MODEL is not a Krea2 model; missing: "
            + ", ".join(missing)
            + "."
        )
    if int(diffusion_model.txtlayers) != 12 or int(diffusion_model.txtdim) != 2560:
        raise ValueError(
            "This node requires Krea2's 12 x 2560 text stack; received "
            f"{diffusion_model.txtlayers} x {diffusion_model.txtdim}."
        )
    if len(diffusion_model.blocks) <= 0:
        raise ValueError("The connected Krea2 model has no shared DiT blocks.")
    return len(diffusion_model.blocks)


def _validated_log_biases(row_weights):
    biases = []
    selected = []
    for index, value in enumerate(row_weights):
        weight = float(value)
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(
                f"Phrase row {index} has invalid weight {value!r}; weights must "
                "be finite and zero or greater."
            )
        if weight == 0.0:
            bias = ZERO_WEIGHT_LOGIT_BIAS
        else:
            bias = math.log(weight)
        biases.append(bias)
        if weight != 1.0:
            selected.append(index)
    return biases, selected


def _call_attention(previous_override, original_attention, *args, **kwargs):
    if previous_override is None:
        return original_attention(*args, **kwargs)
    return previous_override(original_attention, *args, **kwargs)


def _augment_image_attention_qkv(q, k, v, text_length, log_biases):
    """Encode an additive key bias as extra Q/K dimensions.

    The extra image-query component is 1, the corresponding text-key component
    is bias / scale, and every other new component is zero.  Keeping the
    original scale therefore gives:

        scale * (q @ k + 1 * bias / scale) = original_logit + bias

    This function receives image queries only. Text queries run through the
    original Q/K/V tensors in a separate attention call, so no padded dimension
    or phrase bias enters the text-query calculation. Padding the image call to
    a multiple of eight keeps optimized attention backends usable.
    """
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise ValueError("Krea2 shared attention Q/K/V tensors must be rank four.")
    if q.shape[:2] != k.shape[:2] or q.shape[:2] != v.shape[:2]:
        raise ValueError(
            "Krea2 shared attention Q/K/V batch or head dimensions do not match: "
            f"Q={tuple(q.shape)}, K={tuple(k.shape)}, V={tuple(v.shape)}."
        )
    if k.shape[-2] != v.shape[-2]:
        raise ValueError(
            "Krea2 shared attention K/V token counts do not match: "
            f"K={tuple(k.shape)}, V={tuple(v.shape)}."
        )
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError(
            "Krea2 shared attention Q/K/V head dimensions do not match: "
            f"Q={tuple(q.shape)}, K={tuple(k.shape)}, V={tuple(v.shape)}."
        )
    if not (q.is_floating_point() and k.is_floating_point() and v.is_floating_point()):
        raise ValueError("Krea2 shared attention Q/K/V must be floating-point tensors.")

    token_count = k.shape[-2]
    original_dim = q.shape[-1]
    if text_length <= 0 or text_length >= token_count:
        raise ValueError(
            f"Invalid Krea2 text/image split {text_length}/{token_count}."
        )
    if q.shape[-2] != token_count - text_length:
        raise ValueError(
            "The Krea2 image-query count does not match the text/image split: "
            f"{q.shape[-2]} != {token_count} - {text_length}."
        )
    if len(log_biases) != text_length:
        raise ValueError(
            "The weighted prompt contains "
            f"{len(log_biases)} text rows, but Krea2 is running with "
            f"{text_length}. Do not use this patched MODEL with different "
            "conditioning."
        )

    augmented_dim = (
        (original_dim + 1 + HEAD_DIM_ALIGNMENT - 1) // HEAD_DIM_ALIGNMENT
    ) * HEAD_DIM_ALIGNMENT
    extra_dim = augmented_dim - original_dim
    scale = original_dim ** -0.5

    q_extra = torch.ones(
        (*q.shape[:-1], extra_dim),
        dtype=q.dtype,
        device=q.device,
    )
    if extra_dim > 1:
        q_extra[..., 1:] = 0.0

    k_extra = torch.zeros(
        (*k.shape[:-1], extra_dim),
        dtype=k.dtype,
        device=k.device,
    )
    key_bias = torch.tensor(log_biases, dtype=k.dtype, device=k.device)
    k_extra[..., :text_length, 0] = key_bias / scale

    v_extra = torch.zeros(
        (*v.shape[:-1], extra_dim),
        dtype=v.dtype,
        device=v.device,
    )
    return (
        torch.cat((q, q_extra), dim=-1),
        torch.cat((k, k_extra), dim=-1),
        torch.cat((v, v_extra), dim=-1),
        original_dim,
        augmented_dim,
        scale,
    )


def _slice_query_mask(mask, start, end, full_query_length):
    if mask is None:
        return None
    if not torch.is_tensor(mask) or mask.ndim < 2:
        raise ValueError(
            "An existing Krea2 attention mask must be a tensor with at least "
            "two dimensions."
        )
    query_length = mask.shape[-2]
    if query_length == 1:
        return mask
    if query_length != full_query_length:
        raise ValueError(
            "An existing Krea2 attention mask has an incompatible query axis: "
            f"{query_length} != {full_query_length}."
        )
    return mask[..., start:end, :]


def _concat_query_outputs(text_output, image_output, skip_output_reshape):
    axis = -2 if skip_output_reshape else 1
    return torch.cat((text_output, image_output), dim=axis)


def _strip_augmented_output(output, heads, original_dim, augmented_dim, skip_output_reshape):
    if skip_output_reshape:
        if output.ndim != 4 or output.shape[-1] != augmented_dim:
            raise ValueError(
                "The optimized attention backend returned an unexpected Krea2 "
                f"head output shape {tuple(output.shape)}."
            )
        return output[..., :original_dim]

    expected_width = int(heads) * augmented_dim
    if output.ndim != 3 or output.shape[-1] != expected_width:
        raise ValueError(
            "The optimized attention backend returned an unexpected Krea2 "
            f"output shape {tuple(output.shape)}; expected width {expected_width}."
        )
    batch, tokens, _ = output.shape
    return (
        output.reshape(batch, tokens, int(heads), augmented_dim)
        [..., :original_dim]
        .reshape(batch, tokens, int(heads) * original_dim)
    )


def _make_attention_override(row_weights, previous_override, block_count):
    log_biases, selected_rows = _validated_log_biases(row_weights)
    if not selected_rows:
        raise ValueError("Internal error: an attention patch was requested with no weights.")

    def phrase_attention_override(
        original_attention,
        q,
        k,
        v,
        heads,
        *args,
        **kwargs,
    ):
        transformer_options = kwargs.get("transformer_options")
        if not isinstance(transformer_options, dict):
            return _call_attention(
                previous_override,
                original_attention,
                q,
                k,
                v,
                heads,
                *args,
                **kwargs,
            )

        block_index = transformer_options.get("block_index")
        image_slice = transformer_options.get("img_slice")
        is_krea2_shared_block = (
            isinstance(block_index, int)
            and 0 <= block_index < block_count
            and transformer_options.get("block_type") == "single"
            and transformer_options.get("total_blocks") == block_count
            and isinstance(image_slice, (tuple, list))
            and len(image_slice) == 2
            and kwargs.get("skip_reshape", False) is True
            and q.ndim == 4
        )
        if not is_krea2_shared_block:
            return _call_attention(
                previous_override,
                original_attention,
                q,
                k,
                v,
                heads,
                *args,
                **kwargs,
            )

        text_length = int(image_slice[0])
        total_length = int(image_slice[1])
        if q.shape[-2] != total_length or k.shape[-2] != total_length:
            raise ValueError(
                "Krea2's runtime img_slice disagrees with the shared attention "
                f"shape: img_slice={list(image_slice)}, Q={tuple(q.shape)}, "
                f"K={tuple(k.shape)}. No phrase weighting was applied."
            )

        text_kwargs = dict(kwargs)
        image_kwargs = dict(kwargs)
        existing_mask = kwargs.get("mask")
        text_kwargs["mask"] = _slice_query_mask(
            existing_mask,
            0,
            text_length,
            total_length,
        )
        image_kwargs["mask"] = _slice_query_mask(
            existing_mask,
            text_length,
            total_length,
            total_length,
        )

        # Text queries use the original dimensions and receive no phrase bias.
        # Self-attention is separable over query rows, so splitting queries does
        # not change which keys or values these text rows can see.
        text_output = _call_attention(
            previous_override,
            original_attention,
            q[..., :text_length, :],
            k,
            v,
            heads,
            *args,
            **text_kwargs,
        )

        (
            augmented_q,
            augmented_k,
            augmented_v,
            original_dim,
            augmented_dim,
            original_scale,
        ) = _augment_image_attention_qkv(
            q[..., text_length:, :],
            k,
            v,
            text_length,
            log_biases,
        )

        image_kwargs["scale"] = original_scale
        image_output = _call_attention(
            previous_override,
            original_attention,
            augmented_q,
            augmented_k,
            augmented_v,
            heads,
            *args,
            **image_kwargs,
        )
        image_output = _strip_augmented_output(
            image_output,
            heads,
            original_dim,
            augmented_dim,
            bool(image_kwargs.get("skip_output_reshape", False)),
        )
        return _concat_query_outputs(
            text_output,
            image_output,
            bool(image_kwargs.get("skip_output_reshape", False)),
        )

    phrase_attention_override.krea2_phrase_attention_weighting = {
        "row_weights": tuple(float(value) for value in row_weights),
        "selected_rows": tuple(selected_rows),
        "block_count": int(block_count),
    }
    return phrase_attention_override


def _report(layout, block_count, active):
    return json.dumps(
        {
            "operation": "Krea2 image-query to text-key attention-odds weighting",
            "active": bool(active),
            "formula": (
                "selected logit = original logit + log(weight); therefore "
                "selected attention odds = original odds * weight"
            ),
            "zero_weight": (
                f"uses finite logit bias {ZERO_WEIGHT_LOGIT_BIAS} to avoid "
                "NaNs in optimized attention"
            ),
            "shared_blocks": f"0-{block_count - 1}",
            "text_rows_changed": False,
            "token_rows_inserted_or_deleted": False,
            "text_to_text_logits_changed_directly": False,
            "image_to_text_logits_changed": bool(active),
            "clean_text_encoded": layout["clean_text"],
            "token_count": layout["token_count"],
            "weighted_spans": [
                {
                    "source": span["source"],
                    "phrase": span["phrase"],
                    "weight": span["weight"],
                    "rows": span["row_indices"],
                    "pieces": span["pieces"],
                    "token_ids": span["token_ids"],
                }
                for span in layout["weighted_spans"]
            ],
        },
        ensure_ascii=False,
        indent=2,
    )


class Krea2AttentionWeightedPhraseEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "text": (
                    "STRING",
                    {
                        "multiline": True,
                        "dynamicPrompts": False,
                        "default": "",
                        "tooltip": (
                            "Use (phrase:weight). The clean phrase is encoded normally. "
                            "Weight changes only "
                            "the image-query -> phrase-text-key attention odds in every "
                            "Krea2 shared DiT block. 1 is exact no-op; 0 suppresses; "
                            "values above 1 increase. No rows are copied or scaled."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CONDITIONING", "STRING")
    RETURN_NAMES = ("model", "conditioning", "weight_report")
    OUTPUT_TOOLTIPS = (
        "Krea2 MODEL whose shared attention reads weighted phrase keys at weighted odds.",
        "Normal, unscaled Krea2 conditioning for the annotation-free prompt.",
        "Exact phrase rows, weights, affected route, formula, and invariants.",
    )
    FUNCTION = "encode_and_patch"
    CATEGORY = "Krea2/conditioning"
    DESCRIPTION = (
        "Encodes (phrase:weight) as ordinary Krea2 text, then applies the weight "
        "inside shared attention: only image queries reading the phrase's text "
        "keys receive log(weight). It does not scale conditioning vectors, repeat "
        "tokens, alter sequence length, or directly alter text-to-text logits."
    )

    def encode_and_patch(self, model, clip, text):
        if clip is None:
            raise ValueError("The CLIP input is missing.")
        block_count = _validate_krea2_model(model)
        layout = _build_layout(text)

        prompt_tokens = clip.tokenize(layout["clean_text"])
        prompt_batch = _single_token_batch(prompt_tokens, "The weighted prompt")
        prompt_ids = _integer_token_ids(prompt_batch, "The weighted prompt")
        if prompt_ids != layout["full_token_ids"]:
            raise ValueError(
                "The connected CLIP tokenizer does not match the verified local Krea2 "
                "Qwen tokenizer. No phrase weighting was installed."
            )

        conditioning = clip.encode_from_tokens_scheduled(prompt_tokens)
        active = any(weight != 1.0 for weight in layout["row_weights"])
        runtime_layout = {
            "version": 1,
            "operation": "image-query to text-key attention-odds weighting",
            "fingerprint": layout["fingerprint"],
            "clean_text": layout["clean_text"],
            "token_count": layout["token_count"],
            "row_weights": list(layout["row_weights"]),
        }
        output_conditioning = []
        for tensor, metadata in conditioning:
            _validate_conditioning_tensor(
                tensor,
                layout["token_count"],
                "The original Krea2 conditioning",
            )
            updated_metadata = metadata.copy()
            updated_metadata[ATTENTION_METADATA_KEY] = runtime_layout
            # Remove stale metadata from the rejected vector-scaling node if a
            # caller programmatically chains objects.  The tensor itself is not
            # modified here.
            updated_metadata.pop(WEIGHT_METADATA_KEY, None)
            output_conditioning.append([tensor, updated_metadata])

        if not active:
            return (model, output_conditioning, _report(layout, block_count, False))

        patched_model = model.clone()
        transformer_options = patched_model.model_options.setdefault(
            "transformer_options", {}
        )
        previous_override = transformer_options.get("optimized_attention_override")
        transformer_options["optimized_attention_override"] = _make_attention_override(
            layout["row_weights"],
            previous_override,
            block_count,
        )
        return (
            patched_model,
            output_conditioning,
            _report(layout, block_count, True),
        )


__all__ = ["Krea2AttentionWeightedPhraseEncoder"]
