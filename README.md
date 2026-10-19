# ComfyUI-Krea2T-Enhancer
(original readme by Capitan1R)



[![Buy Him A Coffee](https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Support-yellow.svg)](https://buymeacoffee.com/capitan01r)

Prompt-adherence enhancement for Krea2 diffusion models in ComfyUI.

This custom node patches the Krea2 text-fusion path during sampling and applies a controlled internal conditioning adjustment intended to improve how strongly the model follows prompt details.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/capitan01R/ComfyUI-Krea2T-Enhancer.git
```

Restart ComfyUI after installing or updating.

No extra Python packages are required beyond a working ComfyUI Krea2 setup.

## Included Nodes

| Node | Output | Purpose |
|---|---|---|
| **ComfyUI-Krea2T-Enhancer** | `MODEL` | Patches the Krea2 model path to improve prompt adherence during sampling. |
| **Krea2T Enhancer Advanced** | `MODEL` | Same enhancer path, plus a direct post-`txtmlp` `text_scale` control for fused text-token strength. |
| **Krea2 Turbo Reference Sigmas (From Latent)** | `SIGMAS`, `LATENT` | Builds a Turbo sigma schedule based on the official Krea 2 Turbo scheduler settings and validates the connected latent dimensions. |
| **Krea2 Text Encode — Attention-Weighted Phrases** | `MODEL`, `CONDITIONING`, `STRING` | Encodes weighted phrases and changes only the image-query-to-selected-text-key attention odds in Krea2's shared DiT blocks. |

## Usage

Place **ComfyUI-Krea2T-Enhancer** between your Krea2 diffusion model loader and sampler:

```text
Load Diffusion Model -> ComfyUI-Krea2T-Enhancer -> KSampler
```

Or use **Krea2T Enhancer Advanced** when you want the additional text-scale control:

```text
Load Diffusion Model -> Krea2T Enhancer Advanced -> KSampler
```

Use your normal Krea2 text encoder, VAE, latent, and sampler setup.

For the sigma scheduler, connect both the loaded Krea2 Turbo model and the same
Empty Latent Image that will be sent to the sampler:

```text
Load Diffusion Model --\
                        > Krea2 Turbo Reference Sigmas (From Latent) -> SIGMAS to sampler
Empty Latent Image ----/                                             -> LATENT to sampler
```

For attention-weighted phrases, connect the final model after all LoRA loaders
and the Krea2 CLIP to the node. Both primary outputs must be used:

```text
Load Diffusion Model -> LoRA loader(s) -> Krea2 Text Encode — Attention-Weighted Phrases -> MODEL to sampler
Krea2 CLIP ---------------------------> Krea2 Text Encode — Attention-Weighted Phrases -> CONDITIONING to positive
```

Write a weighted section as `(phrase:weight)`. The annotation is removed before
tokenization, while the phrase and its original Qwen token positions remain.
`1.0` is an exact no-op, values above `1.0` increase the phrase's attention odds,
values between `0.0` and `1.0` reduce them, and `0.0` suppresses them.

## Controls

### ComfyUI-Krea2T-Enhancer

| Parameter | Default | Meaning |
|---|---:|---|
| `enabled` | `true` | Turns the patch on or off. |
| `strength` | `1.0` | Blends the enhancement from neutral `0.0` to full `2.0`. |
| `debug` | `false` | Prints concise runtime diagnostics to the ComfyUI console. |

### Krea2T Enhancer Advanced

| Parameter | Default | Meaning |
|---|---:|---|
| `enabled` | `true` | Turns the patch on or off. |
| `strength` | `1.0` | Same enhancer strength as the original node, from neutral `0.0` to full `2.0`. |
| `text_scale` | `1.0` | Multiplies fused text tokens immediately after `txtmlp`, before they enter the shared Krea2 stream. |
| `debug` | `false` | Prints concise runtime diagnostics to the ComfyUI console. |

Suggested starting range for `text_scale` is `1.50` to `2.00`. The neutral value is `1.0`.

### Krea2 Turbo Reference Sigmas (From Latent)

| Parameter | Default | Meaning |
|---|---:|---|
| `model` | — | The loaded Krea2 Turbo diffusion model. |
| `latent` | — | The same latent used for sampling; it is validated and passed through unchanged. |
| `steps` | `8` | Number of Euler denoising steps. The reference Turbo setup uses eight. |
| `denoise` | `1.0` | Uses the complete schedule at `1.0`; lower values retain the final requested steps from a longer schedule. |

### Krea2 Text Encode — Attention-Weighted Phrases

| Parameter | Meaning |
|---|---|
| `model` | The final Krea2 model chain that will be sent to the sampler, including any LoRAs. |
| `clip` | A text encoder loaded with the Krea2 CLIP type. |
| `text` | Literal prompt text with optional `(phrase:weight)` sections. |

#### Why this node exists

Krea2 does not consume a conventional single-layer CLIP embedding. Its text
encoder supplies twelve selected Qwen hidden-state taps, producing a
`12 x 2560` representation for every text-token position. Krea2 then processes
that stack through its internal text-fusion path before the text and image
tokens enter the shared DiT blocks.

Conventional prompt-emphasis methods usually multiply a completed conditioning
row or repeat a token. Those operations do not map cleanly onto this pipeline:
uniform row scaling can be reduced by later normalization, while repetition
changes sequence length and can make one term overwhelm the relationships in a
long prompt.

This node keeps the original prompt sequence intact. It encodes the clean text
normally, locates every Qwen token row belonging to each weighted phrase, and
changes how strongly image queries attend to those selected text keys inside
Krea2's shared DiT attention. It does not copy, delete, average, or rescale the
conditioning rows.

For a phrase weight `w`, the node adds `log(w)` to the selected image-to-text
attention logits. After softmax, this multiplies the selected phrase's attention
odds by `w` relative to their original values. A weight is therefore an
attention-priority control, not a promise that an object will become a literal
multiple larger, more frequent, or more visible in the final image.

#### Why it has both MODEL and CLIP inputs

The CLIP input is used to tokenize and encode the annotation-free prompt into
the normal Krea2 twelve-tap conditioning tensor. The MODEL input is used to
apply the matching attention-odds operation to the exact text-row positions
identified during that encoding. This is why the node produces a paired MODEL
and CONDITIONING result rather than acting as only a text encoder or only a
model patch.

Connect the completed model chain after all desired LoRA loaders to `model`.
Connect the Krea2 text encoder to `clip`. Send the node's MODEL output to the
sampler and its CONDITIONING output to the sampler's positive-conditioning
path. The same prompt supplies both outputs, keeping the phrase-to-row mapping
aligned with the model-side attention operation.

#### Phrase syntax

Use parentheses around any complete word or multi-word phrase followed by a
colon and a non-negative numeric weight:

```text
A scene containing a (primary subject:2.0) beside a (secondary object:0.6)
```

The node removes only the surrounding weight annotation before tokenization.
The words, spaces, tokenizer pieces, token order, and token count remain those
of the clean prompt. If a phrase becomes several Qwen tokenizer pieces, the
same weight is assigned to every piece belonging to that phrase.

Weighted sections must not overlap or contain another weighted section. More
than one separate phrase can be weighted in the same prompt.

#### Weight behavior

| Weight | Effect |
|---:|---|
| `1.0` | Exact neutral value. The phrase receives its original attention odds. |
| Above `1.0` | Gives the phrase more attention priority. |
| Between `0.0` and `1.0` | Reduces the phrase's attention priority. |
| `0.0` | Applies the node's strongest suppression to the selected phrase keys. |

Weights are relative odds multipliers. For example, `2.0` gives the selected
keys twice their original odds before softmax renormalizes all available keys;
it does not guarantee twice the visible effect. Very large weights can cause a
phrase to compete too strongly with composition, spatial relationships, or
other requested details.

#### Practical usage guide

1. Build and test the complete Krea2 workflow first, including the LoRAs and
   sampler settings you intend to use.
2. Keep the seed, resolution, sigmas, sampler, prompt, and LoRA strengths fixed
   while evaluating a phrase weight.
3. Start with the ordinary unannotated prompt or annotate the phrase with
   `1.0` to establish the neutral result.
4. Add weight only to the exact phrase that needs more or less priority. Include
   the complete relationship when the relationship matters instead of weighting
   only one isolated noun.
5. Begin with a moderate increase such as `1.5` or `2.0`. Raise it in deliberate
   increments if the phrase still receives insufficient attention. Use values
   below `1.0` when a phrase is dominating the result.
6. If several prompt sections need adjustment, tune one phrase at a time before
   combining the weights. This makes composition changes attributable to a
   specific phrase instead of several simultaneous changes.
7. Recheck the result across additional seeds only after finding a useful range
   on the fixed comparison seed. Phrase weighting changes attention allocation,
   so its visible strength can vary with the generated composition.

The `STRING` output is an inspection report. It records the clean text, every
weighted phrase, its numeric weight, and the exact Qwen token rows, token IDs,
and decoded pieces selected for that phrase. It can be connected to a text
preview node when the precise tokenizer mapping needs to be verified; it is not
required by the sampler.

This node is specifically validated for text-only Krea2 conditioning with the
`12 x 2560` layout. It rejects a mismatched text encoder, visual or custom
embedding tokens, changed token counts, and a MODEL that does not expose the
expected Krea2 text-fusion and shared-block architecture instead of silently
applying an uncertain mapping.

## Notes

- Designed for Krea2 models using the `12 x 2560` Krea2 text-conditioning layout.
- The original and advanced enhancer nodes return only a patched `MODEL`; they do not modify prompt text or require extra conditioning nodes.
- The attention-weighted phrase node must supply both the sampler's `MODEL` and positive `CONDITIONING` paths. It never copies, deletes, averages, or scales conditioning rows.
- If the loaded model does not match the expected Krea2 text-fusion layout, the patch is skipped.
- The advanced node restores every temporary runtime patch after each model call and does not store debug counters or step-local state in the model config. With the same seed and the same node parameters, ComfyUI can reuse cached graph results normally.
- The reference sigma node uses the Turbo fixed timestep shift `mu=1.15`, based on the official Krea 2 Turbo scheduler settings. It validates that the connected image dimensions are divisible by 16. Turbo does not use the RAW checkpoint's resolution-dependent shift rule.
