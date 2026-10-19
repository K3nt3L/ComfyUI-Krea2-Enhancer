from __future__ import annotations

import math

import torch


NODE_NAME = "Krea2TurboReferenceSigmaSchedulerFromLatent"
TURBO_SHIFT = 1.15
VAE_SCALE = 8
DIT_PATCH_SIZE = 2
PIXEL_ALIGNMENT = VAE_SCALE * DIT_PATCH_SIZE


def _latent_resolution(latent) -> tuple[int, int]:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise TypeError("latent must be a ComfyUI LATENT containing samples")

    samples = latent["samples"]
    if not torch.is_tensor(samples) or samples.ndim != 4:
        shape = getattr(samples, "shape", None)
        raise TypeError(f"Krea2 requires a 4D image latent; received shape {shape}")

    height = int(samples.shape[-2]) * VAE_SCALE
    width = int(samples.shape[-1]) * VAE_SCALE
    if width % PIXEL_ALIGNMENT or height % PIXEL_ALIGNMENT:
        raise ValueError(
            f"Krea2 requires width and height divisible by {PIXEL_ALIGNMENT}; "
            f"this latent represents {width}x{height}. Change the Empty Latent "
            "Image dimensions before sampling."
        )
    return width, height


def _validate_turbo_model(model) -> None:
    base_model = model.model
    if type(base_model).__name__ != "Krea2":
        raise TypeError(
            "This scheduler requires a ComfyUI Krea2 MODEL loaded from the "
            "Krea 2 Turbo checkpoint."
        )

    model_sampling = model.get_model_object("model_sampling")
    shift = float(model_sampling.shift)
    if not math.isclose(shift, TURBO_SHIFT, rel_tol=0.0, abs_tol=1e-7):
        raise ValueError(
            f"Krea 2 Turbo requires the fixed timestep shift {TURBO_SHIFT}; "
            f"the connected MODEL currently uses {shift}. Remove the conflicting "
            "model-sampling shift node."
        )


def _reference_turbo_sigmas(steps: int, denoise: float) -> torch.Tensor:
    if denoise <= 0.0:
        return torch.empty(0, dtype=torch.float32)

    total_steps = steps if denoise >= 1.0 else int(steps / denoise)
    timesteps = torch.linspace(1.0, 0.0, total_steps + 1, dtype=torch.float64)
    exp_shift = math.exp(TURBO_SHIFT)
    sigmas = (exp_shift * timesteps) / (1.0 + (exp_shift - 1.0) * timesteps)
    return sigmas[-(steps + 1) :].to(dtype=torch.float32)


class Krea2TurboReferenceSigmaSchedulerFromLatent:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latent": (
                    "LATENT",
                    {
                        "tooltip": (
                            "Connect the same Empty Latent Image used by the sampler. "
                            "The node validates the actual canvas alignment."
                        )
                    },
                ),
                "steps": (
                    "INT",
                    {
                        "default": 8,
                        "min": 1,
                        "max": 10000,
                        "tooltip": "The reference Krea 2 Turbo setup uses 8 steps.",
                    },
                ),
                "denoise": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.0,
                        "max": 1.0,
                        "step": 0.01,
                        "tooltip": (
                            "1.0 uses the complete reference schedule. Lower values keep "
                            "the final requested number of steps from a longer schedule."
                        ),
                    },
                ),
            }
        }

    RETURN_TYPES = ("SIGMAS", "LATENT")
    RETURN_NAMES = ("sigmas", "latent")
    FUNCTION = "get_sigmas"
    CATEGORY = "model/sampling/schedulers"
    DESCRIPTION = (
        "Creates an Euler sigma schedule based on the official Krea 2 Turbo scheduler "
        "settings with fixed mu=1.15, validates that the connected latent is aligned "
        "to the model's 16-pixel grid, and passes that exact latent onward. Turbo's "
        "shift is fixed; the RAW checkpoint uses a resolution-dependent rule."
    )

    def get_sigmas(self, model, latent, steps, denoise):
        _validate_turbo_model(model)
        _latent_resolution(latent)
        sigmas = _reference_turbo_sigmas(int(steps), float(denoise))
        return sigmas, latent


NODE_CLASS_MAPPINGS = {
    NODE_NAME: Krea2TurboReferenceSigmaSchedulerFromLatent,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    NODE_NAME: "Krea2 Turbo Reference Sigmas (From Latent)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
