import torch

import comfy.patcher_extension

from . import _bounded_float, _enhanced_txtfusion_forward, _is_krea2_dm, _step_progress


WRAPPER_KEY = "krea2t_prompt_adherence_enhancer_advanced"
CONFIG_KEY = "krea2t_prompt_adherence_enhancer_advanced"
ORIGINAL_TXTFUSION_ATTR = "_krea2t_enhancer_advanced_original_forward"
_MISSING = object()


def _install_txtmlp_scale(txtmlp_module, scale: float):
    if scale == 1.0:
        return None, None

    original_forward = txtmlp_module.forward

    def scaled_forward(y):
        return original_forward(y) * float(scale)

    txtmlp_module.forward = scaled_forward
    return txtmlp_module, original_forward


def krea2t_enhancer_advanced_wrapper(executor, *args, **kwargs):
    transformer_options = kwargs.get("transformer_options")
    if transformer_options is None:
        transformer_options = next((arg for arg in args if isinstance(arg, dict)), {})

    cfg = transformer_options.get(CONFIG_KEY, {})
    if not cfg or not cfg.get("enabled", True):
        return executor(*args, **kwargs)

    if cfg.get("_active", False):
        return executor(*args, **kwargs)

    dm = executor.class_obj
    if not _is_krea2_dm(dm):
        if cfg.get("debug", False):
            print("[Krea2TEnhancerAdvanced] skipped: diffusion model does not match Krea2 text-fusion layout")
        return executor(*args, **kwargs)

    strength = _bounded_float(cfg.get("strength", 1.0), 1.0, 0.0, 2.0)
    text_scale = _bounded_float(cfg.get("text_scale", 1.0), 1.0, 0.25, 4.0)
    if strength == 0.0 and text_scale == 1.0:
        return executor(*args, **kwargs)

    txtfusion = dm.txtfusion
    if hasattr(txtfusion, ORIGINAL_TXTFUSION_ATTR):
        txtfusion.forward = getattr(txtfusion, ORIGINAL_TXTFUSION_ATTR)
        delattr(txtfusion, ORIGINAL_TXTFUSION_ATTR)

    original_txtfusion_forward = txtfusion.forward
    txtmlp_module = None
    original_txtmlp_forward = None
    previous_active = cfg.get("_active", _MISSING)
    progress, sigma = _step_progress(transformer_options)
    debug_enabled = bool(cfg.get("debug", False))

    def enhanced_forward(x_in, mask=None, transformer_options=None):
        setattr(txtfusion, ORIGINAL_TXTFUSION_ATTR, original_txtfusion_forward)
        try:
            after, debug = _enhanced_txtfusion_forward(
                txtfusion,
                x_in,
                mask=mask,
                transformer_options=transformer_options or {},
                strength=strength,
                collect_debug=debug_enabled,
            )
        finally:
            if hasattr(txtfusion, ORIGINAL_TXTFUSION_ATTR):
                delattr(txtfusion, ORIGINAL_TXTFUSION_ATTR)

        if debug is not None:
            print(
                "[Krea2TEnhancerAdvanced] "
                f"strength={strength:.3f} text_scale={text_scale:.3f} "
                f"progress={progress:.3f} sigma={sigma:.6g} "
                f"global={debug['global_multiplier']:.6g} "
                f"proj_ratio={debug['projector_rms_ratio']:.6g} "
                f"out_rel={debug['output_rel_delta']:.6g} "
                f"out_cos={debug['output_cosine']:.6g} "
                f"clamp={debug['clamp_mean']:.6g}"
            )
        return after

    try:
        cfg["_active"] = True
        txtmlp_module, original_txtmlp_forward = _install_txtmlp_scale(dm.txtmlp, text_scale)

        if strength != 0.0:
            txtfusion.forward = enhanced_forward

        return executor(*args, **kwargs)
    finally:
        txtfusion.forward = original_txtfusion_forward
        if txtmlp_module is not None and original_txtmlp_forward is not None:
            txtmlp_module.forward = original_txtmlp_forward

        if previous_active is _MISSING:
            cfg.pop("_active", None)
        else:
            cfg["_active"] = previous_active


class Krea2TEnhancerAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "enabled": ("BOOLEAN", {"default": True}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05}),
                "text_scale": ("FLOAT", {"default": 1.0, "min": 0.25, "max": 4.0, "step": 0.05}),
                "debug": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "apply"
    CATEGORY = "conditioning/krea2"

    def apply(self, model, enabled=True, strength=1.0, text_scale=1.0, debug=False):
        patched = model.clone()
        strength = _bounded_float(strength, 1.0, 0.0, 2.0)
        text_scale = _bounded_float(text_scale, 1.0, 0.25, 4.0)
        to = patched.model_options.setdefault("transformer_options", {})
        to[CONFIG_KEY] = {
            "enabled": bool(enabled),
            "strength": strength,
            "text_scale": text_scale,
            "debug": bool(debug),
        }

        if hasattr(patched, "remove_wrappers_with_key"):
            patched.remove_wrappers_with_key(
                comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                WRAPPER_KEY,
            )

        wrappers = to.get("wrappers", {})
        diffusion_wrappers = wrappers.get(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, {})
        diffusion_wrappers.pop(WRAPPER_KEY, None)
        patched.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
            WRAPPER_KEY,
            krea2t_enhancer_advanced_wrapper,
        )

        if debug:
            print(
                "[Krea2TEnhancerAdvanced] attached "
                f"enabled={bool(enabled)} strength={strength:.3f} text_scale={text_scale:.3f}"
            )
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "Krea2T-Enhancer-Advanced": Krea2TEnhancerAdvanced,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Krea2T-Enhancer-Advanced": "Krea2T Enhancer Advanced",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
