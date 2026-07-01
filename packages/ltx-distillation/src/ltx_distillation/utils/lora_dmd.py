"""PEFT LoRA helpers for DMD distillation.

The helpers are intentionally opt-in. Existing full fine-tuning configs do not
activate this path unless ``lora_dmd.enabled`` is set in the config.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import torch


LORA_STATE_SUBSTRINGS = (
    "lora_A",
    "lora_B",
    "lora_embedding_A",
    "lora_embedding_B",
    "modules_to_save",
)


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def lora_dmd_config(config: Any) -> Any:
    return _cfg_get(config, "lora_dmd", None)


def lora_dmd_enabled(config: Any) -> bool:
    return bool(_cfg_get(lora_dmd_config(config), "enabled", False))


def lora_dmd_save_lora_only(config: Any) -> bool:
    cfg = lora_dmd_config(config)
    if not _cfg_get(cfg, "enabled", False):
        return False
    return bool(_cfg_get(cfg, "save_lora_only", True))


def _role_config(config: Any, role: str) -> Any:
    cfg = lora_dmd_config(config)
    return _cfg_get(cfg, role, None)


def _get_transformer_slot(wrapper: torch.nn.Module) -> Tuple[torch.nn.Module, str | None]:
    """Return the module PEFT should wrap and the attribute path to write back."""
    model = getattr(wrapper, "model", None)
    if model is not None and hasattr(model, "velocity_model"):
        return model.velocity_model, "model.velocity_model"
    if model is not None:
        return model, "model"
    return wrapper, None


def _set_transformer_slot(wrapper: torch.nn.Module, slot: str | None, module: torch.nn.Module) -> None:
    if slot == "model.velocity_model":
        wrapper.model.velocity_model = module
    elif slot == "model":
        wrapper.model = module
    elif slot is None:
        raise ValueError("Cannot replace the root wrapper with a PEFT module in-place")
    else:
        raise ValueError(f"Unknown LoRA target slot: {slot}")


def count_trainable_parameters(module: torch.nn.Module) -> tuple[int, int]:
    trainable = sum(p.numel() for p in module.parameters() if p.requires_grad)
    total = sum(p.numel() for p in module.parameters())
    return trainable, total


def apply_lora_to_wrapper(wrapper: torch.nn.Module, role_cfg: Any, role: str) -> None:
    from peft import LoraConfig, get_peft_model

    target_modules = _as_list(_cfg_get(role_cfg, "target_modules", None))
    if not target_modules:
        raise ValueError(f"lora_dmd.{role}.target_modules must be non-empty")

    rank = int(_cfg_get(role_cfg, "rank", _cfg_get(role_cfg, "r", 32)))
    alpha = int(_cfg_get(role_cfg, "alpha", rank))
    dropout = float(_cfg_get(role_cfg, "dropout", 0.0))

    target, slot = _get_transformer_slot(wrapper)
    target.requires_grad_(False)
    try:
        param_dtype = next(p.dtype for p in target.parameters() if p.is_floating_point())
    except StopIteration:
        param_dtype = torch.float32
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        init_lora_weights=True,
    )
    peft_target = get_peft_model(target, lora_config)
    for param in peft_target.parameters():
        if param.is_floating_point() and param.dtype != param_dtype:
            param.data = param.data.to(dtype=param_dtype)
            if param.grad is not None:
                param.grad.data = param.grad.data.to(dtype=param_dtype)
    _set_transformer_slot(wrapper, slot, peft_target)

    trainable, total = count_trainable_parameters(wrapper)
    trainable_dtypes = sorted({str(p.dtype) for p in wrapper.parameters() if p.requires_grad})
    print(
        f"[LoRA-DMD] {role}: rank={rank} alpha={alpha} dropout={dropout} "
        f"target_modules={target_modules} trainable={trainable:,}/{total:,} "
        f"trainable_dtypes={trainable_dtypes}"
    )


def apply_lora_dmd(model: Any, config: Any) -> None:
    if not lora_dmd_enabled(config):
        return

    for role in ("generator", "fake_score"):
        role_cfg = _role_config(config, role)
        if role_cfg is None:
            raise ValueError(f"lora_dmd.enabled=true requires lora_dmd.{role}")
        apply_lora_to_wrapper(getattr(model, role), role_cfg, role)

    model.real_score.requires_grad_(False)
    model.real_score.eval()
    trainable, total = count_trainable_parameters(model.real_score)
    print(f"[LoRA-DMD] real_score frozen: trainable={trainable:,}/{total:,}")


def filter_lora_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in state_dict.items()
        if any(marker in key for marker in LORA_STATE_SUBSTRINGS)
    }


def _unwrap_lora_container(state: Dict[str, Any], role: str) -> Dict[str, torch.Tensor]:
    for key in (f"{role}_lora", role, "state_dict", "lora"):
        value = state.get(key)
        if isinstance(value, dict):
            return value
    return state


def load_lora_state_dict(module: torch.nn.Module, state: Dict[str, Any], role: str, strict: bool = False) -> None:
    lora_state = filter_lora_state_dict(_unwrap_lora_container(state, role))
    if not lora_state:
        raise ValueError(f"No LoRA tensors found for {role}")
    missing, unexpected = module.load_state_dict(lora_state, strict=strict)
    lora_missing = [k for k in missing if any(marker in k for marker in LORA_STATE_SUBSTRINGS)]
    if lora_missing or unexpected:
        print(
            f"[LoRA-DMD] {role} load: lora_missing={len(lora_missing)} "
            f"unexpected={len(unexpected)}"
        )
    print(f"[LoRA-DMD] Loaded {role} LoRA tensors: {len(lora_state)}")
