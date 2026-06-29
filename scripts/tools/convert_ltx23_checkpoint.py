#!/usr/bin/env python3
"""Convert between OmniStream training checkpoints and transformer safetensors.

Common uses:
  # training model_gen.pt -> DiffSynth/transformer-only safetensors
  python scripts/tools/convert_ltx23_checkpoint.py pt-to-safetensors \
    --input ltx_experiments/.../checkpoint_model_000500/model_gen.pt \
    --output /tmp/model_gen_000500.safetensors

  # transformer-only safetensors -> training model_gen.pt style checkpoint
  python scripts/tools/convert_ltx23_checkpoint.py safetensors-to-pt \
    --input /path/to/step-27000.safetensors \
    --output /tmp/model_gen_from_step27000.pt \
    --container-key generator \
    --prefix model.velocity_model.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file, save_file

TRAINING_PREFIXES = (
    "model.velocity_model.",
    "model.diffusion_model.",
    "model.",
)
NON_TRANSFORMER_PREFIXES = (
    "first_stage_model.",
    "text_encoder.",
    "text_encoder_2.",
    "conditioner.",
    "vae.",
    "video_vae.",
    "audio_vae.",
    "vocoder.",
    "embeddings_processor.",
)


def _load_pt_state(path: Path, container_key: str | None) -> Mapping[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if container_key:
        if not isinstance(obj, Mapping) or container_key not in obj:
            keys = list(obj.keys())[:20] if isinstance(obj, Mapping) else type(obj).__name__
            raise KeyError(f"container key {container_key!r} not found in {path}; top-level keys={keys}")
        obj = obj[container_key]
    elif isinstance(obj, Mapping):
        for key in ("generator", "critic", "state_dict", "model"):
            if key in obj and isinstance(obj[key], Mapping):
                obj = obj[key]
                break
    if not isinstance(obj, Mapping):
        raise TypeError(f"expected a state-dict mapping, got {type(obj).__name__}")
    return obj


def _strip_training_prefix(key: str) -> str | None:
    if key.startswith(NON_TRANSFORMER_PREFIXES):
        return None
    for prefix in TRAINING_PREFIXES:
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _add_prefix(key: str, prefix: str) -> str:
    for known_prefix in TRAINING_PREFIXES:
        if key.startswith(known_prefix):
            return key
    return f"{prefix}{key}"


def pt_to_safetensors(args: argparse.Namespace) -> None:
    state = _load_pt_state(Path(args.input), args.container_key)
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    dropped = 0
    for key, value in state.items():
        if not torch.is_tensor(value):
            dropped += 1
            continue
        new_key = _strip_training_prefix(key) if args.strip_prefix else key
        if new_key is None:
            dropped += 1
            continue
        tensor = value.detach().cpu().contiguous()
        converted[new_key] = tensor
    if not converted:
        raise ValueError("no tensor weights were converted")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(converted, output, metadata={"format": "pt"})
    print(f"saved {len(converted)} tensors to {output}")
    print(f"dropped {dropped} non-tensor or non-transformer entries")
    print("first keys:", list(converted.keys())[:10])


def safetensors_to_pt(args: argparse.Namespace) -> None:
    state = load_file(args.input, device="cpu")
    converted: OrderedDict[str, torch.Tensor] = OrderedDict()
    for key, value in state.items():
        new_key = _add_prefix(key, args.prefix) if args.add_prefix else key
        converted[new_key] = value.detach().cpu().contiguous()
    payload = {args.container_key: converted} if args.container_key else converted
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)
    print(f"saved {len(converted)} tensors to {output}")
    print(f"container_key={args.container_key!r}, prefix_added={args.add_prefix}, prefix={args.prefix!r}")
    print("first keys:", list(converted.keys())[:10])


def inspect_checkpoint(args: argparse.Namespace) -> None:
    path = Path(args.input)
    if path.suffix == ".safetensors":
        state = load_file(path, device="cpu")
        print(f"safetensors tensors={len(state)}")
        print("first keys:", list(state.keys())[:args.limit])
        return
    obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    print("pt type:", type(obj).__name__)
    if isinstance(obj, Mapping):
        print("top-level keys:", list(obj.keys())[:args.limit])
        for key in ("generator", "critic", "state_dict", "model"):
            if key in obj and isinstance(obj[key], Mapping):
                print(f"{key} tensors={len(obj[key])}")
                print(f"{key} first keys:", list(obj[key].keys())[:args.limit])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pt-to-safetensors", help="Convert training .pt checkpoint to transformer-only safetensors")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--container-key", default=None, help="Usually generator or critic. Auto-detects when omitted.")
    p.add_argument("--strip-prefix", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=pt_to_safetensors)

    p = sub.add_parser("safetensors-to-pt", help="Wrap transformer-only safetensors as training .pt checkpoint")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--container-key", default="generator", help="Use generator for model_gen.pt, critic for model_critic.pt")
    p.add_argument("--prefix", default="model.velocity_model.")
    p.add_argument("--add-prefix", action=argparse.BooleanOptionalAction, default=True)
    p.set_defaults(func=safetensors_to_pt)

    p = sub.add_parser("inspect", help="Print checkpoint key summary")
    p.add_argument("--input", required=True)
    p.add_argument("--limit", type=int, default=20)
    p.set_defaults(func=inspect_checkpoint)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
