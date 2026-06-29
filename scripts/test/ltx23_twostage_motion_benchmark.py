#!/usr/bin/env python3
"""Run LTX-2.3 two-stage I2AV motion benchmark for one or more checkpoints.

This is a parameterized migration of:
/gemini/platform/public/aigc/human_guozz2/code/zhangyan/DiffSynth-Studio-LTX/LTX-2.3-I2AV-TwoStage-Motion.py

It supports both DiffSynth transformer-only .safetensors and OmniStream training
checkpoints such as checkpoint_model_xxxxxx/model_gen.pt. When a .pt checkpoint
is passed, it is converted once to a transformer-only .safetensors file and then
loaded by DiffSynth.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image, ImageOps
from safetensors.torch import save_file

DEFAULT_DIFFSYNTH_ROOT = Path("/gemini/platform/public/aigc/human_guozz2/code/zhangyan/DiffSynth-Studio-LTX")
DEFAULT_JSON_DIR = Path("/gemini/platform/public/aigc/human_guozz2/code/xqp/code/LLM/caption_change_motion")
DEFAULT_IMAGE_DIR = Path("/gemini/platform/public/aigc/human_guozz2/code/zhangyan/DS_LTX23/docs/VideoMotion")
DEFAULT_BASELINE_PT = Path("models/model_gen_from_step27000.pt")
DEFAULT_OURS_PT = Path(
    "ltx_experiments/ltx23_bidirectional_dmd_4nodes_512x768_121f_normalopt_seq1_bs1_cpuoffload_8step_log500/"
    "checkpoint_model_000500/model_gen.pt"
)

NEGATIVE_PROMPT = (
    "blurry, out of focus, overexposed, underexposed, low contrast, washed out colors, excessive noise, "
    "grainy texture, poor lighting, flickering, motion blur, distorted proportions, unnatural skin tones, "
    "deformed facial features, asymmetrical face, missing facial features, extra limbs, disfigured hands, "
    "wrong hand count, artifacts around text, inconsistent perspective, camera shake, incorrect depth of "
    "field, background too sharp, background clutter, distracting reflections, harsh shadows, inconsistent "
    "lighting direction, color banding, cartoonish rendering, 3D CGI look, unrealistic materials, uncanny "
    "valley effect, incorrect ethnicity, wrong gender, exaggerated expressions, wrong gaze direction, "
    "mismatched lip sync, silent or muted audio, distorted voice, robotic voice, echo, background noise, "
    "off-sync audio, incorrect dialogue, added dialogue, repetitive speech, jittery movement, awkward "
    "pauses, incorrect timing, unnatural transitions, inconsistent framing, tilted camera, flat lighting, "
    "inconsistent tone, cinematic oversaturation, stylized filters, or AI artifacts."
)


def _strip_training_prefix(key: str) -> str:
    for prefix in ("model.velocity_model.", "model.diffusion_model.", "model."):
        if key.startswith(prefix):
            return key[len(prefix):]
    return key


def _load_pt_state(path: Path, container_key: str | None) -> dict[str, torch.Tensor]:
    obj = torch.load(path, map_location="cpu", mmap=True, weights_only=False)
    if container_key:
        obj = obj[container_key]
    elif isinstance(obj, dict):
        for key in ("generator", "critic", "state_dict", "model"):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise TypeError(f"Expected state dict in {path}, got {type(obj).__name__}")
    return obj


def ensure_transformer_safetensors(model_path: Path, cache_dir: Path, container_key: str | None) -> Path:
    model_path = model_path.expanduser().resolve()
    if model_path.suffix == ".safetensors":
        return model_path
    if model_path.suffix != ".pt":
        raise ValueError(f"Unsupported model file suffix: {model_path}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    out_path = cache_dir / f"{model_path.stem}.transformer.safetensors"
    if out_path.exists() and out_path.stat().st_mtime >= model_path.stat().st_mtime:
        print(f"[convert] reuse {out_path}")
        return out_path

    print(f"[convert] {model_path} -> {out_path}")
    state = _load_pt_state(model_path, container_key)
    converted = OrderedDict()
    for key, value in state.items():
        if not torch.is_tensor(value):
            continue
        converted[_strip_training_prefix(key)] = value.detach().cpu().contiguous()
    if not converted:
        raise ValueError(f"No tensors converted from {model_path}")
    save_file(converted, out_path, metadata={"format": "pt"})
    print(f"[convert] saved {len(converted)} tensors")
    return out_path


def parse_model_arg(item: str) -> tuple[str, Path]:
    if "=" in item:
        label, path = item.split("=", 1)
    else:
        path = item
        label = Path(path).stem
    label = label.strip()
    if not label:
        raise ValueError(f"Invalid model label in {item!r}")
    return label, Path(path)


def load_prompt(json_path: Path) -> str:
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        for key in ("prompt", "audio_video_description", "text", "caption", "audiovisual_caption", "audio_content"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    raise KeyError(f"No prompt-like key found in {json_path}")


def iter_cases(json_dir: Path, image_dir: Path, limit: int | None) -> Iterable[tuple[str, Path, Path, str]]:
    count = 0
    for json_path in sorted(json_dir.glob("*.json")):
        name = json_path.stem
        image_matches = []
        for pattern in (f"*{name}.*", f"{name}.*"):
            image_matches.extend(glob.glob(str(image_dir / pattern)))
        image_matches = sorted(set(image_matches))
        if not image_matches:
            print(f"[skip] no image for {json_path}")
            continue
        try:
            prompt = load_prompt(json_path)
        except Exception as exc:
            print(f"[skip] prompt failed for {json_path}: {exc}")
            continue
        yield name, json_path, Path(image_matches[0]), prompt
        count += 1
        if limit is not None and count >= limit:
            break


def build_pipeline(model_path: Path, diffsynth_root: Path):
    if str(diffsynth_root) not in sys.path:
        sys.path.insert(0, str(diffsynth_root))
    from diffsynth.pipelines.ltx2_audio_video import LTX2AudioVideoPipeline, ModelConfig

    vram_config = {
        "offload_dtype": torch.bfloat16,
        "offload_device": "cpu",
        "onload_dtype": torch.bfloat16,
        "onload_device": "cuda",
        "preparing_dtype": torch.bfloat16,
        "preparing_device": "cuda",
        "computation_dtype": torch.bfloat16,
        "computation_device": "cuda",
    }
    return LTX2AudioVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[
            ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized", origin_file_pattern="model-*.safetensors", **vram_config),
            ModelConfig(path=str(model_path), **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="text_encoder_post_modules.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="video_vae_encoder.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="video_vae_decoder.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="audio_vae_decoder.safetensors", **vram_config),
            ModelConfig(model_id="DiffSynth-Studio/LTX-2.3-Repackage", origin_file_pattern="audio_vocoder.safetensors", **vram_config),
            ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-spatial-upscaler-x2-1.1.safetensors", **vram_config),
        ],
        tokenizer_config=ModelConfig(model_id="google/gemma-3-12b-it-qat-q4_0-unquantized"),
        stage2_lora_config=ModelConfig(model_id="Lightricks/LTX-2.3", origin_file_pattern="ltx-2.3-22b-distilled-lora-384-1.1.safetensors"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="LTX-2.3 two-stage motion benchmark for OmniStream checkpoints")
    parser.add_argument("--diffsynth-root", type=Path, default=DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument("--json-dir", type=Path, default=DEFAULT_JSON_DIR)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("ltx_experiments/test_outputs/twostage_motion_compare"))
    parser.add_argument("--cache-dir", type=Path, default=Path("ltx_experiments/converted_safetensors"))
    parser.add_argument("--model", action="append", default=None, help="label=path. May be .pt or .safetensors. Can be repeated.")
    parser.add_argument("--container-key", default="generator", help="Container key for .pt checkpoints")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1536)
    parser.add_argument("--num-frames", type=int, default=250)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-tiled", action="store_true")
    parser.add_argument("--one-stage", action="store_true", help="Disable use_two_stage_pipeline")
    args = parser.parse_args()

    model_args = args.model or [f"baseline_step27000={DEFAULT_BASELINE_PT}", f"ours_ckpt500={DEFAULT_OURS_PT}"]
    models = [parse_model_arg(item) for item in model_args]
    cases = list(iter_cases(args.json_dir, args.image_dir, args.limit))
    if not cases:
        raise RuntimeError(f"No benchmark cases found: json_dir={args.json_dir}, image_dir={args.image_dir}")
    print(f"[cases] {len(cases)}")

    if str(args.diffsynth_root) not in sys.path:
        sys.path.insert(0, str(args.diffsynth_root))
    from diffsynth.utils.data.media_io_ltx2 import write_video_audio_ltx2

    for label, model_path in models:
        transformer_path = ensure_transformer_safetensors(model_path, args.cache_dir, args.container_key)
        print(f"[model] {label}: {transformer_path}")
        pipe = build_pipeline(transformer_path, args.diffsynth_root)
        model_out_dir = args.output_dir / label
        model_out_dir.mkdir(parents=True, exist_ok=True)

        for idx, (name, json_path, image_path, prompt) in enumerate(cases):
            print(f"[{label}] {idx + 1}/{len(cases)} {name}")
            print(f"  json={json_path}")
            print(f"  image={image_path}")
            print(f"  prompt={prompt}")
            image = Image.open(image_path).convert("RGB")
            image = ImageOps.fit(image, (args.width, args.height), centering=(0.5, 0.5)).convert("RGB")
            video, audio = pipe(
                prompt=prompt,
                negative_prompt=NEGATIVE_PROMPT,
                seed=args.seed,
                height=args.height,
                width=args.width,
                num_frames=args.num_frames,
                num_inference_steps=args.num_inference_steps,
                tiled=not args.no_tiled,
                use_two_stage_pipeline=not args.one_stage,
                input_images=[image],
                input_images_indexes=[0],
                input_images_strength=1.0,
                clear_lora_before_state_two=True,
            )
            stage_name = "onestage" if args.one_stage else "twostage"
            save_path = model_out_dir / f"ltx23_{stage_name}_i2av_{name}.mp4"
            write_video_audio_ltx2(video=video, audio=audio, output_path=str(save_path), fps=args.fps)
            pipe.clear_lora()
            print(f"  saved={save_path}")

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
