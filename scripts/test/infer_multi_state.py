"""
Multi-state prompt inference script.

Reads entries from a JSONL file (multi_state_prompts field), runs teacher-model
causal streaming inference with a different state prompt per segment, and saves
one concatenated long video per sample.

Each JSONL entry must contain:
  - file_path              : source video (used only for logging / output naming)
  - multi_state_prompts    : list[str] — one prompt per segment
  - video_train_time       : float, seconds per segment (used to compute frame count)

Usage:
    python scripts/test/infer_multi_state.py \
        --jsonl_path /path/to/openhuman_vid_multi_state.jsonl \
        --image_path /path/to/reference_image.jpg \
        --checkpoint_path /gemini/platform/public/aigc/human_guozz2/model/LTX-2.3/ltx-2.3-22b-dev.safetensors \
        --gemma_path /gemini/platform/public/aigc/human_guozz2/model/gemma-3-12b-it-qat-q4_0-unquantized \
        --generator_ckpt_path /path/to/model_gen.pt \
        --output_dir /path/to/output \
        [--n_samples 5] \
        [--frames_per_segment 169] \
        [--resolution 480p] \
        [--dit_device_idx 0] \
        [--text_encoder_device_idx 1]
"""

import argparse
import json
import os
import sys
import datetime
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import cv2

import torch
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_distillation.inference.causal_pipeline_ltx23_stream_switch import (
    LTX23CausalAVStreamSwitchInferencePipeline,
)
from ltx_distillation.models.ltx_wrapper import create_causal_ltx2_wrapper
from ltx_distillation.models.text_encoder_wrapper import create_text_encoder_wrapper
from ltx_distillation.models.vae_wrapper import create_vae_wrappers
from ltx_core.model.video_vae.tiling import (
    TilingConfig,
    SpatialTilingConfig,
    TemporalTilingConfig,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _remap_state_dict_keys(state_dict: dict) -> dict:
    sample_keys = list(state_dict.keys())[:20]
    if any(k.startswith("model.velocity_model.") for k in sample_keys):
        return {
            k[len("model.velocity_model."):]: v
            for k, v in state_dict.items()
            if k.startswith("model.velocity_model.")
        }
    if any(k.startswith("model.") for k in sample_keys):
        return {
            k[len("model."):]: v
            for k, v in state_dict.items()
            if k.startswith("model.")
        }
    return state_dict


def add_noise(original, noise, sigma):
    if sigma.dim() == 1:
        sigma = sigma.reshape(-1, *[1] * (original.dim() - 1))
    elif sigma.dim() == 2:
        sigma = sigma.reshape(*sigma.shape, *[1] * (original.dim() - 2))
    sigma = sigma.to(dtype=original.dtype)
    return ((1 - sigma) * original + sigma * noise).to(dtype=original.dtype)


def setup_denoising_sigmas(denoising_step_list, device):
    full_sigmas = LTX2Scheduler().execute(steps=40)
    sigmas = []
    for t in denoising_step_list:
        target = t / 1000.0
        idx = (full_sigmas - target).abs().argmin().item()
        sigmas.append(full_sigmas[idx])
    return torch.stack(sigmas).to(device)


def load_models(args, dtype, device, text_encoder_device):
    generator = create_causal_ltx2_wrapper(
        checkpoint_path=args.checkpoint_path,
        gemma_path=args.gemma_path,
        device="cpu",
        dtype=dtype,
        use_flex_attention=False,
        registry=None,
    )
    print("Base generator loaded.")

    if args.generator_ckpt_path:
        print(f"Loading finetuned generator from {args.generator_ckpt_path}")
        ckpt = torch.load(args.generator_ckpt_path, map_location="cpu")
        gen_sd = ckpt.get("generator", ckpt)
        gen_sd = _remap_state_dict_keys(gen_sd)
        missing, unexpected = generator.model.velocity_model.load_state_dict(
            gen_sd, strict=False
        )
        real_missing = [k for k in missing if "mask_builder" not in k]
        if real_missing:
            print(f"  [generator] missing keys ({len(real_missing)}): {real_missing[:5]}")
        if unexpected:
            print(f"  [generator] unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    else:
        print("No finetuned generator checkpoint provided, using base weights.")

    generator.to(device)
    print("Generator moved to device.")

    text_encoder = create_text_encoder_wrapper(
        checkpoint_path=args.checkpoint_path,
        gemma_path=args.gemma_path,
        device=text_encoder_device,
        dtype=dtype,
        load_in_8bit=False,
        registry=None,
    )
    video_vae, audio_vae = create_vae_wrappers(
        checkpoint_path=args.checkpoint_path,
        device=device,
        dtype=dtype,
        registry=None,
    )
    return generator, text_encoder, video_vae, audio_vae


def build_pipeline(args, generator, text_encoder, video_vae, audio_vae,
                   denoising_sigmas, device, dtype, text_encoder_device):
    tiling_config = TilingConfig(
        spatial_config=SpatialTilingConfig(
            tile_size_in_pixels=512,
            tile_overlap_in_pixels=64,
        ),
        temporal_config=TemporalTilingConfig(
            tile_size_in_frames=24,
            tile_overlap_in_frames=8,
        ),
    )
    pipeline = LTX23CausalAVStreamSwitchInferencePipeline(
        generator=generator,
        text_encoder=text_encoder,
        video_vae=video_vae,
        audio_vae=audio_vae,
        device=device,
        dtype=dtype,
        use_kv_cache=True,
        clear_cuda_cache_per_round=True,
        add_noise_fn=add_noise,
        denoising_sigmas=denoising_sigmas,
        num_frame_per_block=3,
        num_audio_token_per_block=25,
        text_encoder_device=text_encoder_device,
        tiling_config=tiling_config,
    )
    return pipeline


def extract_first_frame(video_path: str, tmp_dir: str) -> str:
    """Extract the first frame of a video and save as a temp JPEG, return its path."""
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read first frame from {video_path}")
    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(tmp_dir, f"{stem}_frame0.jpg")
    cv2.imwrite(out_path, frame)
    return out_path


def frames_for_segment(video_train_time: float, frame_rate: int = 24) -> int:
    """Convert segment duration (seconds) to frame count aligned to 8n+1."""
    raw = int(round(video_train_time * frame_rate))
    # must satisfy (F-1) % 8 == 0
    aligned = ((raw - 1) // 8) * 8 + 1
    return max(aligned, 9)  # minimum 1 latent block = 9 frames


def build_prompt_list(multi_state_prompts, frames_per_segment, frame_rate=24,
                      num_frame_per_block=3):
    """
    Convert list of prompt strings to the format expected by
    LTX23CausalAVStreamSwitchInferencePipeline.generate():
      [{"content": str, "len": int_num_blocks}, ...]

    _parse_prompt_list treats "len" as the number of AV blocks for that segment:
      total_frame_num += round(item["len"]) * 24

    Each AV block covers num_frame_per_block latent frames.
    frames_per_segment is in pixel frames: latent_frames = (F-1)//8 + 1 (approx F/8).
    num_blocks = (latent_frames - 1) // num_frame_per_block
    Then total_frame_num = num_blocks * 24 ≈ seg pixel frames (pipeline convention).
    """
    latent_frames = (frames_per_segment - 1) // 8 + 1
    num_blocks = max(1, (latent_frames - 1) // num_frame_per_block)
    prompt_list = [{"content": p, "len": num_blocks} for p in multi_state_prompts]
    return prompt_list


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl_path", type=str,
                        default="/gemini/platform/public/aigc/human_guozz2/code/rq/prompt_generate/qwen_response/openhuman_vid_multi_state.jsonl",
                        help="Path to the multi-state JSONL file")
    parser.add_argument("--image_path", type=str, default=None,
                        help="Reference image for I2V. If not set, the first frame of each "
                             "entry's file_path video is extracted automatically.")
    parser.add_argument("--checkpoint_path", type=str,
                        default="/gemini/platform/public/aigc/human_guozz2/model/LTX-2.3/ltx-2.3-22b-dev.safetensors")
    parser.add_argument("--gemma_path", type=str,
                        default="/gemini/platform/public/aigc/human_guozz2/model/gemma-3-12b-it-qat-q4_0-unquantized")
    parser.add_argument("--generator_ckpt_path", type=str, default=None,
                        help="Path to finetuned model_gen.pt. If not set, uses base weights.")
    parser.add_argument("--output_dir", type=str,
                        default="ltx_experiments/test_output_multi_state")
    parser.add_argument("--n_samples", type=int, default=5,
                        help="Number of JSONL entries to process (0 = all)")
    parser.add_argument("--frames_per_segment", type=int, default=None,
                        help="Override frames per segment. If not set, derived from video_train_time.")
    parser.add_argument("--resolution", type=str, default="480p")
    parser.add_argument("--frame_rate", type=int, default=24)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--denoising_step_list", type=int, nargs="+",
                        default=[1000, 757, 522, 0])
    parser.add_argument("--dit_device_idx", type=int, default=0)
    parser.add_argument("--text_encoder_device_idx", type=int, default=1)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = torch.device(f"cuda:{args.dit_device_idx}")
    text_encoder_device = torch.device(f"cuda:{args.text_encoder_device_idx}")
    print(f"device={device}  text_encoder_device={text_encoder_device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ── load models ──────────────────────────────────────────────────────────
    generator, text_encoder, video_vae, audio_vae = load_models(
        args, dtype, device, text_encoder_device
    )
    denoising_sigmas = setup_denoising_sigmas(args.denoising_step_list, device)
    pipeline = build_pipeline(
        args, generator, text_encoder, video_vae, audio_vae,
        denoising_sigmas, device, dtype, text_encoder_device,
    )

    # ── read JSONL ────────────────────────────────────────────────────────────
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        entries = [json.loads(line) for line in f if line.strip()]

    if args.n_samples > 0:
        entries = entries[: args.n_samples]
    print(f"Processing {len(entries)} entries from {args.jsonl_path}")

    # ── inference loop ────────────────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, entry in enumerate(entries):
            multi_state_prompts = entry["multi_state_prompts"]
            video_train_time = float(entry.get("video_train_time", 6.5))
            src_path = entry.get("file_path", f"sample_{i}")

            if not multi_state_prompts:
                print(f"[{i}] No multi_state_prompts, skipping.")
                continue

            # resolve reference image: per-entry video first frame, or global override
            if args.image_path:
                image_path = args.image_path
            else:
                if not os.path.exists(src_path):
                    print(f"[{i}] Video not found: {src_path}, skipping.")
                    continue
                image_path = extract_first_frame(src_path, tmp_dir)

            if args.frames_per_segment is not None:
                seg_frames = args.frames_per_segment
            else:
                seg_frames = frames_for_segment(video_train_time, args.frame_rate)

            n_segs = len(multi_state_prompts)
            prompt_list = build_prompt_list(multi_state_prompts, seg_frames, args.frame_rate)

            stem = os.path.splitext(os.path.basename(src_path))[0]
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            out_name = f"{stem}_{n_segs}segs_{ts}.mp4"
            out_path = os.path.join(args.output_dir, out_name)

            print(f"\n[{i+1}/{len(entries)}] {stem}")
            print(f"  image={image_path}")
            print(f"  segments={n_segs}  seg_frames={seg_frames}  total≈{seg_frames*n_segs}f")
            for j, p in enumerate(multi_state_prompts):
                print(f"  seg{j}: {p[:80]}{'...' if len(p) > 80 else ''}")

            try:
                pipeline.generate(
                    image_path=image_path,
                    prompt_list=prompt_list,
                    video_num_frames=seg_frames,
                    resolution=args.resolution,
                    save_video=True,
                    output_dir=args.output_dir,
                    frame_rate=args.frame_rate,
                )
                # pipeline writes with its own timestamp name; rename to ours
                candidates = sorted(
                    [f for f in os.listdir(args.output_dir) if f.endswith(".mp4")],
                    key=lambda f: os.path.getmtime(os.path.join(args.output_dir, f)),
                )
                if candidates:
                    latest = os.path.join(args.output_dir, candidates[-1])
                    if latest != out_path:
                        os.rename(latest, out_path)
                print(f"  saved → {out_path}")
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()

            torch.cuda.empty_cache()

    print(f"\nDone. Results in {args.output_dir}")


if __name__ == "__main__":
    main()
