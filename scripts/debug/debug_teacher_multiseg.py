"""
Standalone debug script for multi-segment teacher inference.

Mirrors the training loop in LTXDMDTrainer.fwdbwd_one_step / generator_loss_segment:
  - Segment 0: causal rollout (no KV prefix) → KL grad from teacher
  - Segment 1: causal rollout with KV prefix from seg0 → KL grad from teacher

Supports either single-GPU or two-GPU placement:
  - Single GPU (default): everything on --device.
  - Two GPUs: pass --device_teacher cuda:1 to place real_score + fake_score on
    a separate device, while generator + VAEs + text_encoder stay on --device.
    KL inputs are moved across the bus only at the Step 2 boundary.

No distributed setup required.

Usage:
    # single GPU
    python scripts/debug/debug_teacher_multiseg.py \
        --config ltx_experiments/0519_5nodes_bs200_causaldmd/ltx23_causal_dmd.yaml \
        [--device cuda:0] [--n_segments 2] [--fixed_exit_step 1]

    # two GPUs (recommended when single card OOMs)
    python scripts/debug/debug_teacher_multiseg.py \
        --config ltx_experiments/0519_5nodes_bs200_causaldmd/ltx23_causal_dmd.yaml \
        --device cuda:0 --device_teacher cuda:1
"""

import argparse
import sys
import os
import math

# Make packages importable when running from project root
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from dataclasses import replace
from einops import rearrange

from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_core.components.patchifiers import get_pixel_coords
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
from ltx_pipelines.utils.types import PipelineComponents

from ltx_distillation.models.causal_dmd_ltx23 import CausalLTX23DMD
from ltx_causal.attention.mask_builder import compute_aligned_audio_frames


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_batch(batch_size: int, height: int, width: int,
                     num_frames: int, device: str, dtype: torch.dtype):
    """Create a synthetic pixel batch that looks like ImageVideoAudioDataset output."""
    # pixel_values: [B, F, C, H, W], float32 in [0,1]
    pixel_values = torch.rand(batch_size, num_frames, 3, height, width,
                              device=device, dtype=torch.float32)
    # audio_data: [B, 2, samples] at 16kHz, ~num_frames/24 seconds (stereo)
    audio_samples = int(math.ceil(num_frames / 24.0 * 16000))
    audio_data = torch.randn(batch_size, 2, audio_samples, device=device, dtype=torch.float32)
    return {
        "pixel_values": pixel_values,
        "audio_data": audio_data,
        "text": ["a person talking"] * batch_size,
        "file_path": ["debug_sample.mp4"] * batch_size,
    }


def adapt_batch_single_gpu(
    batch: dict,
    model: CausalLTX23DMD,
    num_training_latents: int,
    frame_rate: int,
    device: torch.device,
    dtype: torch.dtype,
    prev_video_output=None,
    prompt_override=None,
    enable_offload: bool = False,
):
    """
    Encode one batch into LatentState dicts.
    Mirrors LTXDMDTrainer.adapt_batch but without the accelerator dependency.
    """
    from ltx_core.types import Audio
    B, F, C, H, W = batch["pixel_values"].shape

    # Encode audio
    audio_data = Audio(batch["audio_data"].to(device, dtype), sampling_rate=16000)
    with torch.no_grad():
        if enable_offload:
            model.audio_vae = model.audio_vae.to(device=device)
        audio_clean_latent = model.audio_vae.encode(audio_data.to(device=device))
        if enable_offload:
            model.audio_vae = model.audio_vae.to(device="cpu")
            torch.cuda.empty_cache()
    audio_clean_latent = rearrange(audio_clean_latent, "b c t f -> b t (c f)")

    # Encode video
    with torch.no_grad():
        if enable_offload:
            model.video_vae = model.video_vae.to(device=device)
        video_clean_latent = model.video_vae.encode(
            rearrange(batch["pixel_values"], "b f c h w -> b c f h w").to(device=device, dtype=dtype))
        if enable_offload:
            model.video_vae = model.video_vae.to(device="cpu")
    torch.cuda.empty_cache()

    B2, C2, F2, H2, W2 = video_clean_latent.shape
    token_per_frame = H2 * W2
    video_clean_latent = rearrange(video_clean_latent, "b c t h w -> b (t h w) c")

    latent_num_frames = num_training_latents
    num_frames_out = (latent_num_frames - 1) * 8 + 1
    seq_len = latent_num_frames * token_per_frame

    if prev_video_output is not None:
        prev_last_frame = prev_video_output[:, -token_per_frame:].detach()
        video_clean_latent = torch.cat([
            prev_last_frame,
            torch.randn([B2, seq_len - token_per_frame, video_clean_latent.shape[2]],
                        dtype=dtype, device=device),
        ], dim=1)
    else:
        video_clean_latent = torch.cat([
            video_clean_latent[:, :token_per_frame],
            torch.randn([B2, seq_len - token_per_frame, video_clean_latent.shape[2]],
                        dtype=dtype, device=device),
        ], dim=1)

    # Clip / randomise audio
    video_pixel_shape = VideoPixelShape(batch=B2, frames=num_frames_out,
                                        width=W, height=H, fps=frame_rate)
    audio_latent_shape = AudioLatentShape.from_video_pixel_shape(video_pixel_shape)
    audio_clean_latent = torch.randn(
        [B2, audio_latent_shape.frames, audio_clean_latent.shape[2]], dtype=dtype, device=device)

    # Text encode
    prompts = (
        prompt_override if isinstance(prompt_override, list)
        else [prompt_override] * B2 if prompt_override is not None
        else batch["text"]
    )
    v_ctx_p_list, a_ctx_p_list, v_ctx_n_list, a_ctx_n_list = [], [], [], []
    with torch.no_grad():
        if enable_offload:
            model.text_encoder = model.text_encoder.to(device=device)
        for prompt in prompts:
            ctx_p, ctx_n = model.text_encoder([prompt, DEFAULT_NEGATIVE_PROMPT], device=device)
            v_ctx_p_list.append(ctx_p.video_encoding)
            a_ctx_p_list.append(ctx_p.audio_encoding)
            v_ctx_n_list.append(ctx_n.video_encoding)
            a_ctx_n_list.append(ctx_n.audio_encoding)
        if enable_offload:
            model.text_encoder = model.text_encoder.to(device="cpu")
            torch.cuda.empty_cache()
    v_context_p = torch.cat(v_ctx_p_list, dim=0)
    a_context_p = torch.cat(a_ctx_p_list, dim=0)
    v_context_n = torch.cat(v_ctx_n_list, dim=0)
    a_context_n = torch.cat(a_ctx_n_list, dim=0)

    # Noisy latents
    torch_rng = torch.Generator(device=device).manual_seed(42)
    noisy_video_latent = torch.randn(video_clean_latent.shape, dtype=dtype,
                                     device=device, generator=torch_rng)
    noisy_audio_latent = torch.randn(audio_clean_latent.shape, dtype=dtype,
                                     device=device, generator=torch_rng)

    # Denoise mask
    video_denoise_mask = torch.ones(video_clean_latent.shape[:2] + (1,), device=device, dtype=torch.float32)
    video_denoise_mask[:, :token_per_frame] = 0
    audio_denoise_mask = torch.ones(audio_clean_latent.shape[:2] + (1,), device=device, dtype=torch.float32)

    # Positions
    components = PipelineComponents(dtype=dtype, device=device)
    video_latent_shape = VideoLatentShape.from_pixel_shape(
        shape=video_pixel_shape,
        latent_channels=components.video_latent_channels,
        scale_factors=components.video_scale_factors,
    )
    video_latent_coords = components.video_patchifier.get_patch_grid_bounds(
        output_shape=video_latent_shape, device=device)
    video_positions = get_pixel_coords(
        latent_coords=video_latent_coords,
        scale_factors=components.video_scale_factors,
        causal_fix=True,
    ).float()
    video_positions[:, 0, ...] = video_positions[:, 0, ...] / frame_rate

    audio_latent_coords = components.audio_patchifier.get_patch_grid_bounds(
        output_shape=audio_latent_shape, device=device)
    audio_positions = audio_latent_coords

    with torch.no_grad():
        video_state = LatentState(
            latent=noisy_video_latent.detach(),
            denoise_mask=video_denoise_mask.detach(),
            positions=video_positions.detach(),
            clean_latent=video_clean_latent.detach(),
            attention_mask=None,
        )
        audio_state = LatentState(
            latent=noisy_audio_latent.detach(),
            denoise_mask=audio_denoise_mask.detach(),
            positions=audio_positions.detach(),
            clean_latent=audio_clean_latent.detach(),
            attention_mask=None,
        )
        conditional_dict = {
            "v_context": v_context_p.detach().to(device),
            "a_context": a_context_p.detach().to(device),
        }
        unconditional_dict = {
            "v_context": v_context_n.detach().to(device),
            "a_context": a_context_n.detach().to(device),
        }

    return {
        "initial_video_latent_state": video_state,
        "initial_audio_latent_state": audio_state,
        "conditional_dict": conditional_dict,
        "unconditional_dict": unconditional_dict,
        "video_latent_num_frames": latent_num_frames,
        "video_num_frames": num_frames_out,
        "width": W,
        "height": H,
        "token_per_frame": token_per_frame,
    }


def log_tensor_stats(name: str, t: torch.Tensor):
    if t is None:
        print(f"  {name}: None")
        return
    has_nan = torch.isnan(t).any().item()
    has_inf = torch.isinf(t).any().item()
    print(f"  {name}: shape={list(t.shape)}  "
          f"min={t.float().min():.4f}  max={t.float().max():.4f}  "
          f"mean={t.float().mean():.4f}  nan={has_nan}  inf={has_inf}")


# ---------------------------------------------------------------------------
# Main debug loop
# ---------------------------------------------------------------------------

def run_debug(config_path: str, device_str: str, n_segments: int, fixed_exit_step: int,
              enable_offload: bool = False, device_teacher_str: str = None,
              height_override: int = None, width_override: int = None,
              num_frames_override: int = None, num_training_latents_override: int = None):
    device_gen = torch.device(device_str)
    device_teacher = torch.device(device_teacher_str) if device_teacher_str else device_gen
    multi_gpu = (device_gen != device_teacher)

    print(f"\n{'='*60}")
    print(f"  Debug: teacher multi-segment inference")
    print(f"  config:          {config_path}")
    print(f"  device_gen:      {device_gen}")
    print(f"  device_teacher:  {device_teacher}")
    print(f"  multi_gpu:       {multi_gpu}")
    print(f"  n_segments:      {n_segments}")
    print(f"  exit_step:       {fixed_exit_step}")
    print(f"  offload:         {enable_offload}")
    print(f"{'='*60}\n")

    if multi_gpu and enable_offload:
        raise ValueError("--offload and --device_teacher are mutually exclusive. "
                         "Use --device_teacher for two-GPU mode instead of offloading.")

    config = OmegaConf.load(config_path)
    device = device_gen  # alias used by adapt_batch and the segment loop
    dtype = torch.bfloat16 if getattr(config, "mixed_precision", True) else torch.float32

    # -----------------------------------------------------------------
    # Load model (generator + teacher only; skip fake_score if desired)
    # -----------------------------------------------------------------
    print("[1] Initialising CausalLTX23DMD model …")

    model = CausalLTX23DMD(config, device=device_gen, accelerator=None)
    model.init_models()

    if multi_gpu:
        # Generator + VAEs + text_encoder on device_gen
        model.generator = model.generator.to(device=device_gen)
        model.text_encoder = model.text_encoder.to(device=device_gen)
        model.video_vae = model.video_vae.to(device=device_gen)
        model.audio_vae = model.audio_vae.to(device=device_gen)
        # Teacher models on device_teacher
        model.real_score = model.real_score.to(device=device_teacher)
        model.fake_score = model.fake_score.to(device=device_teacher)
        print(f"  [Multi-GPU] generator/VAE/text on {device_gen}, "
              f"real_score/fake_score on {device_teacher}")
    elif enable_offload:
        print("  [Offload Mode] Keeping models on CPU, will move to GPU on-demand")
        model.generator = model.generator.to(device="cpu")
        model.real_score = model.real_score.to(device="cpu")
        model.fake_score = model.fake_score.to(device="cpu")
        model.text_encoder = model.text_encoder.to(device="cpu")
        model.video_vae = model.video_vae.to(device="cpu")
        model.audio_vae = model.audio_vae.to(device="cpu")
    else:
        model.generator = model.generator.to(device=device)
        model.real_score = model.real_score.to(device=device)
        model.fake_score = model.fake_score.to(device=device)
        model.text_encoder = model.text_encoder.to(device=device)
        model.video_vae = model.video_vae.to(device=device)
        model.audio_vae = model.audio_vae.to(device=device)

    model.eval()
    model.generator.train()  # keep generator in train mode as in real training
    model.generator.enable_gradient_checkpointing()
    print("  Model loaded.\n")

    # Init causal inference pipeline
    model._initialize_causal_inference_pipeline()

    num_training_latents = num_training_latents_override or int(getattr(config, "video_num_training_latents", 22))
    frame_rate = int(getattr(config, "frame_rate", 24))
    batch_size = 1
    height = height_override or int(getattr(config, "video_height", 512))
    width = width_override or int(getattr(config, "video_width", 768))
    num_frames = num_frames_override or int(getattr(config, "video_sample_n_frames", 169))

    # -----------------------------------------------------------------
    # Multi-segment loop
    # -----------------------------------------------------------------
    seq_state = None   # carries kv_cache, prev_video_output, segment_idx, etc.

    for seg_idx in range(n_segments):
        print(f"\n{'─'*50}")
        print(f"  === SEGMENT {seg_idx} ===")
        print(f"{'─'*50}")

        prev_video_output = seq_state["video_output"] if seq_state else None
        persistent_kv = seq_state["kv_cache_list"] if seq_state else None
        segment_video_offset = seq_state["video_offset"] if seq_state else 0
        prev_video_seqlen_frame = seq_state["video_seqlen_frame"] if seq_state else None
        shared_exit_step = seq_state["exit_step"] if seq_state else fixed_exit_step

        print(f"  segment_video_offset = {segment_video_offset}")
        print(f"  prev_video_seqlen_frame = {prev_video_seqlen_frame}")
        print(f"  shared_exit_step = {shared_exit_step}")
        print(f"  persistent_kv = {'yes' if persistent_kv is not None else 'no'}")

        # --- Build batch ---
        batch = _make_fake_batch(batch_size, height, width, num_frames, device_str, dtype)
        if seg_idx > 0 and seq_state and "seg0_batch" in seq_state:
            batch = seq_state["seg0_batch"]  # pin resolution across segments

        new_batch = adapt_batch_single_gpu(
            batch=batch,
            model=model,
            num_training_latents=num_training_latents,
            frame_rate=frame_rate,
            device=device,
            dtype=dtype,
            prev_video_output=prev_video_output,
            enable_offload=enable_offload,
        )

        video_state = new_batch["initial_video_latent_state"]
        audio_state = new_batch["initial_audio_latent_state"]
        conditional_dict = new_batch["conditional_dict"]
        unconditional_dict = new_batch["unconditional_dict"]
        video_latent_num_frames = new_batch["video_latent_num_frames"]
        token_per_frame = new_batch["token_per_frame"]

        print(f"\n  Batch shapes:")
        log_tensor_stats("video_state.latent", video_state.latent)
        log_tensor_stats("audio_state.latent", audio_state.latent)
        log_tensor_stats("video_state.denoise_mask", video_state.denoise_mask)
        log_tensor_stats("video_state.positions", video_state.positions)
        log_tensor_stats("audio_state.positions", audio_state.positions)

        # -----------------------------------------------------------------
        # Step 1: Causal rollout (no_grad, capture replay_states)
        # -----------------------------------------------------------------
        print(f"\n  [Step1] Causal rollout (generator) …")
        if enable_offload:
            model.generator = model.generator.to(device=device)
        with torch.no_grad():
            pred_video, pred_audio, rollout_log, updated_kv, replay_states = \
                model.causal_inference_pipeline.inference_with_persistent_kv_cache(
                    video_latent_state=video_state,
                    audio_latent_state=audio_state,
                    video_latent_num_frames=video_latent_num_frames,
                    text_context_dict=conditional_dict,
                    persistent_kv_cache_list=persistent_kv,
                    segment_video_offset=segment_video_offset,
                    compute_grad=False,
                    prev_video_seqlen_frame=prev_video_seqlen_frame,
                    return_replay_state=True,
                    shared_exit_step=shared_exit_step,
                )

        print(f"  rollout_log: {rollout_log}")
        log_tensor_stats("pred_video", pred_video)
        log_tensor_stats("pred_audio", pred_audio)
        print(f"  replay_states count: {len(replay_states)}")
        for rs in replay_states:
            print(f"    block {rs['block_idx']}: "
                  f"v[{rs['video_start']}:{rs['video_end']}] "
                  f"a[{rs['audio_start']}:{rs['audio_end']}]")

        # Check KV cache sizes
        if updated_kv:
            kv0 = updated_kv[0]
            for kv_name, kv_val in kv0.items():
                if isinstance(kv_val, dict):
                    k_t = kv_val.get("k")
                    if k_t is not None:
                        if isinstance(k_t, torch.Tensor):
                            print(f"  kv_cache[0][{kv_name}][k]: {list(k_t.shape)}")
                        elif isinstance(k_t, list):
                            print(f"  kv_cache[0][{kv_name}][k]: list of {len(k_t)}, first shape={list(k_t[0].shape) if k_t and isinstance(k_t[0], torch.Tensor) else type(k_t[0])}")
                        else:
                            print(f"  kv_cache[0][{kv_name}][k]: {type(k_t)}")

        # Offload KV cache to CPU to save GPU memory (updated_kv can be large)
        def _tensor_to(obj, tgt_device):
            """Recursively move tensors in nested list/dict to tgt_device."""
            if isinstance(obj, torch.Tensor):
                return obj.to(device=tgt_device, non_blocking=True)
            elif isinstance(obj, list):
                return [_tensor_to(x, tgt_device) for x in obj]
            elif isinstance(obj, dict):
                return {k: _tensor_to(v, tgt_device) for k, v in obj.items()}
            return obj

        if enable_offload and updated_kv is not None:
            print("  [Offload] Moving KV cache to CPU …")
            updated_kv = _tensor_to(updated_kv, "cpu")
            torch.cuda.empty_cache()

        # Offload replay_states tensors to CPU — each Modality holds latent +
        # positions + context on GPU; 7 blocks × those tensors is a lot of VRAM.
        def _modality_to_cpu(m):
            """Return a new Modality-like object with all tensors on CPU."""
            from ltx_core.model.transformer.modality import Modality
            return Modality(
                latent=m.latent.cpu() if m.latent is not None else None,
                sigma=m.sigma.cpu() if m.sigma is not None else None,
                timesteps=m.timesteps.cpu() if m.timesteps is not None else None,
                positions=m.positions.cpu() if m.positions is not None else None,
                context=m.context.cpu() if m.context is not None else None,
                enabled=m.enabled,
                context_mask=m.context_mask.cpu() if m.context_mask is not None else None,
                attention_mask=m.attention_mask.cpu() if m.attention_mask is not None else None,
            )

        def _modality_to_device(m, tgt):
            """Return a new Modality with all tensors on tgt device."""
            from ltx_core.model.transformer.modality import Modality
            return Modality(
                latent=m.latent.to(device=tgt, non_blocking=True) if m.latent is not None else None,
                sigma=m.sigma.to(device=tgt, non_blocking=True) if m.sigma is not None else None,
                timesteps=m.timesteps.to(device=tgt, non_blocking=True) if m.timesteps is not None else None,
                positions=m.positions.to(device=tgt, non_blocking=True) if m.positions is not None else None,
                context=m.context.to(device=tgt, non_blocking=True) if m.context is not None else None,
                enabled=m.enabled,
                context_mask=m.context_mask.to(device=tgt, non_blocking=True) if m.context_mask is not None else None,
                attention_mask=m.attention_mask.to(device=tgt, non_blocking=True) if m.attention_mask is not None else None,
            )

        if enable_offload and replay_states:
            print("  [Offload] Moving replay_states tensors to CPU …")
            for rs in replay_states:
                rs["video_modality"] = _modality_to_cpu(rs["video_modality"])
                rs["audio_modality"] = _modality_to_cpu(rs["audio_modality"])
                if rs.get("first_block_denoise_mask") is not None:
                    rs["first_block_denoise_mask"] = rs["first_block_denoise_mask"].cpu()
                if rs.get("first_block_clean_latent") is not None:
                    rs["first_block_clean_latent"] = rs["first_block_clean_latent"].cpu()
            torch.cuda.empty_cache()

        # -----------------------------------------------------------------
        # Step 2: Teacher KL gradient on full predicted segment
        # -----------------------------------------------------------------
        print(f"\n  [Step2] Teacher KL grad (real_score + fake_score) …")
        if enable_offload:
            model.generator = model.generator.to(device="cpu")
            torch.cuda.empty_cache()
            model.real_score = model.real_score.to(device=device)
            model.fake_score = model.fake_score.to(device=device)

        B = video_state.clean_latent.shape[0]
        video_state_for_kl = replace(video_state, latent=pred_video.detach())
        audio_state_for_kl = replace(audio_state, latent=pred_audio.detach())

        # In multi-GPU mode, temporarily point model.device to teacher GPU so
        # _prepare_modality_from_state routes tensors to the correct device.
        if multi_gpu:
            model.device = device_teacher

        with torch.no_grad():
            # sigma_lookup buffer lives on device_gen, so compute sigma there first
            sampled_timestep = torch.randint(
                model.min_step, model.max_step + 1, (B,),
                device=device_gen, dtype=torch.long,
            )
            sigma = model.timestep_to_sigma(sampled_timestep)
            if multi_gpu:
                sigma = sigma.to(device=device_teacher)
            print(f"  DMD sigma = {sigma.tolist()}")

            grad_video, grad_audio, kl_log = model._compute_kl_grad(
                video_state=video_state_for_kl,
                audio_state=audio_state_for_kl,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                video_latent_num_frames=video_latent_num_frames,
                sigma=sigma,
                video_loss_mask=video_state.denoise_mask,
                audio_loss_mask=audio_state.denoise_mask,
            )

        # Restore model.device and move gradients back to generator device
        if multi_gpu:
            model.device = device_gen
            grad_video = grad_video.to(device=device_gen)
            grad_audio = grad_audio.to(device=device_gen)

        if enable_offload:
            model.real_score = model.real_score.to(device="cpu")
            model.fake_score = model.fake_score.to(device="cpu")
            torch.cuda.empty_cache()

        log_tensor_stats("grad_video", grad_video)
        log_tensor_stats("grad_audio", grad_audio)
        for k in ("dmdtrain_gradient_norm_video", "dmdtrain_gradient_norm_audio",
                  "dmdtrain_noise_sigma"):
            print(f"  {k}: {kl_log.get(k)}")

        # -----------------------------------------------------------------
        # Step 3: Per-block replay forward (autograd) + surrogate loss check
        # -----------------------------------------------------------------
        print(f"\n  [Step3] Per-block replay forward + surrogate loss …")
        if enable_offload:
            # Load generator back to GPU for replay forward
            model.generator = model.generator.to(device=device)

        video_loss_mask_bool = video_state.denoise_mask.bool().squeeze(-1)
        audio_loss_mask_bool = audio_state.denoise_mask.bool().squeeze(-1)
        video_active_total = video_loss_mask_bool.sum().clamp_min(1).item()
        audio_active_total = audio_loss_mask_bool.sum().clamp_min(1).item()
        video_feat_dim = pred_video.shape[-1]
        audio_feat_dim = pred_audio.shape[-1]
        video_denom = float(video_active_total * video_feat_dim)
        audio_denom = float(audio_active_total * audio_feat_dim)

        for rs in replay_states:
            v_s, v_e = rs["video_start"], rs["video_end"]
            a_s, a_e = rs["audio_start"], rs["audio_end"]
            video_frame_tokens = pred_video.shape[1] // video_latent_num_frames
            v_tok_s = v_s * video_frame_tokens
            v_tok_e = v_e * video_frame_tokens

            v_mask_blk = video_loss_mask_bool[:, v_tok_s:v_tok_e]
            a_mask_blk = audio_loss_mask_bool[:, a_s:a_e]
            if not v_mask_blk.any() and not a_mask_blk.any():
                print(f"    block {rs['block_idx']}: skipped (mask all False)")
                continue

            grad_video_blk = grad_video[:, v_tok_s:v_tok_e].detach()
            grad_audio_blk = grad_audio[:, a_s:a_e].detach()

            # Move this block's replay tensors back to GPU before forward pass
            if enable_offload:
                rs["video_modality"] = _modality_to_device(rs["video_modality"], device)
                rs["audio_modality"] = _modality_to_device(rs["audio_modality"], device)
                if rs.get("first_block_denoise_mask") is not None:
                    rs["first_block_denoise_mask"] = rs["first_block_denoise_mask"].to(device=device, non_blocking=True)
                if rs.get("first_block_clean_latent") is not None:
                    rs["first_block_clean_latent"] = rs["first_block_clean_latent"].to(device=device, non_blocking=True)

            pred_video_blk, pred_audio_blk = \
                model.causal_inference_pipeline.replay_block_exit_forward(rs)

            log_tensor_stats(f"    block{rs['block_idx']}/pred_video_blk", pred_video_blk)
            log_tensor_stats(f"    block{rs['block_idx']}/pred_audio_blk", pred_audio_blk)

            # Compute surrogate loss (mirroring generator_loss_segment step3)
            block_loss = torch.zeros((), device=device, dtype=pred_video_blk.dtype)
            if v_mask_blk.any():
                target_v = (pred_video_blk.double() - grad_video_blk.double()).detach()
                diff_v = (pred_video_blk.double() - target_v) ** 2
                diff_v = diff_v * v_mask_blk.unsqueeze(-1).double()
                loss_v = 0.5 * diff_v.sum() / video_denom
                block_loss = block_loss + loss_v.to(block_loss.dtype)
            if a_mask_blk.any():
                target_a = (pred_audio_blk.double() - grad_audio_blk.double()).detach()
                diff_a = (pred_audio_blk.double() - target_a) ** 2
                diff_a = diff_a * a_mask_blk.unsqueeze(-1).double()
                loss_a = 0.5 * diff_a.sum() / audio_denom
                block_loss = block_loss + loss_a.to(block_loss.dtype)

            print(f"    block{rs['block_idx']}: surrogate_loss={block_loss.item():.6f}  "
                  f"has_grad={block_loss.requires_grad}")

            # Check gradients flow
            block_loss.backward()
            grad_norms = []
            for name, p in model.generator.model.velocity_model.named_parameters():
                if p.grad is not None:
                    grad_norms.append(p.grad.norm().item())
                    break  # just check one param to confirm grads flow
            if grad_norms:
                print(f"      ✓ grads flow into generator (first param grad norm: {grad_norms[0]:.6f})")
            else:
                print(f"      ✗ WARNING: no grads in generator params!")

            # Zero grads before next block
            model.generator.zero_grad()

            del pred_video_blk, pred_audio_blk, block_loss
            torch.cuda.empty_cache()

        del replay_states

        # -----------------------------------------------------------------
        # Update seq_state for next segment
        # -----------------------------------------------------------------
        cur_video_seqlen_frame = video_state.latent.shape[1] // video_latent_num_frames
        seq_state = {
            "kv_cache_list": updated_kv,
            "video_output": pred_video.detach(),
            "audio_output": pred_audio.detach(),
            "segment_idx": seg_idx + 1,
            "video_offset": segment_video_offset + video_latent_num_frames,
            "video_seqlen_frame": cur_video_seqlen_frame,
            "seg0_batch": batch if seg_idx == 0 else seq_state.get("seg0_batch"),
            "exit_step": rollout_log.get("exit_step", shared_exit_step),
        }

        print(f"\n  Segment {seg_idx} done.")
        print(f"  Updated video_offset for next segment: {seq_state['video_offset']}")
        print(f"  exit_step carried to next segment: {seq_state['exit_step']}")
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  All {n_segments} segment(s) completed successfully.")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=str,
                        default="ltx_experiments/0519_5nodes_bs200_causaldmd/ltx23_causal_dmd.yaml",
                        help="Path to OmegaConf yaml config (relative to project root)")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device for generator / VAEs / text_encoder")
    parser.add_argument("--device_teacher", type=str, default=None,
                        help="Device for real_score + fake_score (e.g. cuda:1). "
                             "When set, enables two-GPU mode to avoid OOM.")
    parser.add_argument("--n_segments", type=int, default=2,
                        help="Number of segments to run (matches seq_steps_per_update in config)")
    parser.add_argument("--fixed_exit_step", type=int, default=1,
                        help="Fixed exit step index (0-based) used as shared_exit_step; "
                             "-1 = random per segment")
    parser.add_argument("--offload", action="store_true",
                        help="Enable CPU offloading to save GPU memory. Models will be moved "
                             "between CPU and GPU on-demand. Mutually exclusive with --device_teacher.")
    parser.add_argument("--height", type=int, default=None,
                        help="Override video height (pixels). Reduces KV cache size. E.g. 256")
    parser.add_argument("--width", type=int, default=None,
                        help="Override video width (pixels). Reduces KV cache size. E.g. 384")
    parser.add_argument("--num_frames", type=int, default=None,
                        help="Override number of video frames. E.g. 25")
    parser.add_argument("--num_training_latents", type=int, default=None,
                        help="Override number of training latent frames. E.g. 4")
    args = parser.parse_args()

    # Resolve config path relative to project root when not absolute
    if not os.path.isabs(args.config):
        config_path = os.path.join(ROOT, args.config)
    else:
        config_path = args.config

    fixed_exit = args.fixed_exit_step if args.fixed_exit_step >= 0 else None

    run_debug(
        config_path=config_path,
        device_str=args.device,
        n_segments=args.n_segments,
        fixed_exit_step=fixed_exit,
        enable_offload=args.offload,
        device_teacher_str=args.device_teacher,
        height_override=args.height,
        width_override=args.width,
        num_frames_override=args.num_frames,
        num_training_latents_override=args.num_training_latents,
    )


if __name__ == "__main__":
    main()
