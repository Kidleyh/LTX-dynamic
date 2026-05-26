"""
LTX-2 DMD (Distribution Matching Distillation) Module.

This module implements DMD for LTX-2 audio-video joint generation,
adapted from CausVid's DMD implementation.

Key differences from CausVid:
- Handles both video and audio modalities jointly
- Uses LTX-2's sigma-based timestep format
- Supports audio-video time alignment
"""

from contextlib import nullcontext
from typing import Tuple, Dict, Any, Optional, List
import math
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from ltx_core.loader.registry import StateDictRegistry

from ltx_core.components.schedulers import LTX2Scheduler

from ltx_distillation.models.ltx_wrapper import LTX2DiffusionWrapper, create_ltx2_wrapper, create_causal_ltx2_wrapper
from ltx_distillation.models.text_encoder_wrapper import GemmaTextEncoderWrapper, create_text_encoder_wrapper
from ltx_distillation.models.vae_wrapper import VideoVAEWrapper, AudioVAEWrapper, create_vae_wrappers
from ltx_distillation.loss import get_denoising_loss
try:
    from ltx_causal.wrapper import CausalLTX2DiffusionWrapper
    from ltx_causal.attention.mask_builder import compute_av_blocks
    from ltx_causal.transformer.causal_model import CausalLTXModel, CausalLTXModelConfig
except ImportError:
    CausalLTX2DiffusionWrapper = None
    compute_av_blocks = None
    CausalLTXModel = None
    CausalLTXModelConfig = None

from ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
    CausalLTXModelConfigurator,
    X0Model,
)

from ltx_core.model.transformer.model import LTXModel, LTXModelType

from ltx_causal.transformer.causal_model import CausalX0Model
from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from dataclasses import dataclass, replace
from ltx_core.model.transformer.modality import Modality

class CausalLTX23DMD(nn.Module):
    """
    DMD (Distribution Matching Distillation) module for LTX-2.

    Implements the DMD algorithm for distilling a multi-step diffusion model
    to a few-step model, supporting audio-video joint generation.

    The module contains three diffusion models:
    - generator: Student model being trained
    - real_score: Teacher model (frozen)
    - fake_score: Critic model for discriminating real vs fake

    Training alternates between:
    1. Generator training: minimize DMD loss (KL divergence from teacher)
    2. Critic training: learn to distinguish generator outputs from teacher
    """

    # Audio-video time alignment constants
    VIDEO_LATENT_FPS = 3.0  # 24fps / 8
    AUDIO_LATENT_FPS = 25.0  # 16kHz / 160 / 4

    def __init__(self, args, device: torch.device, accelerator=None):
        """
        Initialize the DMD module.

        Args:
            args: Configuration object with:
                - checkpoint_path: Path to LTX-2 checkpoint
                - gemma_path: Path to Gemma text encoder
                - denoising_step_list: List of denoising timesteps
                - num_train_timestep: Total training timesteps
                - real_video_guidance_scale: CFG scale for teacher (video)
                - real_audio_guidance_scale: CFG scale for teacher (audio)
                - gradient_checkpointing: Enable gradient checkpointing
                - mixed_precision: Use bfloat16
                - denoising_loss_type: Type of denoising loss
                - video_shape: [B, F, C, H, W]
                - audio_shape: [B, F_a, C]
            device: Target device
        """
        super().__init__()

        self.accelerator = accelerator
        self.args = args
        self.device = device
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32

        # Task types
        self.generator_task_type = getattr(args, "generator_task_type", args.generator_task)
        self.real_task_type = getattr(args, "real_task_type", args.generator_task)
        self.fake_task_type = getattr(args, "fake_task_type", args.generator_task)
        self.training_mode = getattr(args, "training_mode", "bidirectional")
        self.enable_self_forcing = "self_forcing" in str(self.training_mode).lower()
        inferred_causal = (
            "causal" in str(self.training_mode).lower()
            or "causal" in str(self.generator_task_type).lower()
            or "causal" in str(self.real_task_type).lower()
            or "causal" in str(self.fake_task_type).lower()
        )
        self.use_causal_wrapper = bool(getattr(args, "use_causal_wrapper", inferred_causal))
        # Per-model causal wrapper switches (CausVid-style hybrid default).
        # By default:
        # - generator follows global use_causal_wrapper
        # - real/fake follow their task types, enabling bidirectional teacher/critic
        self.generator_use_causal_wrapper = bool(
            getattr(args, "generator_use_causal_wrapper", self.use_causal_wrapper)
        )
        self.real_score_use_causal_wrapper = bool(
            getattr(args, "real_score_use_causal_wrapper", "causal" in str(self.real_task_type).lower())
        )
        self.fake_score_use_causal_wrapper = bool(
            getattr(args, "fake_score_use_causal_wrapper", "causal" in str(self.fake_task_type).lower())
        )
        self.alignment_rounding = str(getattr(args, "alignment_rounding", "round")).lower()
        if self.alignment_rounding not in {"round", "floor", "ceil"}:
            raise ValueError(
                f"Invalid alignment_rounding={self.alignment_rounding}, expected round|floor|ceil"
            )
        if (
            self.generator_use_causal_wrapper
            or self.real_score_use_causal_wrapper
            or self.fake_score_use_causal_wrapper
        ) and CausalLTX2DiffusionWrapper is None:
            raise ImportError(
                "Causal wrapper requires ltx-causal package. "
                "Install with: pip install -e packages/ltx-causal"
            )
        if self.enable_self_forcing and not self.generator_use_causal_wrapper:
            raise ValueError("Stage3 Self-Forcing requires generator_use_causal_wrapper=true")
        self.self_forcing_runtime = str(
            getattr(args, "self_forcing_runtime", "prefix_rerun")
        ).lower()
        if self.self_forcing_runtime not in {"prefix_rerun", "kv_cache"}:
            raise ValueError(
                f"Invalid self_forcing_runtime={self.self_forcing_runtime}, "
                "expected prefix_rerun|kv_cache"
            )
        self.self_forcing_min_generated_blocks = getattr(
            args, "self_forcing_min_generated_blocks", None
        )
        self.self_forcing_max_generated_blocks = getattr(
            args, "self_forcing_max_generated_blocks", None
        )
        self.self_forcing_loss_scope = str(
            getattr(args, "self_forcing_loss_scope", "last_block")
        ).lower()
        if self.self_forcing_loss_scope != "last_block":
            raise ValueError(
                f"Invalid self_forcing_loss_scope={self.self_forcing_loss_scope}, "
                "only last_block is currently supported"
            )

        # Initialize models (will be populated by _init_models or external loading)
        self.generator: LTX2DiffusionWrapper = None
        self.real_score: LTX2DiffusionWrapper = None
        self.fake_score: LTX2DiffusionWrapper = None
        self.text_encoder: GemmaTextEncoderWrapper = None
        self.video_vae: VideoVAEWrapper = None
        self.audio_vae: AudioVAEWrapper = None

        # DMD hyperparameters
        self.num_train_timestep = args.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        self.real_video_guidance_scale = getattr(args, "real_video_guidance_scale", 3.0)
        self.real_audio_guidance_scale = getattr(args, "real_audio_guidance_scale", 7.0)

        # DMD latent noise mode for KL gradient computation.
        # "direct_noise": add Gaussian noise at target sigma (standard DMD)
        # "teacher_denoise": teacher denoises from high noise to target sigma
        self.dmd_latent_mode = getattr(args, "dmd_latent_mode", "direct_noise")

        # Video/Audio loss weighting for ablation experiments.
        # video_loss_weight + audio_loss_weight need not sum to 1.
        # Supports two-phase training: video-only phase then joint phase.
        self.video_loss_weight = getattr(args, "video_loss_weight", 1.0)
        self.audio_loss_weight = getattr(args, "audio_loss_weight", 1.0)
        # Two-phase: if audio_start_step > 0, audio_loss_weight=0 until that step
        self.audio_start_step = getattr(args, "audio_start_step", 0)

        # Denoising sigmas aligned with ODE pair generation.
        # ODE pairs are generated with a fine-grained schedule (e.g. 40 steps)
        # then subsampled to denoising_step_list by finding the closest sigma.
        # We replicate that logic here so Stage 1/3 DMD training uses the exact
        # same sigma values as the ODE trajectories stored in LMDB.
        _ode_num_steps = getattr(args, "num_inference_steps", 40)
        _full_sigmas = LTX2Scheduler().execute(steps=_ode_num_steps)
        _denoising_sigmas = []
        for t in args.denoising_step_list:
            target_sigma = t / 1000.0
            idx = (_full_sigmas - target_sigma).abs().argmin().item()
            _denoising_sigmas.append(_full_sigmas[idx])
        self.denoising_sigmas = torch.stack(_denoising_sigmas).to(device)

        # Pre-compute sigma lookup table for random timestep → sigma conversion.
        # This matches CausVid's approach where scheduler.add_noise() internally
        # does argmin lookup against the scheduler's sigma schedule.
        # We compute a 1001-entry table (timestep 0..1000) using the native
        # LTX2Scheduler's shifted+stretched sigmoid formula.
        # sigma_lookup[t] gives the actual sigma for integer timestep t.
        scheduler = LTX2Scheduler()
        full_sigmas = scheduler.execute(steps=self.num_train_timestep).to(device)  # [1001] values
        # full_sigmas goes from ~1.0 (noise) to 0.0 (clean), same order as timesteps 1000→0
        # We need sigma_lookup[t] where t=0 → sigma=0 (clean) and t=1000 → sigma≈1 (noise)
        # full_sigmas is ordered: sigma[0]=high (noise), sigma[-1]=0 (clean)
        # So sigma_lookup[t] = full_sigmas[1000 - t] maps t=1000→full_sigmas[0], t=0→full_sigmas[1000]
        self.register_buffer(
            'sigma_lookup',
            full_sigmas.flip(0),  # Reverse so index 0=clean(σ≈0), index 1000=noise(σ≈1)
        )

        # Teacher denoise config (only used when dmd_latent_mode == "teacher_denoise")
        if self.dmd_latent_mode == "teacher_denoise":
            self.teacher_num_steps = getattr(args, "teacher_num_steps", 40)
            # How many teacher schedule steps above target_sigma to start from.
            # E.g. offset=5 means: start 5 steps before target in the sigma schedule,
            # so the teacher only runs ~5 Euler steps regardless of target sigma.
            # Smaller = faster + more student structure preserved.
            # Larger = more "on teacher trajectory" but slower.
            self.teacher_start_offset = getattr(args, "teacher_start_offset", 5)
            # Pre-compute fine-grained teacher sigma schedule.
            # teacher_sigmas[0] ≈ 1.0 (noise), teacher_sigmas[-1] = 0.0 (clean)
            teacher_sigmas = LTX2Scheduler().execute(steps=self.teacher_num_steps).to(device)
            self.register_buffer('teacher_sigmas', teacher_sigmas)

        # Loss function
        self.denoising_loss_func = get_denoising_loss(args.denoising_loss_type)()

        # Block-aware loss weighting for over-exposure suppression.
        self.num_frame_per_block = getattr(args, "num_frame_per_block", 3)
        self.block_weight_mode = getattr(args, "block_weight_mode", "uniform")
        self.block_weight_min = getattr(args, "block_weight_min", 0.5)

        # Inference pipeline (lazy init)
        self.inference_pipeline = None
        self.causal_inference_pipeline = None

        # Current training step (updated by trainer)
        self.current_step = 0

    def get_loss_weights(self) -> Tuple[float, float]:
        """Get current video/audio loss weights based on training step."""
        video_w = self.video_loss_weight
        audio_w = self.audio_loss_weight
        if self.audio_start_step > 0 and self.current_step < self.audio_start_step:
            audio_w = 0.0
        return video_w, audio_w

    def timestep_to_sigma(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Convert integer timestep (0-1000) to sigma using LTX2Scheduler's lookup table.

        Uses a pre-computed lookup table from the native LTX2Scheduler (shifted+stretched
        sigmoid schedule) instead of a linear t/1000 mapping. This matches CausVid's
        approach where scheduler.add_noise() does internal argmin lookup.

        Args:
            timestep: Integer timestep tensor [B, F] in range [0, num_train_timestep]

        Returns:
            Sigma tensor with same shape, values in [0, 1]
        """
        # Clamp to valid range and index into pre-computed lookup table
        t_clamped = timestep.long().clamp(0, self.num_train_timestep)
        return self.sigma_lookup[t_clamped]

    def add_noise(
        self,
        original: torch.Tensor,
        noise: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """
        Add noise to samples using flow matching interpolation.

        Flow matching formula: x_t = (1 - sigma) * x_0 + sigma * epsilon

        Args:
            original: Clean samples x_0, shape [B, ...]
            noise: Gaussian noise epsilon, shape [B, ...]
            sigma: Noise level, shape [B] or [B, T] or scalar

        Returns:
            Noisy samples x_t
        """
        # Reshape sigma for broadcasting
        if sigma.dim() == 1:
            # [B] -> [B, 1, 1, 1, ...] for proper broadcasting
            sigma = sigma.reshape(-1, *[1] * (original.dim() - 1))
        elif sigma.dim() == 2:
            # [B, T] -> [B, T, 1, 1, ...] for video/audio
            sigma = sigma.reshape(*sigma.shape, *[1] * (original.dim() - 2))
        sigma = sigma.to(dtype=original.dtype)
        return ((1 - sigma) * original + sigma * noise).to(dtype=original.dtype)

    def init_models(self):
        """
        Initialize all models from checkpoints.

        This method should be called BEFORE FSDP wrapping in distributed training.
        Models must exist before they can be wrapped with FSDP.
        """
        args = self.args

        def _init_log(message: str) -> None:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[DMDInit] {message}", flush=True)

        # Get video dimensions from config
        # video_height = getattr(args, "video_height", 512) # [NOTE] deprecated
        # video_width = getattr(args, "video_width", 768) # [NOTE] deprecated

        # Create diffusion wrappers per model (CausVid-style hybrid setup):
        # generator can be causal while real/fake remain bidirectional.
        if isinstance(self.device, int):
            target_device = f"cuda:{self.device}"
        else:
            target_device = str(self.device)

        def _load_checkpoint_state_dict(checkpoint_path: str) -> dict:
            if checkpoint_path in checkpoint_state_cache:
                return checkpoint_state_cache[checkpoint_path]
            if checkpoint_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                loaded = load_file(checkpoint_path)
                checkpoint_state_cache[checkpoint_path] = loaded
                return loaded

            loaded = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(loaded, dict) and "generator" in loaded:
                loaded = loaded["generator"]
            elif isinstance(loaded, dict) and "model" in loaded:
                loaded = loaded["model"]
            elif isinstance(loaded, dict) and "state_dict" in loaded:
                loaded = loaded["state_dict"]
            checkpoint_state_cache[checkpoint_path] = loaded
            return loaded

        # def _remap_state_dict_keys(state_dict: dict) -> dict:
        #     if not state_dict:
        #         return state_dict

        #     non_transformer_prefixes = (
        #         "vae.", "audio_vae.", "vocoder.",
        #         "model.vae.", "model.audio_vae.", "model.vocoder.",
        #     )
        #     remapped_non_transformer_prefixes = (
        #         "model.audio_embeddings_connector.",
        #         "model.video_embeddings_connector.",
        #     )

        #     sample_keys = list(state_dict.keys())[:20]
        #     has_diffusion_model = any(k.startswith("model.diffusion_model.") for k in sample_keys)
        #     if not has_diffusion_model:
        #         has_diffusion_model = any(k.startswith("model.diffusion_model.") for k in state_dict)

        #     if has_diffusion_model:
        #         remapped = {}
        #         for k, v in state_dict.items():
        #             if not k.startswith("model.diffusion_model."):
        #                 continue
        #             new_key = "model." + k[len("model.diffusion_model."):]
        #             if any(new_key.startswith(p) for p in remapped_non_transformer_prefixes):
        #                 continue
        #             remapped[new_key] = v
        #         return remapped

        #     first_key = next(iter(state_dict))
        #     if first_key.startswith("model.velocity_model."):
        #         return {
        #             "model." + k[len("model.velocity_model."):]: v
        #             for k, v in state_dict.items()
        #             if k.startswith("model.velocity_model.")
        #         }
        #     if first_key.startswith("model."):
        #         return {
        #             k: v for k, v in state_dict.items()
        #             if not any(k.startswith(p) for p in non_transformer_prefixes)
        #         }
        #     return {
        #         "model." + k: v
        #         for k, v in state_dict.items()
        #         if not any(k.startswith(p) for p in non_transformer_prefixes)
        #     }

        def _remap_state_dict_keys_generator(state_dict: dict) -> dict:
            sample_keys = list(state_dict.keys())[:20]
            has_model = any(k.startswith("model.") for k in sample_keys)
            if not has_model:
                has_model = any(k.startswith("model.") for k in state_dict)
            has_velocity_model = any(k.startswith("model.velocity_model.") for k in sample_keys)
            if not has_velocity_model:
                has_velocity_model = any(k.startswith("model.velocity_model.") for k in state_dict)
            
            if has_velocity_model:
                remapped = {}
                for k, v in state_dict.items():
                    if not k.startswith("model.velocity_model."):
                        continue
                    new_key = k[len("model.velocity_model."):]
                    remapped[new_key] = v
                return remapped

            elif has_model:
                remapped = {}
                for k, v in state_dict.items():
                    if not k.startswith("model."):
                        continue
                    new_key = k[len("model."):]
                    remapped[new_key] = v
                return remapped
            
            return state_dict

        def _build_wrapper(use_causal: bool):

            if use_causal:
                return create_causal_ltx2_wrapper(
                    checkpoint_path=args.checkpoint_path,
                    gemma_path=args.gemma_path,
                    device=torch.device("cpu"),
                    dtype=self.dtype,
                    use_flex_attention=args.use_flex_attention,
                    # video_height=None,
                    # video_width=None,
                    registry=shared_registry,
                )

            _init_log("build bidirectional wrapper start")
            return create_ltx2_wrapper(
                checkpoint_path=args.checkpoint_path,
                gemma_path=args.gemma_path,
                device=torch.device("cpu"),
                dtype=self.dtype,
                # video_height=video_height,
                # video_width=video_width,
                registry=shared_registry,
            )

        checkpoint_state_cache: Dict[str, dict] = {}
        shared_registry = StateDictRegistry()
        _init_log("generator wrapper init start")
        self.generator = _build_wrapper(self.generator_use_causal_wrapper)
        _init_log("generator wrapper init done")
        _init_log("real_score wrapper init start")
        self.real_score = _build_wrapper(self.real_score_use_causal_wrapper)
        _init_log("real_score wrapper init done")
        _init_log("fake_score wrapper init start")
        self.fake_score = _build_wrapper(self.fake_score_use_causal_wrapper)
        _init_log("fake_score wrapper init done")

        _init_log("text encoder init start")
        self.text_encoder = create_text_encoder_wrapper(
            checkpoint_path=args.checkpoint_path,
            gemma_path=args.gemma_path,
            device=torch.device("cpu"),
            dtype=self.dtype,
            load_in_8bit=False,
            registry=shared_registry,
        )
        _init_log("text encoder init done")

        _init_log("vae init start")
        self.video_vae, self.audio_vae = create_vae_wrappers(
            checkpoint_path=args.checkpoint_path,
            device=self.device,
            dtype=self.dtype,
            registry=shared_registry,
        )
        _init_log("vae init done")

        # Set gradients

        # enable all grads
        self.generator.set_module_grad(args.generator_grad)
        self.real_score.set_module_grad(args.real_score_grad)
        self.fake_score.set_module_grad(args.fake_score_grad)

        # [DEBUG] enable part of grads
        if getattr(args, "debug_mode", False):
            _init_log("debug mode enabled, enable part of grads")
            self.generator.set_module_grad({'model': False})
            self.real_score.set_module_grad({'model': False})
            self.fake_score.set_module_grad({'model': False})
            # enable 2 layers' grad
            for i in range(48):
                for param in self.generator.model.velocity_model.transformer_blocks[i].parameters():
                    param.requires_grad = True
            for i in range(48):
                for param in self.fake_score.model.velocity_model.transformer_blocks[i].parameters():
                    param.requires_grad = True

        self.text_encoder.requires_grad_(False)
        self.video_vae.requires_grad_(False)
        self.audio_vae.requires_grad_(False)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # Checkpoint loading with priority:
        #   resume_checkpoint > generator_ckpt > stage1_ckpt_path
        stage1_ckpt = getattr(args, "stage1_ckpt_path", None)
        stage1_strict = getattr(args, "stage1_ckpt_strict", False)
        generator_ckpt = getattr(args, "generator_ckpt", None)
        generator_ckpt_strict = getattr(args, "generator_ckpt_strict", False)

        if generator_ckpt:
            print(f"Loading pretrained generator from {generator_ckpt}")
            ckpt = torch.load(generator_ckpt, map_location="cpu")
            gen_sd = ckpt.get("generator", ckpt)
            if self.generator_use_causal_wrapper:
                gen_sd = _remap_state_dict_keys_generator(gen_sd)
            missing_g, unexpected_g = self.generator.model.velocity_model.load_state_dict(gen_sd, strict=generator_ckpt_strict)
            real_missing_g = [k for k in missing_g if "mask_builder" not in k]
            if real_missing_g:
                print(f"  [generator] missing keys ({len(real_missing_g)}): {real_missing_g[:10]}...")
            if unexpected_g:
                print(f"  [generator] unexpected keys ({len(unexpected_g)}): {unexpected_g[:10]}...")

            print("[Stage3] Generator checkpoint load complete")

    def _round_align(self, value: float) -> int:
        if self.alignment_rounding == "floor":
            return int(torch.floor(torch.tensor(value)).item())
        if self.alignment_rounding == "ceil":
            return int(torch.ceil(torch.tensor(value)).item())
        return int(round(value))

    @staticmethod
    def _is_bidirectional_task(task_type: Optional[str]) -> bool:
        return "bidirectional" in str(task_type).lower()

    @staticmethod
    def _is_causal_task(task_type: Optional[str]) -> bool:
        return "causal" in str(task_type).lower()

    def _get_causal_blocks(self, num_video_frames: int):
        if compute_av_blocks is None:
            raise ImportError("Causal block utilities require the ltx-causal package")
        return compute_av_blocks(
            total_video_latent_frames=num_video_frames,
            num_frame_per_block=self.num_frame_per_block,
        )

    def _build_current_block_masks(
        self,
        num_video_frames: int,
        num_audio_frames: int,
        block_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        blocks = self._get_causal_blocks(num_video_frames)
        batch_size = block_indices.shape[0]

        video_mask = torch.zeros(
            batch_size, num_video_frames, device=block_indices.device, dtype=torch.bool
        )
        audio_mask = torch.zeros(
            batch_size, num_audio_frames, device=block_indices.device, dtype=torch.bool
        )

        for batch_idx, block_idx in enumerate(block_indices.tolist()):
            block = blocks[block_idx]
            video_mask[batch_idx, block.video_start:block.video_end] = True
            audio_end = min(block.audio_end, num_audio_frames)
            if audio_end > block.audio_start:
                audio_mask[batch_idx, block.audio_start:audio_end] = True

        return video_mask, audio_mask

    def _sample_causal_training_blocks(
        self,
        batch_size: int,
        num_video_frames: int,
    ) -> torch.Tensor:
        blocks = self._get_causal_blocks(num_video_frames)
        if len(blocks) <= 1:
            raise ValueError(
                f"Causal training requires at least one standard block, got {num_video_frames} video frames"
            )
        return torch.randint(
            1,
            len(blocks),
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )

    def _sample_i2v_timesteps(
        self,
        batch_size: int,
        video_mask: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        video_timestep = torch.zeros(
            video_mask.shape,
            device=self.device,
            dtype=torch.long,
        )
        audio_timestep = torch.zeros(
            audio_mask.shape,
            device=self.device,
            dtype=torch.long,
        )

        sampled_timestep = torch.randint(
            self.min_step,
            self.max_step + 1,
            (batch_size,),
            device=self.device,
            dtype=torch.long,
        )
        for batch_idx in range(batch_size):
            video_timestep[batch_idx, video_mask[batch_idx].bool()] = sampled_timestep[batch_idx]
            audio_timestep[batch_idx, audio_mask[batch_idx].bool()] = sampled_timestep[batch_idx]

        return video_timestep, audio_timestep, sampled_timestep

    def _process_timestep(self, timestep: torch.Tensor, task_type: str) -> torch.Tensor:
        """
        Process timestep based on task type.

        For causal tasks, each block of num_frame_per_block frames shares the
        same timestep (noise level), matching CausVid semantics.

        Args:
            timestep: [B, F] tensor of timesteps
            task_type: "bidirectional_av", "bidirectional_video", "causal_av", etc.

        Returns:
            Processed timestep tensor
        """
        if self._is_bidirectional_task(task_type):
            for i in range(timestep.shape[0]):
                timestep[i] = timestep[i, 0]
            return timestep
        elif "causal" in task_type:
            result = timestep.clone()
            if result.shape[1] <= 1:
                return result
            idx = 1
            while idx < result.shape[1]:
                end = min(idx + self.num_frame_per_block, result.shape[1])
                result[:, idx:end] = result[:, idx:idx + 1].expand(-1, end - idx)
                idx = end
            return result
        else:
            return timestep

    def _compute_audio_timestep(
        self,
        video_timestep: torch.Tensor,
        num_audio_frames: int,
        task_type: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Compute audio timestep from video timestep.

        In bidirectional mode, all frames use the same timestep.
        In causal mode, audio frames inherit the timestep from their
        corresponding video block via the AV alignment ratio.
        """
        B = video_timestep.shape[0]
        num_video_frames = video_timestep.shape[1]
        mode = task_type or self.real_task_type

        if self._is_bidirectional_task(mode):
            return video_timestep[:, 0:1].expand(B, num_audio_frames)

        # Causal/non-bidirectional: map audio blocks to the video block sigma
        # defined by the causal wrapper's Global Prefix schedule.
        audio_timestep = torch.zeros(
            B, num_audio_frames, device=video_timestep.device, dtype=video_timestep.dtype
        )
        for block in self._get_causal_blocks(num_video_frames):
            if block.audio_start >= num_audio_frames:
                break
            audio_end = min(block.audio_end, num_audio_frames)
            if audio_end <= block.audio_start:
                continue
            audio_timestep[:, block.audio_start:audio_end] = video_timestep[
                :, block.video_start:block.video_start + 1
            ].expand(B, audio_end - block.audio_start)
        return audio_timestep

    @torch.no_grad()
    def _teacher_denoise_cfg_step(
        self,
        noisy_video: torch.Tensor,
        noisy_audio: torch.Tensor,
        video_sigma: torch.Tensor,
        audio_sigma: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One teacher denoising step with classifier-free guidance.

        Calls real_score twice (conditional + unconditional) and applies
        LTX-2's CFG formula with separate video/audio guidance scales.

        Returns:
            Tuple of (pred_video_x0, pred_audio_x0)
        """
        pred_cond_video, pred_cond_audio = self.real_score(
            noisy_image_or_video=noisy_video,
            conditional_dict=conditional_dict,
            timestep=video_sigma,
            noisy_audio=noisy_audio,
            audio_timestep=audio_sigma,
        )

        pred_uncond_video, pred_uncond_audio = self.real_score(
            noisy_image_or_video=noisy_video,
            conditional_dict=unconditional_dict,
            timestep=video_sigma,
            noisy_audio=noisy_audio,
            audio_timestep=audio_sigma,
        )

        # CFG: output = cond + (scale - 1) * (cond - uncond)
        pred_video_x0 = pred_cond_video + (self.real_video_guidance_scale - 1) * (
            pred_cond_video - pred_uncond_video
        )
        pred_audio_x0 = pred_cond_audio + (self.real_audio_guidance_scale - 1) * (
            pred_cond_audio - pred_uncond_audio
        )

        return pred_video_x0, pred_audio_x0

    @torch.no_grad()
    def _get_noisy_latent_via_teacher_denoise(
        self,
        clean_video: torch.Tensor,
        clean_audio: torch.Tensor,
        target_video_sigma: torch.Tensor,
        target_audio_sigma: torch.Tensor,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Teacher denoises from high noise to target sigma, producing latents
        on the teacher's ODE trajectory instead of the Gaussian interpolation line.

        The number of Euler steps is determined solely by where target_sigma
        falls in the pre-computed teacher_sigmas schedule — no randomization.

        Args:
            clean_video: Generator-predicted clean video [B, F_v, C, H, W]
            clean_audio: Generator-predicted clean audio [B, F_a, C]
            target_video_sigma: Target sigma [B, F_v]
            target_audio_sigma: Target sigma [B, F_a]
            conditional_dict: Conditional embeddings
            unconditional_dict: Unconditional embeddings

        Returns:
            (noisy_video, noisy_audio, actual_video_sigma, actual_audio_sigma)
            where actual sigmas are the exact teacher schedule values at the
            stop point, guaranteed to match the returned latents' noise level.
        """
        B = clean_video.shape[0]
        F_v = clean_video.shape[1]
        F_a = clean_audio.shape[1]
        device = clean_video.device

        teacher_sigmas = self.teacher_sigmas  # [N+1], descending: [0]≈1.0, [-1]=0.0
        offset = self.teacher_start_offset

        # In bidirectional mode all frames share the same sigma
        target_scalar = target_video_sigma[:, 0]  # [B]

        # Find the closest index in teacher_sigmas for each batch element
        target_idx = torch.argmin(
            (teacher_sigmas.unsqueeze(0) - target_scalar.unsqueeze(1)).abs(),
            dim=1,
        )  # [B]
        target_idx = target_idx.clamp(min=1)  # at least 1 denoising step

        # Start index: `offset` steps before target in the schedule.
        # start_idx < target_idx, so teacher_sigmas[start_idx] > teacher_sigmas[target_idx].
        # Clamped to 0 so we never go before the schedule start.
        start_idx = (target_idx - offset).clamp(min=0)  # [B]
        num_steps = target_idx - start_idx  # [B], exactly how many Euler steps each element needs

        # NCCL safety: all ranks must call real_score() the same number of times.
        max_steps = num_steps.max().item()
        if dist.is_initialized():
            max_steps_tensor = torch.tensor(max_steps, device=device, dtype=torch.long)
            dist.all_reduce(max_steps_tensor, op=dist.ReduceOp.MAX)
            max_steps = max_steps_tensor.item()

        # Add noise at each element's start sigma (not necessarily pure noise)
        noise_video = torch.randn_like(clean_video)
        noise_audio = torch.randn_like(clean_audio)

        start_sigma_per_elem = teacher_sigmas[start_idx]  # [B]
        s_v = start_sigma_per_elem.unsqueeze(1).expand(B, F_v)
        s_a = start_sigma_per_elem.unsqueeze(1).expand(B, F_a)

        current_video = self.add_noise(
            clean_video.flatten(0, 1),
            noise_video.flatten(0, 1),
            s_v.flatten(0, 1),
        ).unflatten(0, (B, F_v))
        current_audio = self.add_noise(clean_audio, noise_audio, s_a)

        # Teacher Euler denoising: each element runs from its own start_idx to target_idx.
        # We iterate max_steps times; each element's absolute schedule index is start_idx + step_i.
        for step_i in range(max_steps):
            active = (step_i < num_steps)  # [B]

            # Each element may be at a different position in the schedule
            abs_idx = start_idx + step_i  # [B]
            cur_sigma = teacher_sigmas[abs_idx]    # [B]
            nxt_sigma = teacher_sigmas[(abs_idx + 1).clamp(max=len(teacher_sigmas) - 1)]  # [B]

            v_sigma = cur_sigma.unsqueeze(1).expand(B, F_v)
            a_sigma = cur_sigma.unsqueeze(1).expand(B, F_a)

            pred_v_x0, pred_a_x0 = self._teacher_denoise_cfg_step(
                noisy_video=current_video,
                noisy_audio=current_audio,
                video_sigma=v_sigma,
                audio_sigma=a_sigma,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
            )

            # Euler step: v = (x_t - x_0) / sigma, x_{t'} = x_t + v * (sigma' - sigma)
            cur_sigma_bcast = cur_sigma.view(B, 1, 1, 1, 1)
            dsigma = (nxt_sigma - cur_sigma).view(B, 1, 1, 1, 1)
            vel_v = (current_video - pred_v_x0) / cur_sigma_bcast
            nxt_v = current_video + vel_v * dsigma

            cur_sigma_audio = cur_sigma.view(B, 1, 1)
            dsigma_audio = (nxt_sigma - cur_sigma).view(B, 1, 1)
            vel_a = (current_audio - pred_a_x0) / cur_sigma_audio
            nxt_a = current_audio + vel_a * dsigma_audio

            m_v = active.view(B, 1, 1, 1, 1).expand_as(current_video)
            m_a = active.view(B, 1, 1).expand_as(current_audio)
            current_video = torch.where(m_v, nxt_v, current_video)
            current_audio = torch.where(m_a, nxt_a, current_audio)

        # Actual sigma at the exact stopping point (from the teacher schedule)
        actual_sigma = teacher_sigmas[target_idx]  # [B]
        actual_video_sigma = actual_sigma.unsqueeze(1).expand(B, F_v)
        actual_audio_sigma = actual_sigma.unsqueeze(1).expand(B, F_a)

        return current_video, current_audio, actual_video_sigma, actual_audio_sigma

    def _compute_kl_grad(
        self,
        # noisy_video: torch.Tensor,
        # noisy_audio: torch.Tensor,
        # clean_video: torch.Tensor,
        # clean_audio: torch.Tensor,
        # video_sigma: torch.Tensor,
        # audio_sigma: torch.Tensor,
        video_state: LatentState,
        audio_state: LatentState,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        video_latent_num_frames: int,
        sigma: torch.Tensor,
        video_loss_mask: Optional[torch.Tensor] = None,
        audio_loss_mask: Optional[torch.Tensor] = None,
        normalization: bool = True,
        segment_video_offset: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Compute KL gradient for both video and audio.

        This implements Equation 7 from the DMD paper.

        Args:
            video_sigma: Noise level sigma [B, F_v], passed directly to score networks.
            audio_sigma: Noise level sigma [B, F_a], passed directly to score networks.
            segment_video_offset: When > 0 (2nd+ segment), apply TI2V-style RoPE time
                shift to teacher (real_score) only: frame-0 patches → time=0, rest += 10s.
        """

        # replace first frame from generator's prediction with original clean latent
        video_latent_with_first_frame = video_state.latent * video_state.denoise_mask + video_state.clean_latent * (1 - video_state.denoise_mask)
        video_state = replace(video_state, latent=video_latent_with_first_frame)

        video_modality_cond, audio_modality_cond = self._prepare_modality_from_state(
            video_latent_state=video_state,
            audio_latent_state=audio_state,
            video_latent_num_frames=video_latent_num_frames,
            sigma=sigma,
            text_context_dict=conditional_dict,
        )

        video_modality_uncond, audio_modality_uncond = self._prepare_modality_from_state(
            video_latent_state=video_state,
            audio_latent_state=audio_state,
            video_latent_num_frames=video_latent_num_frames,
            sigma=sigma,
            text_context_dict=unconditional_dict,
        )

        # replace noisy latent input to align with trivial cfg setting
        video_modality_uncond = replace(video_modality_uncond, latent=video_modality_cond.latent)
        audio_modality_uncond = replace(audio_modality_uncond, latent=audio_modality_cond.latent)

        # Step 1: Fake score prediction (original positions, no shift)
        pred_fake_video, pred_fake_audio = self.fake_score(
            # noisy_image_or_video=noisy_video,
            # conditional_dict=conditional_dict,
            # timestep=video_sigma,
            # noisy_audio=noisy_audio,
            # audio_timestep=audio_sigma,
            video=video_modality_cond,
            audio=audio_modality_cond,
            perturbations=None,
        )

        # Step 2: Real score prediction with CFG.
        # For 2nd+ segments, apply TI2V RoPE time shift to teacher positions only:
        # frame-0 patches → time=0, remaining patches → time += 10s.
        if segment_video_offset > 0:
            v_pos = video_modality_cond.positions  # (B, 3, T, 2)
            n_patches_spatial = video_state.latent.shape[1] // video_latent_num_frames
            if n_patches_spatial < v_pos.shape[2]:
                new_time = v_pos[:, :1, :, :].clone()
                new_time[:, :, :n_patches_spatial, :] = 0.0
                new_time[:, :, n_patches_spatial:, :] += 10.0
                teacher_v_pos = torch.cat([new_time, v_pos[:, 1:, :, :]], dim=1)
            else:
                teacher_v_pos = v_pos
            a_pos = audio_modality_cond.positions  # (B, 1, T, 2)
            if a_pos.shape[2] > 1:
                new_a_time = a_pos.clone()
                new_a_time[:, :, 0:1, :] = 0.0
                new_a_time[:, :, 1:, :] += 10.0
                teacher_a_pos = new_a_time
            else:
                teacher_a_pos = a_pos
            video_modality_cond_teacher = replace(video_modality_cond, positions=teacher_v_pos)
            audio_modality_cond_teacher = replace(audio_modality_cond, positions=teacher_a_pos)
            video_modality_uncond_teacher = replace(video_modality_uncond, positions=teacher_v_pos)
            audio_modality_uncond_teacher = replace(audio_modality_uncond, positions=teacher_a_pos)
        else:
            video_modality_cond_teacher = video_modality_cond
            audio_modality_cond_teacher = audio_modality_cond
            video_modality_uncond_teacher = video_modality_uncond
            audio_modality_uncond_teacher = audio_modality_uncond

        pred_real_cond_video, pred_real_cond_audio = self.real_score(
            # noisy_image_or_video=noisy_video,
            # conditional_dict=conditional_dict,
            # timestep=video_sigma,
            # noisy_audio=noisy_audio,
            # audio_timestep=audio_sigma,
            video=video_modality_cond_teacher,
            audio=audio_modality_cond_teacher,
            perturbations=None,
        )

        pred_real_uncond_video, pred_real_uncond_audio = self.real_score(
            # noisy_image_or_video=noisy_video,
            # conditional_dict=unconditional_dict,
            # timestep=video_sigma,
            # noisy_audio=noisy_audio,
            # audio_timestep=audio_sigma,
            video=video_modality_uncond_teacher,
            audio=audio_modality_uncond_teacher,
            perturbations=None,
        )

        # Apply CFG: output = cond + (scale - 1) * (cond - uncond)
        # This matches LTX-2's native CFGGuider.delta = (scale - 1) * (cond - uncond)
        # With video_scale=3.0: effective = 3.0*cond - 2.0*uncond
        # With audio_scale=7.0: effective = 7.0*cond - 6.0*uncond
        pred_real_video = pred_real_cond_video + (self.real_video_guidance_scale - 1) * (
            pred_real_cond_video - pred_real_uncond_video
        )
        pred_real_audio = pred_real_cond_audio + (self.real_audio_guidance_scale - 1) * (
            pred_real_cond_audio - pred_real_uncond_audio
        )

        # Step 3: Compute DMD gradient
        grad_video = pred_fake_video - pred_real_video
        grad_audio = pred_fake_audio - pred_real_audio

        clean_video = video_state.latent.to(device=self.device)
        clean_audio = audio_state.latent.to(device=self.device)

        # Step 4: Gradient normalization (Eq. 8)
        if normalization:
            # Video normalization
            p_real_video = clean_video - pred_real_video
            if video_loss_mask is not None:
                video_mask = video_loss_mask.to(device=p_real_video.device, dtype=p_real_video.dtype)
                # Count all active latent elements, not just active frames.
                video_active = video_mask.expand_as(p_real_video).sum(dim=[1, 2], keepdim=True).clamp_min(1.0)
                normalizer_video = (torch.abs(p_real_video) * video_mask).sum(dim=[1, 2], keepdim=True)
                normalizer_video = normalizer_video / video_active
            else:
                normalizer_video = torch.abs(p_real_video).mean(dim=[1, 2, 3, 4], keepdim=True)
            grad_video = grad_video / (normalizer_video + 1e-8)

            # Audio normalization
            p_real_audio = clean_audio - pred_real_audio
            if audio_loss_mask is not None:
                audio_mask = audio_loss_mask.to(device=p_real_audio.device, dtype=p_real_audio.dtype)
                audio_active = audio_mask.expand_as(p_real_audio).sum(dim=[1, 2], keepdim=True).clamp_min(1.0)
                normalizer_audio = (torch.abs(p_real_audio) * audio_mask).sum(dim=[1, 2], keepdim=True)
                normalizer_audio = normalizer_audio / audio_active
            else:
                normalizer_audio = torch.abs(p_real_audio).mean(dim=[1, 2], keepdim=True)
            grad_audio = grad_audio / (normalizer_audio + 1e-8)

        grad_video = torch.nan_to_num(grad_video)
        grad_audio = torch.nan_to_num(grad_audio)

        log_dict = {
            "dmdtrain_gradient_norm_video": torch.mean(torch.abs(grad_video)).detach().item(),
            "dmdtrain_gradient_norm_audio": torch.mean(torch.abs(grad_audio)).detach().item(),
            # "real_score_video": torch.mean(torch.abs(pred_real_video)).item(),
            # "real_score_audio": torch.mean(torch.abs(pred_real_audio)).item(),
            # "fake_score_video": torch.mean(torch.abs(pred_fake_video)).item(),
            # "fake_score_audio": torch.mean(torch.abs(pred_fake_audio)).item(),
            "dmdtrain_clean_latent_video": clean_video.detach(),
            "dmdtrain_clean_latent_audio": clean_audio.detach(),
            "dmdtrain_noisy_latent_video": video_modality_cond.latent.detach(),
            "dmdtrain_noisy_latent_audio": audio_modality_cond.latent.detach(),
            "dmdtrain_pred_real_video": pred_real_video.detach(),
            "dmdtrain_pred_fake_video": pred_fake_video.detach(),
            "dmdtrain_pred_real_audio": pred_real_audio.detach(),
            "dmdtrain_pred_fake_audio": pred_fake_audio.detach(),
            "dmdtrain_noise_sigma": sigma[0].detach().item(),
        }

        return grad_video, grad_audio, log_dict

    def _compute_block_weights(self, num_frames: int, *, is_audio: bool = False) -> torch.Tensor:
        """
        Compute per-frame loss weights based on block position.

        For "linear_ramp", early blocks get lower weight (block_weight_min)
        ramping linearly to 1.0 at the last block.
        For "uniform" or "none", returns all-ones.

        Returns:
            Tensor [num_frames] of per-frame weights on self.device.
        """

        if self.block_weight_mode == "uniform" or self.block_weight_mode == "none":
            return torch.ones(num_frames, device=self.device, dtype=torch.float64)

        if self._is_causal_task(self.generator_task_type):
            blocks = self._get_causal_blocks(
                math.ceil((num_frames - 1) / 25) * self.num_frame_per_block + 1
            ) if is_audio else self._get_causal_blocks(num_frames)
            if is_audio:
                blocks = [block for block in blocks if block.audio_start < num_frames]
            else:
                blocks = [block for block in blocks if block.video_start < num_frames]
            n_blocks = len(blocks)
        else:
            nfpb = self.num_frame_per_block
            n_blocks = math.ceil(num_frames / nfpb)
            blocks = None

        if n_blocks <= 1:
            return torch.ones(num_frames, device=self.device, dtype=torch.float64)

        weights = torch.ones(num_frames, device=self.device, dtype=torch.float64)
        if blocks is not None:
            for blk_idx, block in enumerate(blocks):
                w = self.block_weight_min + (1.0 - self.block_weight_min) * blk_idx / (n_blocks - 1)
                start = block.audio_start if is_audio else block.video_start
                end = min(block.audio_end if is_audio else block.video_end, num_frames)
                weights[start:end] = w
        else:
            for blk in range(n_blocks):
                start = blk * nfpb
                end = min(start + nfpb, num_frames)
                w = self.block_weight_min + (1.0 - self.block_weight_min) * blk / (n_blocks - 1)
                weights[start:end] = w

        return weights

    @staticmethod
    def _masked_weighted_mean(
        values: torch.Tensor,
        weights: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask is None:
            return (values * weights).mean()
        mask_f = mask.to(values.dtype)
        weighted = values * weights * mask_f
        denom = (weights * mask_f).sum().clamp_min(1.0)
        return weighted.sum() / denom

    def _compute_masked_denoising_loss(
        self,
        *,
        target: torch.Tensor,
        prediction: torch.Tensor,
        noise: torch.Tensor,
        flow_pred: Optional[torch.Tensor],
        timestep: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mask is None:
            return self.denoising_loss_func(
                x=target,
                x_pred=prediction,
                noise=noise,
                noise_pred=None,
                alphas_cumprod=None,
                timestep=timestep,
                flow_pred=flow_pred,
            )

        loss_type = str(getattr(self.args, "denoising_loss_type", "velocity")).lower()
        if loss_type == "x0":
            diff = (target.double() - prediction.double()) ** 2
        elif loss_type in {"velocity", "flow"}:
            pred = flow_pred.double() if flow_pred is not None else (noise.double() - prediction.double())
            diff = (pred - (noise.double() - target.double())) ** 2
        else:
            raise NotImplementedError(
                f"Masked causal critic loss does not support denoising_loss_type={loss_type}"
            )

        reduce_dims = tuple(range(2, diff.dim()))
        per_frame = diff.mean(dim=reduce_dims)
        return self._masked_weighted_mean(
            per_frame,
            torch.ones_like(per_frame, dtype=per_frame.dtype),
            mask,
        ).to(target.dtype)

    def compute_distribution_matching_loss(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        video_latent_num_frames: int,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        video_loss_mask: Optional[torch.Tensor] = None,
        audio_loss_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute the DMD loss for video and audio jointly.

        Supports block-aware per-frame weighting for causal over-exposure suppression
        and causal block-wise timestep unification.

        Args:
            video_latent: Clean video latent [B, F, C, H, W]
            audio_latent: Clean audio latent [B, F_a, C]
            conditional_dict: Conditional embeddings
            unconditional_dict: Unconditional embeddings

        Returns:
            Tuple of (total_loss, log_dict)
        """

        B = video_state.clean_latent.shape[0] # video_shape[0]
        F_v = video_latent_num_frames
        F_a = audio_state.clean_latent.shape[1] # audio_shape[1]

        with torch.no_grad():
            # bidirectional dmd
            if self._is_bidirectional_task(self.generator_task_type):

                sampled_timestep = torch.randint(
                    self.min_step,
                    self.max_step + 1,
                    (B,),
                    device=self.device,
                    dtype=torch.long,
                )

                sigma = self.timestep_to_sigma(sampled_timestep)

                grad_video, grad_audio, log_dict = self._compute_kl_grad(
                    video_state=video_state,
                    audio_state=audio_state,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    video_latent_num_frames=video_latent_num_frames,
                    sigma=sigma,
                    video_loss_mask=video_loss_mask,
                    audio_loss_mask=audio_loss_mask,
                )
            else:
                sampled_timestep = torch.randint(
                    self.min_step,
                    self.max_step + 1,
                    (B,),
                    device=self.device,
                    dtype=torch.long,
                )
                sigma = self.timestep_to_sigma(sampled_timestep)
                grad_video, grad_audio, log_dict = self._compute_kl_grad(
                    video_state=video_state,
                    audio_state=audio_state,
                    conditional_dict=conditional_dict,
                    unconditional_dict=unconditional_dict,
                    video_latent_num_frames=video_latent_num_frames,
                    sigma=sigma,
                    video_loss_mask=video_loss_mask,
                    audio_loss_mask=audio_loss_mask,
                )

        # Block-aware per-frame loss weighting (over-exposure suppression)
        # video_block_w = self._compute_block_weights(F_v)  # [F_v]
        # audio_block_w = self._compute_block_weights(F_a, is_audio=True)  # [F_a]

        # Per-frame MSE then weight
        # video_diff = video_state.latent.double() - (video_state.latent.double() - grad_video.double()).detach()
        # video_per_frame = (video_diff ** 2).mean(dim=[2, 3, 4])  # [B, F_v]
        # video_loss = 0.5 * self._masked_weighted_mean(
        #     video_per_frame,
        #     video_block_w.unsqueeze(0),
        #     video_loss_mask,
        # )
        video_loss_mask = video_loss_mask.bool()
        video_dmd_loss = 0.5 * F.mse_loss(video_state.latent.double(
            )[video_loss_mask.squeeze(-1)], (video_state.latent.double() - grad_video.double()).detach()[video_loss_mask.squeeze(-1)], reduction="mean")

        # audio_diff = audio_state.latent.double() - (audio_state.latent.double() - grad_audio.double()).detach()
        # audio_per_frame = (audio_diff ** 2).mean(dim=2)  # [B, F_a]
        # audio_loss = 0.5 * self._masked_weighted_mean(
        #     audio_per_frame,
        #     audio_block_w.unsqueeze(0),
        #     audio_loss_mask,
        # )
        audio_loss_mask = audio_loss_mask.bool()
        audio_dmd_loss = 0.5 * F.mse_loss(audio_state.latent.double(
            )[audio_loss_mask.squeeze(-1)], (audio_state.latent.double() - grad_audio.double()).detach()[audio_loss_mask.squeeze(-1)], reduction="mean")

        video_w, audio_w = self.get_loss_weights()
        total_loss = video_w * video_dmd_loss + audio_w * audio_dmd_loss

        log_dict["video_dmd_loss"] = video_dmd_loss.detach()
        log_dict["audio_dmd_loss"] = audio_dmd_loss.detach()
        log_dict["video_loss_weight"] = video_w
        log_dict["audio_loss_weight"] = audio_w
        log_dict["video/audio_sigma_mean"] = sigma.float().mean().item()

        return total_loss, log_dict

    def _initialize_inference_pipeline(self):
        """Initialize the inference pipeline for backward simulation."""
        # from ltx_distillation.inference.bidirectional_pipeline import BidirectionalAVTrajectoryPipeline
        from ltx_distillation.inference.bidirectional_pipeline_ltx23 import LTX23BidirectionalAVTrajectoryPipeline

        self.inference_pipeline = LTX23BidirectionalAVTrajectoryPipeline(
            generator=self.generator,
            add_noise_fn=self.add_noise,
            denoising_sigmas=self.denoising_sigmas,
            device=self.device,
        )

    @torch.no_grad()
    def _consistency_backward_simulation_bidirectional(
        self,
        video_latent_state: LatentState,
        audio_latent_state: LatentState,
        video_latent_num_frames: int,
        text_context_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Simulate generator input using backward simulation.

        Returns trajectory of noisy inputs at each denoising step.

        Note: The generator is temporarily switched to eval() mode during
        backward simulation. This disables gradient checkpointing, which
        would otherwise conflict with FSDP under torch.no_grad() (checkpoint
        requires grad-enabled tensors). After simulation, the generator is
        restored to train() mode so that gradient checkpointing remains
        active for the subsequent gradient-enabled forward pass — essential
        for the 19B model's memory footprint.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        # Temporarily disable gradient checkpointing by switching to eval().
        # Under @torch.no_grad(), FSDP + gradient checkpointing conflicts
        # because checkpoint requires grad-enabled tensors.
        self.generator.eval()
        try:
            result = self.inference_pipeline.inference_with_trajectory(
                video_latent_state=video_latent_state,
                audio_latent_state=audio_latent_state,
                video_latent_num_frames=video_latent_num_frames,
                text_context_dict=text_context_dict,
            )
        finally:
            # Restore train() so gradient checkpointing is active for
            # the gradient-enabled generator forward pass that follows.
            self.generator.train()

        return result

    def _prepare_modality_from_state(
        self,
        video_latent_state: LatentState,
        audio_latent_state: LatentState,
        video_latent_num_frames: int,
        sigma: torch.Tensor,
        text_context_dict: Dict[str, Any],
        noisy_video_latent: Optional[torch.Tensor] = None,
        noisy_audio_latent: Optional[torch.Tensor] = None,
        ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare modality from latent state.
        if noisy latents provided, use them to replace the default latents in the corresponding modality.
        Otherwise, add noise using the given sigma to the clean latents and replace the default latents.
        [TODO]: Now only support uniform timestep in I2AV training.
        """
        B = video_latent_state.latent.shape[0]
        F_v = video_latent_num_frames # video_noise.shape[1]
        token_length_v = video_latent_state.latent.shape[1]
        token_length_a = audio_latent_state.latent.shape[1]

        video_denoise_mask = video_latent_state.denoise_mask.to(device=self.device, dtype=self.dtype)
        video_timesteps = sigma * video_denoise_mask
        audio_timesteps = sigma * audio_latent_state.denoise_mask.to(device=self.device, dtype=self.dtype)

        if noisy_video_latent is not None:
            video_noisy_input = noisy_video_latent
        else:
            original_video = video_latent_state.latent.to(device=self.device, dtype=self.dtype)
            noise_video = torch.randn_like(original_video)
            noisy_video = self.add_noise(
                original_video,
                noise_video,
                video_timesteps,
            )
            video_noisy_input = noisy_video

        if noisy_audio_latent is not None:
            audio_noisy_input = noisy_audio_latent
        else:
            original_audio = audio_latent_state.latent.to(device=self.device, dtype=self.dtype)
            noise_audio = torch.randn_like(original_audio)
            noisy_audio = self.add_noise(
                original_audio,
                noise_audio,
                audio_timesteps,
            )
            audio_noisy_input = noisy_audio

        batch_size, video_seqlen, feat_dim = video_latent_state.latent.shape 
        batch_size, audio_seqlen, feat_dim = audio_latent_state.latent.shape
        
        # 每次都拿原始视频clean的first frame替换first frame
        video_latent_model_input = video_noisy_input * video_denoise_mask + video_latent_state.clean_latent.to(device=self.device, dtype=self.dtype) * (1 - video_denoise_mask)
        audio_latent_model_input = audio_noisy_input
        
        video_modality = Modality(
            latent=video_latent_model_input.to(device=self.device, dtype=self.dtype),
            sigma=sigma.repeat(batch_size).to(device=self.device, dtype=self.dtype),
            timesteps=video_timesteps.to(device=self.device, dtype=self.dtype),
            positions=video_latent_state.positions.to(device=self.device, dtype=self.dtype),  
            context=text_context_dict['v_context'].to(device=self.device, dtype=self.dtype),
            enabled=True,
            context_mask=None,
            attention_mask=None,
        )
        audio_modality = Modality(
            latent=audio_latent_model_input.to(device=self.device, dtype=self.dtype),
            sigma=sigma.repeat(batch_size).to(device=self.device, dtype=self.dtype),
            timesteps=audio_timesteps.to(device=self.device, dtype=self.dtype),
            positions=audio_latent_state.positions.to(device=self.device, dtype=self.dtype), 
            context=text_context_dict['a_context'].to(device=self.device, dtype=self.dtype), 
            enabled=True,
            context_mask=None,
            attention_mask=None,
        )
        return video_modality, audio_modality

    def _initialize_causal_inference_pipeline(self):
        """Initialize the inference pipeline for backward simulation."""
        # from ltx_distillation.inference.bidirectional_pipeline import BidirectionalAVTrajectoryPipeline
        from ltx_distillation.inference.causal_pipeline_ltx23 import LTX23CausalAVInferencePipeline

        self.causal_inference_pipeline = LTX23CausalAVInferencePipeline(
            generator=self.generator,
            add_noise_fn=self.add_noise,
            denoising_sigmas=self.denoising_sigmas,
            num_frame_per_block=self.num_frame_per_block,
            device=self.device,
            dtype=self.dtype,
            use_kv_cache=True,
            clear_cuda_cache_per_round=True,
            accelerator=self.accelerator,
        )

    def _consistency_backward_simulation_self_forcing(
        self,
        video_latent_state,
        audio_latent_state,
        video_latent_num_frames,
        text_context_dict,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
        """
        Run generator with self-forcing rollout.

        Returns predicted clean video and audio.
        """
        if self.causal_inference_pipeline is None:
            self._initialize_causal_inference_pipeline()

        # self.generator.train()
        result = self.causal_inference_pipeline.inference_with_trajectory(
            video_latent_state=video_latent_state,
            audio_latent_state=audio_latent_state,
            video_latent_num_frames=video_latent_num_frames,
            text_context_dict=text_context_dict,
        )
        return result


    def _run_generator(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        video_latent_num_frames: int,
        text_context_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, Any]]:
        """
        Run generator with backward simulation.

        Returns predicted clean video and audio.
        """
        B = video_state.clean_latent.shape[0] # video_shape[0]
        F_v = video_latent_num_frames
        F_a = audio_state.clean_latent.shape[1] # audio_shape[1]

        video_loss_mask = None
        audio_loss_mask = None
        rollout_log: Dict[str, Any] = {}

        # bidirectional dmd train
        # Step 1: Backward simulation or ODE data
        if self._is_bidirectional_task(self.generator_task_type):
            simulated_video, simulated_audio = self._consistency_backward_simulation_bidirectional(
                video_latent_state=video_state,
                audio_latent_state=audio_state,
                video_latent_num_frames=video_latent_num_frames,
                text_context_dict=text_context_dict,
            )
            simulated_video = simulated_video.view(B, len(self.denoising_sigmas)-1, video_latent_num_frames, -1, simulated_video.shape[-1]) # 无timestep==0

            # Step 2: Random timestep selection
            num_steps = len(self.denoising_sigmas) - 1
            index = torch.randint(0, num_steps, [B, F_v], device=self.device, dtype=torch.long)
            index = self._process_timestep(index, self.generator_task_type)

            # Keep the Stage-1 bidirectional path byte-for-byte aligned with
            # the 88fb145 DMD fix semantics: one shared step per sample across
            # all video and audio frames.
            noisy_video = torch.gather(
                simulated_video,
                dim=1,
                index=index[:, :1, None, None, None].expand(-1, -1, F_v, *simulated_video.shape[-2:]),
            ).squeeze(1)
            noisy_audio = torch.gather(
                simulated_audio,
                dim=1,
                index=index[:, :1, None, None].expand(-1, -1, F_a, simulated_audio.shape[-1]),
            ).squeeze(1)

            sigma = self.denoising_sigmas[index[:, 0]]

            video_modality, audio_modality = self._prepare_modality_from_state(
                video_latent_state=video_state,
                audio_latent_state=audio_state,
                video_latent_num_frames=video_latent_num_frames,
                sigma=sigma,
                text_context_dict=text_context_dict, 
                noisy_video_latent=noisy_video.view(B, video_state.clean_latent.shape[1], -1),
                noisy_audio_latent=noisy_audio,
            )

            # Predict x0
            pred_video, pred_audio = self.generator(
                video_modality,
                audio_modality,
                perturbations=None,
            )

            video_loss_mask = video_state.denoise_mask
            audio_loss_mask = audio_state.denoise_mask

            rollout_log["dmdtrain_generator_sigma"] = sigma[0].detach().item()

        else:
            if self.enable_self_forcing:
                # pred_video, pred_audio, video_loss_mask, audio_loss_mask, rollout_log = (
                #     self._run_self_forcing_rollout(
                #         video_latent_state=video_state,
                #         audio_latent_state=audio_state,
                #         video_latent_num_frames=video_latent_num_frames,
                #         text_context_dict=text_context_dict,
                #     )
                # )
                pred_video, pred_audio, rollout_log = self._consistency_backward_simulation_self_forcing(
                    video_latent_state=video_state,
                    audio_latent_state=audio_state,
                    video_latent_num_frames=video_latent_num_frames,
                    text_context_dict=text_context_dict,
                )
                video_loss_mask = video_state.denoise_mask
                audio_loss_mask = audio_state.denoise_mask

                return pred_video, pred_audio, video_loss_mask, audio_loss_mask, rollout_log
            else:
                raise ValueError("Only support self-forcing for non-bidirectional dmd tasks.")
        
        

        return pred_video, pred_audio, video_loss_mask, audio_loss_mask, rollout_log

    def generator_loss(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        video_latent_num_frames: int,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute generator loss using DMD.

        Args:
            video_shape: [B, F, C, H, W]
            audio_shape: [B, F_a, C]
            conditional_dict: Conditional embeddings
            unconditional_dict: Unconditional embeddings
            clean_video: Clean video latent (optional, for non-backward-simulation)
            clean_audio: Clean audio latent (optional)

        Returns:
            Tuple of (loss, log_dict)
        """
        # Run generator
        pred_video, pred_audio, video_loss_mask, audio_loss_mask, rollout_log = self._run_generator(
            video_state=video_state,
            audio_state=audio_state,
            video_latent_num_frames=video_latent_num_frames,
            text_context_dict=conditional_dict,
        )

        video_state_estimated = replace(video_state, latent=pred_video)
        audio_state_estimated = replace(audio_state, latent=pred_audio)

        # Compute DMD loss
        dmd_loss, log_dict = self.compute_distribution_matching_loss(
            video_state=video_state_estimated,
            audio_state=audio_state_estimated,
            video_latent_num_frames=video_latent_num_frames,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            video_loss_mask=video_loss_mask,
            audio_loss_mask=audio_loss_mask,
        )
        log_dict.update(rollout_log)

        return dmd_loss, log_dict

    def critic_loss(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        video_latent_num_frames: int,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute critic (fake_score) loss.

        The critic learns to denoise generated samples.
        """
        # Step 1: Generate samples (no gradient)
        with torch.no_grad():
            generated_video, generated_audio, video_loss_mask, audio_loss_mask, _ = self._run_generator(
                video_state=video_state,
                audio_state=audio_state,
                video_latent_num_frames=video_latent_num_frames,
                text_context_dict=conditional_dict,
            )
        # if self._is_bidirectional_task(self.generator_task_type):
        # add noise to generated video and audio
        B = generated_video.shape[0]
        critic_timestep = torch.randint(
            0,
            self.num_train_timestep,
            (B,),
            device=self.device,
            dtype=torch.long,
        )
        critic_sigma = self.timestep_to_sigma(critic_timestep)
        video_denoise_mask = video_state.denoise_mask.to(device=self.device, dtype=self.dtype)
        video_critic_timesteps = critic_sigma * video_denoise_mask
        audio_critic_timesteps = critic_sigma * audio_state.denoise_mask.to(device=self.device, dtype=self.dtype)

        noise_video = torch.randn_like(generated_video)
        noisy_video = self.add_noise(
            generated_video,
            noise_video,
            video_critic_timesteps,
        )

        noise_audio = torch.randn_like(generated_audio)
        noisy_audio = self.add_noise(
            generated_audio,
            noise_audio,
            audio_critic_timesteps,
        )

        video_modality_cond, audio_modality_cond = self._prepare_modality_from_state(
            video_latent_state=video_state,
            audio_latent_state=audio_state,
            video_latent_num_frames=video_latent_num_frames,
            sigma=critic_sigma,
            text_context_dict=conditional_dict, 
            noisy_video_latent=noisy_video,
            noisy_audio_latent=noisy_audio,
        )

        # Step 4: Critic prediction
        pred_video, pred_audio = self.fake_score(
            # noisy_image_or_video=noisy_video,
            # conditional_dict=conditional_dict,
            # timestep=video_sigma,
            # noisy_audio=noisy_audio,
            # audio_timestep=audio_sigma,
            video=video_modality_cond,
            audio=audio_modality_cond,
            perturbations=None,
        )
        # Step 5: Compute flow matching loss for critic
        # CausVid uses flow_pred = (xt - x0_pred) / sigma, NOT simple x0 MSE.
        # The 1/sigma factor gives implicit 1/sigma^2 gradient weighting,
        # making the critic accurate at low-noise timesteps (critical for DMD).
        # Float64 for numerical stability, then cast back (matches CausVid).
        video_sigma_4d = critic_sigma.double().reshape(-1, 1, 1).clamp_min(1e-8)
        flow_pred_video = (
            (noisy_video.double() - pred_video.double())
            / video_sigma_4d
        ).to(self.dtype)

        audio_sigma_2d = critic_sigma.double().unsqueeze(-1).clamp_min(1e-8)
        flow_pred_audio = (
            (noisy_audio.double() - pred_audio.double())
            / audio_sigma_2d
        ).to(self.dtype)

        # flow_true = noise - x0 (target flow)
        video_loss = self._compute_masked_denoising_loss(
            target=generated_video,
            prediction=pred_video,
            noise=noise_video,
            flow_pred=flow_pred_video,
            timestep=video_critic_timesteps, # not used under 'velocity' mode
            mask=video_loss_mask,
        )

        audio_loss = self._compute_masked_denoising_loss(
            target=generated_audio,
            prediction=pred_audio,
            noise=noise_audio,
            flow_pred=flow_pred_audio,
            timestep=audio_critic_timesteps,  # not used under 'velocity' mode
            mask=audio_loss_mask,
        )

        video_w, audio_w = self.get_loss_weights()
        total_loss = video_w * video_loss + audio_w * audio_loss

        log_dict = {
            "critic_video_loss": video_loss.item(),
            "critic_audio_loss": audio_loss.item(),
        }

        return total_loss, log_dict

    # ------------------------------------------------------------------
    # Method-B: KV-cache-persistent segment-level training
    # ------------------------------------------------------------------

    def generator_loss_segment(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        video_latent_num_frames: int,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        persistent_kv_cache_list: Optional[list] = None,
        segment_video_offset: int = 0,
        prev_video_seqlen_frame: Optional[int] = None,
        loss_scale: float = 1.0,
        shared_exit_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any], list]:
        """
        Generator DMD loss for one segment using RELIC §4.4.2 replayed
        back-propagation. Peak memory is bounded to a single rollout-block's
        compute graph instead of the full multi-block rollout.

        Flow:
          1. Run the rollout under no_grad, capturing per-block "replay states"
             (the exit-step inputs + cursor snapshot for each block).
          2. Under no_grad, compute the full-sequence DMD score-difference map
             Δŝ = pred_fake − pred_real with normalization (same as
             compute_distribution_matching_loss does internally), detached.
          3. For each rollout block, re-run only that block's exit-step forward
             with autograd enabled, build the MSE-style surrogate loss
             0.5 * mse(x̂_l_grad[mask], (x̂_l_grad[mask] − Δŝ_l).detach()),
             scale by loss_scale, backward immediately, and free the block's
             graph before moving to the next block.

        `loss_scale` is the same factor the caller would otherwise apply to
        `loss / (accumulation_steps * seq_steps_per_update)` before its own
        `.backward()`. The caller MUST NOT call `.backward()` on the returned
        loss: this function has already accumulated parameter gradients.

        Returns (loss_for_log, log_dict, updated_kv_cache_list) where
        `loss_for_log` is a *detached* scalar tensor suitable for logging only.
        """
        if self.causal_inference_pipeline is None:
            self._initialize_causal_inference_pipeline()

        # --- Step 1: full rollout, no grad, capture per-block replay states ---
        with torch.no_grad():
            pred_video, pred_audio, rollout_log, updated_kv, replay_states = \
                self.causal_inference_pipeline.inference_with_persistent_kv_cache(
                    video_latent_state=video_state,
                    audio_latent_state=audio_state,
                    video_latent_num_frames=video_latent_num_frames,
                    text_context_dict=conditional_dict,
                    persistent_kv_cache_list=persistent_kv_cache_list,
                    segment_video_offset=segment_video_offset,
                    compute_grad=False,
                    prev_video_seqlen_frame=prev_video_seqlen_frame,
                    return_replay_state=True,
                    shared_exit_step=shared_exit_step,
                )

        video_loss_mask = video_state.denoise_mask
        audio_loss_mask = audio_state.denoise_mask

        # --- Step 2: full-sequence Δŝ under no_grad (matches the per-block
        # target the original compute_distribution_matching_loss would use).
        # We reuse compute_distribution_matching_loss to get the same MSE-style
        # loss formulation, then split it block-by-block in step 3.
        # To get Δŝ rather than only the scalar loss, we mirror the internals
        # of compute_distribution_matching_loss: sample sigma, then
        # _compute_kl_grad returns (grad_video, grad_audio) under no_grad.
        B = video_state.clean_latent.shape[0]
        video_state_for_kl = replace(video_state, latent=pred_video)
        audio_state_for_kl = replace(audio_state, latent=pred_audio)

        with torch.no_grad():
            sampled_timestep = torch.randint(
                self.min_step, self.max_step + 1, (B,),
                device=self.device, dtype=torch.long,
            )
            sigma = self.timestep_to_sigma(sampled_timestep)
            grad_video, grad_audio, log_dict = self._compute_kl_grad(
                video_state=video_state_for_kl,
                audio_state=audio_state_for_kl,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                video_latent_num_frames=video_latent_num_frames,
                sigma=sigma,
                video_loss_mask=video_loss_mask,
                audio_loss_mask=audio_loss_mask,
                segment_video_offset=segment_video_offset,
            )
            # The MSE-style loss in compute_distribution_matching_loss is
            # 0.5 * mse(latent, (latent - grad).detach()), whose gradient
            # w.r.t. `latent` is precisely `grad`. So Δŝ = grad_{video,audio}
            # (already detached because we're under no_grad).
            grad_video = grad_video.detach()
            grad_audio = grad_audio.detach()
            video_loss_mask_bool = video_loss_mask.bool().squeeze(-1)
            audio_loss_mask_bool = audio_loss_mask.bool().squeeze(-1)

        # --- Step 3: per-block replay + backward ---
        video_w, audio_w = self.get_loss_weights()
        loss_video_accum = 0.0
        loss_audio_accum = 0.0
        # Per-block normalization mirrors the full-sequence reduction='mean' MSE:
        # divide each block's contribution by the *total* active-element count so
        # summing all blocks reproduces the full-sequence mean.
        video_active_total = video_loss_mask_bool.sum().clamp_min(1).item()
        audio_active_total = audio_loss_mask_bool.sum().clamp_min(1).item()
        # video_state.latent has shape [B, T_v, C]; multiply by C so the
        # per-element mean over the [active T * C] grid matches MSE 'mean'.
        video_feat_dim = pred_video.shape[-1]
        audio_feat_dim = pred_audio.shape[-1]
        video_denom = float(video_active_total * video_feat_dim)
        audio_denom = float(audio_active_total * audio_feat_dim)

        for rs in replay_states:
            v_s = rs["video_start"]
            v_e = rs["video_end"]
            a_s = rs["audio_start"]
            a_e = rs["audio_end"]
            video_frame_tokens = pred_video.shape[1] // video_latent_num_frames
            v_tok_s = v_s * video_frame_tokens
            v_tok_e = v_e * video_frame_tokens

            # Per-block masks (sliced from the full-sequence mask).
            v_mask_blk = video_loss_mask_bool[:, v_tok_s:v_tok_e]
            a_mask_blk = audio_loss_mask_bool[:, a_s:a_e]
            if not v_mask_blk.any() and not a_mask_blk.any():
                continue

            # Per-block target Δŝ (detached).
            grad_video_blk = grad_video[:, v_tok_s:v_tok_e]
            grad_audio_blk = grad_audio[:, a_s:a_e]

            # Re-run this block's exit-step forward with autograd enabled.
            pred_video_blk, pred_audio_blk = \
                self.causal_inference_pipeline.replay_block_exit_forward(rs)

            # MSE-style surrogate: gradient w.r.t. pred_*_blk equals Δŝ_blk.
            # Sum (not mean) per-block; we divide by the full-sequence denom so
            # the aggregate matches the original mean-over-active-elements loss.
            block_loss = torch.zeros((), device=self.device, dtype=pred_video_blk.dtype)
            if v_mask_blk.any():
                target_video_blk = (pred_video_blk.double()
                                    - grad_video_blk.double()).detach()
                diff_v = (pred_video_blk.double() - target_video_blk) ** 2
                # Apply mask and sum, then normalize by full denom.
                diff_v = diff_v * v_mask_blk.unsqueeze(-1).double()
                loss_v_blk = 0.5 * diff_v.sum() / video_denom
                block_loss = block_loss + (video_w * loss_v_blk).to(block_loss.dtype)
                loss_video_accum = loss_video_accum + float(loss_v_blk.detach().item())
            if a_mask_blk.any():
                target_audio_blk = (pred_audio_blk.double()
                                    - grad_audio_blk.double()).detach()
                diff_a = (pred_audio_blk.double() - target_audio_blk) ** 2
                diff_a = diff_a * a_mask_blk.unsqueeze(-1).double()
                loss_a_blk = 0.5 * diff_a.sum() / audio_denom
                block_loss = block_loss + (audio_w * loss_a_blk).to(block_loss.dtype)
                loss_audio_accum = loss_audio_accum + float(loss_a_blk.detach().item())

            # Backward through *only this block's* compute graph, then free it.
            (block_loss * loss_scale).backward()
            del pred_video_blk, pred_audio_blk, block_loss
            torch.cuda.empty_cache()

        # Free the captured replay states (their stored Modality tensors are no
        # longer needed) before returning.
        del replay_states

        # Populate the log dict similarly to compute_distribution_matching_loss
        # so downstream logging keys exist.
        total_loss_for_log = loss_video_accum * video_w + loss_audio_accum * audio_w
        log_dict.update({
            "video_dmd_loss": torch.tensor(loss_video_accum, device=self.device),
            "audio_dmd_loss": torch.tensor(loss_audio_accum, device=self.device),
            "dmdtrain_noise_sigma": float(sigma[0].item()) if sigma.numel() > 0 else 0.0,
            "dmdtrain_clean_latent_video": pred_video.detach(),
            "dmdtrain_clean_latent_audio": pred_audio.detach(),
        })
        log_dict.update(rollout_log)

        # Return a detached scalar — caller must NOT call .backward() on this.
        loss_scalar = torch.tensor(total_loss_for_log, device=self.device)
        return loss_scalar, log_dict, updated_kv

    def critic_loss_segment(
        self,
        video_state: LatentState,
        audio_state: LatentState,
        video_latent_num_frames: int,
        conditional_dict: Dict[str, Any],
        unconditional_dict: Dict[str, Any],
        persistent_kv_cache_list: Optional[list] = None,
        segment_video_offset: int = 0,
        prev_video_seqlen_frame: Optional[int] = None,
        shared_exit_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Dict[str, Any], list]:
        """
        Critic (fake_score) loss for one segment with persistent KV cache.

        Returns (loss, log_dict, updated_kv_cache_list).
        """
        if self.causal_inference_pipeline is None:
            self._initialize_causal_inference_pipeline()

        with torch.no_grad():
            generated_video, generated_audio, rollout_log, updated_kv = \
                self.causal_inference_pipeline.inference_with_persistent_kv_cache(
                    video_latent_state=video_state,
                    audio_latent_state=audio_state,
                    video_latent_num_frames=video_latent_num_frames,
                    text_context_dict=conditional_dict,
                    persistent_kv_cache_list=persistent_kv_cache_list,
                    segment_video_offset=segment_video_offset,
                    prev_video_seqlen_frame=prev_video_seqlen_frame,
                    shared_exit_step=shared_exit_step,
                )

        video_loss_mask = video_state.denoise_mask
        audio_loss_mask = audio_state.denoise_mask

        B = generated_video.shape[0]
        critic_timestep = torch.randint(0, self.num_train_timestep, (B,), device=self.device, dtype=torch.long)
        critic_sigma = self.timestep_to_sigma(critic_timestep)
        video_denoise_mask = video_state.denoise_mask.to(device=self.device, dtype=self.dtype)
        video_critic_timesteps = critic_sigma * video_denoise_mask
        audio_critic_timesteps = critic_sigma * audio_state.denoise_mask.to(device=self.device, dtype=self.dtype)

        noise_video = torch.randn_like(generated_video)
        noisy_video = self.add_noise(generated_video, noise_video, video_critic_timesteps)
        noise_audio = torch.randn_like(generated_audio)
        noisy_audio = self.add_noise(generated_audio, noise_audio, audio_critic_timesteps)

        video_modality_cond, audio_modality_cond = self._prepare_modality_from_state(
            video_latent_state=video_state,
            audio_latent_state=audio_state,
            video_latent_num_frames=video_latent_num_frames,
            sigma=critic_sigma,
            text_context_dict=conditional_dict,
            noisy_video_latent=noisy_video,
            noisy_audio_latent=noisy_audio,
        )

        pred_video, pred_audio = self.fake_score(
            video=video_modality_cond,
            audio=audio_modality_cond,
            perturbations=None,
        )

        video_sigma_4d = critic_sigma.double().reshape(-1, 1, 1).clamp_min(1e-8)
        flow_pred_video = ((noisy_video.double() - pred_video.double()) / video_sigma_4d).to(self.dtype)
        audio_sigma_2d = critic_sigma.double().unsqueeze(-1).clamp_min(1e-8)
        flow_pred_audio = ((noisy_audio.double() - pred_audio.double()) / audio_sigma_2d).to(self.dtype)

        video_loss = self._compute_masked_denoising_loss(
            target=generated_video, prediction=pred_video, noise=noise_video,
            flow_pred=flow_pred_video, timestep=video_critic_timesteps, mask=video_loss_mask,
        )
        audio_loss = self._compute_masked_denoising_loss(
            target=generated_audio, prediction=pred_audio, noise=noise_audio,
            flow_pred=flow_pred_audio, timestep=audio_critic_timesteps, mask=audio_loss_mask,
        )

        video_w, audio_w = self.get_loss_weights()
        total_loss = video_w * video_loss + audio_w * audio_loss

        log_dict = {
            "critic_video_loss": video_loss.item(),
            "critic_audio_loss": audio_loss.item(),
            "exit_step": rollout_log.get("exit_step"),
        }

        return total_loss, log_dict, updated_kv
