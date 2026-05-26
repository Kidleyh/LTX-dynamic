"""
LTX-2 Diffusion Model Wrapper for DMD distillation.

This wrapper adapts LTX-2's audio-video joint generation model for use in
DMD (Distribution Matching Distillation) training.

Model Architecture:
- patch_size = (1, 1, 1): No spatial/temporal grouping
- Patchification: Simple reshape [B, C, F, H, W] → [B, F*H*W, C]
- Each token: 128-dimensional latent vector (one per spatial-temporal position)
- Model input projection: Linear(128, 4096)
"""

import types
from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn

from ltx_core.utils import to_denoised, to_velocity
from ltx_core.model.transformer import LTXModel, X0Model
from ltx_core.model.transformer import CausalX0Model
from ltx_core.model.transformer.modality import Modality
from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.components.patchifiers import (
    VideoLatentPatchifier,
    AudioPatchifier,
    get_pixel_coords,
)
from ltx_core.types import (
    VideoLatentShape,
    AudioLatentShape,
    SpatioTemporalScaleFactors,
)

from ltx_core.components.schedulers import LTX2Scheduler
from ltx_trainer.model_loader import load_transformer, load_causal_transformer
from ltx_distillation.utils.scheduler import SchedulerInterface
from ltx_core.model.transformer.modality import KVCache

class LTX2DiffusionWrapper(nn.Module):
    """
    Wrapper for LTX-2 model to provide DMD-compatible interface.

    Handles:
    - Input format conversion: [B, F, C, H, W] -> Modality
    - Timestep handling: sigma values for all tokens
    - Position computation for video (3D) and audio (1D)
    - Output format: x0 predictions for both video and audio

    Uses official LTX-2 patchifiers (patch_size=1) to ensure consistency
    with the pretrained model weights.
    """

    # Time alignment constants
    VIDEO_LATENT_FPS = 3.0  # 24fps / 8 (VAE compression)
    AUDIO_LATENT_FPS = 25.0  # 16kHz / 160 / 4 (mel hop / VAE compression)
    ALIGNMENT_RATIO = AUDIO_LATENT_FPS / VIDEO_LATENT_FPS  # ~8.33

    # Video FPS for position computation
    VIDEO_FPS = 24.0

    # VAE scale factors (temporal=8, height=32, width=32)
    DEFAULT_SCALE_FACTORS = SpatioTemporalScaleFactors.default()

    def __init__(
        self,
        model: LTXModel,
        # video_height: int = 512,
        # video_width: int = 768,
        vae_spatial_compression: int = 32,
    ):
        """
        Args:
            model: X0Model instance (wraps velocity model, returns x0 predictions)
            video_height: Video height in pixels
            video_width: Video width in pixels
            vae_spatial_compression: VAE spatial compression factor
        """
        super().__init__()
        self.model = model
        # self.video_height = video_height
        # self.video_width = video_width
        self.vae_spatial_compression = vae_spatial_compression

        # Compute latent dimensions
        # self.latent_height = video_height // vae_spatial_compression  # 16
        # self.latent_width = video_width // vae_spatial_compression    # 24

        # Official patchifiers with patch_size=1 (no spatial grouping)
        self.video_patchifier = VideoLatentPatchifier(patch_size=1)
        self.audio_patchifier = AudioPatchifier(patch_size=1)

        # Frame sequence length: with patch_size=1, each spatial position is one token
        # For 512x768: H'*W' = 16*24 = 384 tokens per frame
        # self.video_frame_seqlen = self.latent_height * self.latent_width  # 384
        
        # qy
        self.scheduler =  LTX2Scheduler()
        self.sigmas = self.scheduler.execute(steps=30).to(dtype=torch.float32)
        self.post_init()

        self.kv_cache = None

        self.num_transformer_blocks = len(self.model.velocity_model.transformer_blocks)  # 48

    def set_module_grad(self, module_grad: Dict[str, bool]) -> None:
        """
        Set gradient requirements for model components.

        Args:
            module_grad: Dict mapping component names to requires_grad flags
        """
        if module_grad.get("model", True):
            self.model.requires_grad_(True)
        else:
            self.model.requires_grad_(False)
            self.model.eval()

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for memory efficiency."""
        if hasattr(self.model, "velocity_model"):
            self.model.velocity_model.set_gradient_checkpointing(True)
        elif hasattr(self.model, "set_gradient_checkpointing"):
            self.model.set_gradient_checkpointing(True)

    def forward(
        self,
        video: Modality | None=None,
        audio: Modality | None=None,
        perturbations: BatchedPerturbationConfig = None,
        kv_cache_list=None,
        kv_cache_snapshot: Dict[str, int] | None = None,
    ) -> torch.Tensor:
        """
        class Modality:
            latent: (
                torch.Tensor
            )  # Shape: (B, T, D) where B is the batch size, T is the number of tokens, and D is input dimension
            sigma: torch.Tensor  # Shape: (B,). Current sigma value, used for cross-attention timestep calculation.
            timesteps: torch.Tensor  # Shape: (B, T) where T is the number of timesteps
            positions: (
                torch.Tensor
            )  # Shape: (B, 3, T) for video, where 3 is the number of dimensions and T is the number of tokens
            context: torch.Tensor
            enabled: bool = True
            context_mask: torch.Tensor | BlockMask | None = None
            attention_mask: torch.Tensor | BlockMask |None = None
        """
        # The model returns x0 predictions (X0Model wraps velocity model)
        if kv_cache_list is None:
            video_x0, audio_x0 = self.model(
                video=video,
                audio=audio,
                perturbations=perturbations,
            )
        else:
            video_x0, audio_x0 = self.model(
                video=video,
                audio=audio,
                perturbations=perturbations,
                kv_cache_list=kv_cache_list,
                kv_cache_snapshot=kv_cache_snapshot,
            )

        # Unflatten video output: [B, T, C] -> [B, F, C, H, W]
        # if video_x0 is not None:
        #     video_x0 = self._unflatten_video_latent(video_x0, num_video_frames)

        return video_x0, audio_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()

    # TODO: add ltx kv cache initialization
    def _initialize_kv_cache(self, batch_size, dtype, device, kv_cache_size: dict):
        """
        Initialize a Per-GPU KV cache for the LTX2 model.
        """
        self.kv_cache = KVCache()
        # video_self_attn_kv_cache = []
        # video_cross_attn_kv_cache = []
        # audio_self_attn_kv_cache = []
        # audio_cross_attn_kv_cache = []
        # a2v_cross_attn_kv_cache = []
        # v2a_cross_attn_kv_cache = []
        # Use the default KV cache size
        # kv_cache_size = self.num_frame_per_block * (32*32)

        self.kv_cache_list = []

        for _ in range(self.num_transformer_blocks):
            
            video_self_attn_kv_cache = {
                    "k": torch.zeros((batch_size, kv_cache_size['video_self_attn_kv_cache_size'], 4096), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, kv_cache_size['video_self_attn_kv_cache_size'], 4096), dtype=dtype, device=device),
                   }
            video_cross_attn_kv_cache = {
                    "k": [torch.zeros((batch_size, kv_cache_size['video_cross_attn_kv_cache_size'], 4096), dtype=dtype, device=device) \
                             for _ in range(kv_cache_size["num_sigmas"])],
                    "v": [torch.zeros((batch_size, kv_cache_size['video_cross_attn_kv_cache_size'], 4096), dtype=dtype, device=device) \
                             for _ in range(kv_cache_size["num_sigmas"])],
                    "is_init": False,
                   }
            audio_self_attn_kv_cache = {
                    "k": torch.zeros((batch_size, kv_cache_size['audio_self_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, kv_cache_size['audio_self_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
                   }
            audio_cross_attn_kv_cache = {
                    "k": [torch.zeros((batch_size, kv_cache_size['audio_cross_attn_kv_cache_size'], 2048), dtype=dtype, device=device) \
                                 for _ in range(kv_cache_size["num_sigmas"])],
                    "v": [torch.zeros((batch_size, kv_cache_size['audio_cross_attn_kv_cache_size'], 2048), dtype=dtype, device=device) \
                                 for _ in range(kv_cache_size["num_sigmas"])],
                    "is_init": False,
                   }
            a2v_cross_attn_kv_cache = {
                    "k": torch.zeros((batch_size, kv_cache_size['a2v_cross_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, kv_cache_size['a2v_cross_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
                   }
            v2a_cross_attn_kv_cache = {
                    "k": torch.zeros((batch_size, kv_cache_size['v2a_cross_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, kv_cache_size['v2a_cross_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
                   }

            self.kv_cache_list.append({
                "video_self_attn_kv_cache": video_self_attn_kv_cache,
                "video_cross_attn_kv_cache": video_cross_attn_kv_cache,
                "audio_self_attn_kv_cache": audio_self_attn_kv_cache,
                "audio_cross_attn_kv_cache": audio_cross_attn_kv_cache,
                "a2v_cross_attn_kv_cache": a2v_cross_attn_kv_cache,
                "v2a_cross_attn_kv_cache": v2a_cross_attn_kv_cache,
            })
            

            # video_self_attn_kv_cache.append({
            #         "k": torch.zeros((batch_size, kv_cache_size['video_self_attn_kv_cache_size'], 4096), dtype=dtype, device=device),
            #         "v": torch.zeros((batch_size, kv_cache_size['video_self_attn_kv_cache_size'], 4096), dtype=dtype, device=device),
            #        })
            # video_cross_attn_kv_cache.append({
            #         "k": torch.zeros((batch_size, kv_cache_size['video_cross_attn_kv_cache_size'], 4096), dtype=dtype, device=device),
            #         "v": torch.zeros((batch_size, kv_cache_size['video_cross_attn_kv_cache_size'], 4096), dtype=dtype, device=device),
            #        })
            # audio_self_attn_kv_cache.append({
            #         "k": torch.zeros((batch_size, kv_cache_size['audio_self_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
            #         "v": torch.zeros((batch_size, kv_cache_size['audio_self_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
            #        })
            # audio_cross_attn_kv_cache.append({
            #         "k": torch.zeros((batch_size, kv_cache_size['audio_cross_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
            #         "v": torch.zeros((batch_size, kv_cache_size['audio_cross_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
            #        })
            # a2v_cross_attn_kv_cache.append({
            #         "k": torch.zeros((batch_size, kv_cache_size['a2v_cross_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
            #         "v": torch.zeros((batch_size, kv_cache_size['a2v_cross_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
            #        })
            # v2a_cross_attn_kv_cache.append({
            #         "k": torch.zeros((batch_size, kv_cache_size['v2a_cross_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
            #         "v": torch.zeros((batch_size, kv_cache_size['v2a_cross_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
            #        })
        
        # always store the clean cache
        # different types of kv cache
        # self.kv_cache.video_self_attn_kv_cache = video_self_attn_kv_cache
        # self.kv_cache.video_cross_attn_kv_cache = video_cross_attn_kv_cache 
        # self.kv_cache.audio_self_attn_kv_cache = audio_self_attn_kv_cache 
        # self.kv_cache.audio_cross_attn_kv_cache = audio_cross_attn_kv_cache 
        # self.kv_cache.a2v_cross_attn_kv_cache = a2v_cross_attn_kv_cache 
        # self.kv_cache.v2a_cross_attn_kv_cache = v2a_cross_attn_kv_cache

        # self.video_self_attn_kv_cache = video_self_attn_kv_cache
        # self.video_cross_attn_kv_cache = video_cross_attn_kv_cache
        # self.audio_self_attn_kv_cache = audio_self_attn_kv_cache
        # self.audio_cross_attn_kv_cache = audio_cross_attn_kv_cache
        # self.a2v_cross_attn_kv_cache = a2v_cross_attn_kv_cache
        # self.v2a_cross_attn_kv_cache = v2a_cross_attn_kv_cache

        # kv cache start index
        self.kv_cache.current_video_kv_cache_start = 0
        self.kv_cache.current_audio_kv_cache_start = 0
        
        # kv cache sink len
        self.kv_cache.current_video_kv_cache_sink_len = 0 # to be initialized
        self.kv_cache.current_audio_kv_cache_sink_len = 26 # 1+25

        # current transformer block index
        self.kv_cache.current_transformer_block_index = None



    def load_state_dict(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """Load state dict, handling potential key mismatches."""
        # Remove 'model.' prefix if present
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_state_dict[k] = v
            else:
                new_state_dict[f"model.{k}"] = v

        super().load_state_dict(new_state_dict, strict=strict)


def create_causal_ltx2_wrapper(
    checkpoint_path: str,
    gemma_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    # video_height: int = 512,
    # video_width: int = 768,
    use_flex_attention: bool = True,
    registry = None,
) -> LTX2DiffusionWrapper:
    """
    Factory function to create LTX2DiffusionWrapper from checkpoint.

    Args:
        checkpoint_path: Path to LTX-2 checkpoint
        gemma_path: Path to Gemma text encoder
        device: Target device
        dtype: Model dtype
        video_height: Video height
        video_width: Video width

    Returns:
        Configured LTX2DiffusionWrapper
    """
    # from ltx_pipelines.utils.model_ledger import ModelLedger

    # # IMPORTANT: Load to CPU first, then move to target device
    # # safetensors doesn't support device indices like "cuda:4"
    # # It only accepts "cuda" or "cpu"
    # ledger = ModelLedger(
    #     dtype=dtype,
    #     device=torch.device("cpu"),  # Load to CPU first
    #     checkpoint_path=checkpoint_path,
    #     gemma_root_path=gemma_path,
    # )

    # # Get X0Model (wraps velocity model)
    # x0_model = ledger.transformer()

    # # Move to target device
    # x0_model = x0_model.to(device=device, dtype=dtype)

    velocity_model = load_causal_transformer(checkpoint_path, device, dtype, use_flex_attention=use_flex_attention)
    x0_model = CausalX0Model(velocity_model)

    wrapper = LTX2DiffusionWrapper(
        model=x0_model,
        # video_height=video_height,
        # video_width=video_width,
    )

    return wrapper

def create_ltx2_wrapper(
    checkpoint_path: str,
    gemma_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    video_height: int = 512,
    video_width: int = 768,
    registry = None,
) -> LTX2DiffusionWrapper:
    """
    Factory function to create LTX2DiffusionWrapper from checkpoint.

    Args:
        checkpoint_path: Path to LTX-2 checkpoint
        gemma_path: Path to Gemma text encoder
        device: Target device
        dtype: Model dtype
        video_height: Video height
        video_width: Video width

    Returns:
        Configured LTX2DiffusionWrapper
    """
    # from ltx_pipelines.utils.model_ledger import ModelLedger

    # # IMPORTANT: Load to CPU first, then move to target device
    # # safetensors doesn't support device indices like "cuda:4"
    # # It only accepts "cuda" or "cpu"
    # ledger = ModelLedger(
    #     dtype=dtype,
    #     device=torch.device("cpu"),  # Load to CPU first
    #     checkpoint_path=checkpoint_path,
    #     gemma_root_path=gemma_path,
    # )

    # # Get X0Model (wraps velocity model)
    # x0_model = ledger.transformer()

    # # Move to target device
    # x0_model = x0_model.to(device=device, dtype=dtype)

    velocity_model = load_transformer(checkpoint_path, device, dtype)
    x0_model = X0Model(velocity_model)

    wrapper = LTX2DiffusionWrapper(
        model=x0_model,
        # video_height=video_height,
        # video_width=video_width,
    )

    return wrapper
