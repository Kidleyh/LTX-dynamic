from enum import Enum
from dataclasses import dataclass, replace
import torch
import torch.nn as nn

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.adaln import AdaLayerNormSingle, adaln_embedding_coefficient
from ltx_core.model.transformer.attention import AttentionCallable, AttentionFunction
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import LTXRopeType
# from ltx_core.model.transformer.transformer import BasicAVTransformerBlock, TransformerConfig
# from ltx_core.model.transformer.transformer_args import (
#     MultiModalTransformerArgsPreprocessor,
#     TransformerArgs,
#     TransformerArgsPreprocessor,
# )
from ltx_core.utils import to_denoised

from ltx_causal.config import (
    CausalMaskConfig,
    VIDEO_LATENT_FPS,
    AUDIO_LATENT_FPS,
)
from ltx_causal.attention.mask_builder import (
    AVCausalMaskBuilder,
    build_all_causal_masks,
    compute_aligned_audio_frames,
    compute_av_blocks,
    compute_causal_log_scales,
)
from ltx_causal.transformer.causal_block import (
    CausalAVTransformerBlock,
    TransformerConfig,
    CausalTransformerArgs,
    rms_norm,
    MultiModalTransformerArgsPreprocessor,
    TransformerArgsPreprocessor,
    TransformerArgs,
)
from ltx_causal.transformer.compat import (
    AdaLayerNormSingle,
    PixArtAlphaTextProjection,
)
from ltx_causal.rope.causal_rope import (
    CausalRopeType,
    causal_precompute_freqs_cis,
)

from typing import Optional, Tuple


@dataclass
class CausalLTXModelConfig:
    """Configuration for CausalLTXModel."""

    # Model dimensions
    num_layers: int = 48
    video_dim: int = 4096
    audio_dim: int = 2048
    video_heads: int = 32
    audio_heads: int = 32
    video_d_head: int = 128
    audio_d_head: int = 64

    # Cross-attention context dimension
    cross_attention_dim: int = 4096
    audio_cross_attention_dim: int = 2048  # Also used as inner_dim for cross-modal RoPE

    # Patch embedding (LTX-2 uses patch_size=1 with nn.Linear)
    in_channels: int = 128
    out_channels: int = 128
    patch_size: Tuple[int, int, int] = (1, 1, 1)

    # Caption (text) projection
    caption_channels: int = 3840  # Gemma text encoder output dim

    # Position embedding
    pe_theta: float = 10000.0
    pe_max_pos: Tuple[int, int, int] = (20, 2048, 2048)
    audio_pe_max_pos: Tuple[int] = (20,)

    # Timestep embedding
    timestep_scale_multiplier: int = 1000
    av_ca_timestep_scale_multiplier: int = 1

    # Normalization
    norm_eps: float = 1e-6
    
    # [NOTE]: causal 
    # Causal generation
    num_frame_per_block: int = 3

    # Audio sink tokens
    num_audio_sink_tokens: int = 16 # 16 0

    # RoPE
    rope_type: CausalRopeType = CausalRopeType.INTERLEAVED

    # Token sizes
    # [NOTE]: should support multi-resolution training
    video_frame_seqlen: int = 384  # For 512x768: (512/32)*(768/32)
    audio_frame_seqlen: int = 1

    # Log-ratio entropy-aligned rescaling for causal attention outputs.
    # When True, each token's causal attention output is scaled by
    # log(1 + visible_tokens) / log(1 + total_tokens), compensating for
    # the information deficit caused by causal masking vs bidirectional.
    # No learnable parameters — purely structural rescaling.
    enable_causal_log_rescale: bool = False


class LTXModelType(Enum):
    AudioVideo = "ltx av model"
    VideoOnly = "ltx video only model"
    AudioOnly = "ltx audio only model"

    def is_video_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.VideoOnly)

    def is_audio_enabled(self) -> bool:
        return self in (LTXModelType.AudioVideo, LTXModelType.AudioOnly)


class CausalLTXModel(torch.nn.Module):
    """
    LTX model transformer implementation.
    This class implements the transformer blocks for the LTX model.
    """

    def __init__(  # noqa: PLR0913
        self,
        *,
        model_type: LTXModelType = LTXModelType.AudioVideo,
        num_attention_heads: int = 32,
        attention_head_dim: int = 128,
        in_channels: int = 128,
        out_channels: int = 128,
        num_layers: int = 48,
        cross_attention_dim: int = 4096,
        norm_eps: float = 1e-06,
        attention_type: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
        positional_embedding_theta: float = 10000.0,
        positional_embedding_max_pos: list[int] | None = None,
        timestep_scale_multiplier: int = 1000,
        use_middle_indices_grid: bool = True,
        audio_num_attention_heads: int = 32,
        audio_attention_head_dim: int = 64,
        audio_in_channels: int = 128,
        audio_out_channels: int = 128,
        audio_cross_attention_dim: int = 2048,
        audio_positional_embedding_max_pos: list[int] | None = None,
        av_ca_timestep_scale_multiplier: int = 1,
        rope_type: CausalRopeType = CausalRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        caption_projection: torch.nn.Module | None = None,
        audio_caption_projection: torch.nn.Module | None = None,
        cross_attention_adaln: bool = False,
    ):
        super().__init__()

        # causal non-causal区别：
        # 1. rope type == CausalRopeType.INTERLEAVED(causal)/SPLIT(non-causal)
        # 2. attention: apply_rope / if mask is enabled
        # 3. if forward of causalmodel includes causal preprocessing
        # 4. 
        self.rope_type = CausalRopeType.INTERLEAVED # [NOTE]: forced to be causal
        # self.rope_type = rope_type

        self.config = CausalLTXModelConfig()
        self.num_audio_sink_tokens = self.config.num_audio_sink_tokens # strange bug fix

        self._enable_gradient_checkpointing = False
        self.cross_attention_adaln = cross_attention_adaln
        self.use_middle_indices_grid = use_middle_indices_grid
        
        self.double_precision_rope = double_precision_rope
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.positional_embedding_theta = positional_embedding_theta
        self.model_type = model_type
        cross_pe_max_pos = None
        if model_type.is_video_enabled():
            if positional_embedding_max_pos is None:
                positional_embedding_max_pos = [20, 2048, 2048]
            self.positional_embedding_max_pos = positional_embedding_max_pos
            self.num_attention_heads = num_attention_heads
            self.inner_dim = num_attention_heads * attention_head_dim
            self._init_video(
                in_channels=in_channels,
                out_channels=out_channels,
                norm_eps=norm_eps,
                caption_projection=caption_projection,
            )

        if model_type.is_audio_enabled():
            if audio_positional_embedding_max_pos is None:
                audio_positional_embedding_max_pos = [20]
            self.audio_positional_embedding_max_pos = audio_positional_embedding_max_pos
            self.audio_num_attention_heads = audio_num_attention_heads
            self.audio_inner_dim = self.audio_num_attention_heads * audio_attention_head_dim
            self._init_audio(
                in_channels=audio_in_channels,
                out_channels=audio_out_channels,
                norm_eps=norm_eps,
                caption_projection=audio_caption_projection,
            )

        if model_type.is_video_enabled() and model_type.is_audio_enabled():
            cross_pe_max_pos = max(self.positional_embedding_max_pos[0], self.audio_positional_embedding_max_pos[0])
            self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier
            self.audio_cross_attention_dim = audio_cross_attention_dim
            self._init_audio_video(num_scale_shift_values=4)

        self._init_preprocessors(cross_pe_max_pos)
        # Initialize transformer blocks
        self._init_transformer_blocks(
            num_layers=num_layers,
            attention_head_dim=attention_head_dim if model_type.is_video_enabled() else 0,
            cross_attention_dim=cross_attention_dim,
            audio_attention_head_dim=audio_attention_head_dim if model_type.is_audio_enabled() else 0,
            audio_cross_attention_dim=audio_cross_attention_dim,
            norm_eps=norm_eps,
            attention_type=attention_type,
            apply_gated_attention=apply_gated_attention,
        )

        # === Audio Sink Tokens ===
        if self.config.num_audio_sink_tokens > 0:
            self.audio_sink_tokens = nn.Parameter(
                torch.zeros(1, self.config.num_audio_sink_tokens, self.config.audio_dim)
            )
            nn.init.normal_(self.audio_sink_tokens, std=0.02)
        
        self.caual_masks = {}


    @property
    def _adaln_embedding_coefficient(self) -> int:
        return adaln_embedding_coefficient(self.cross_attention_adaln)

    def _init_video(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize video-specific components."""
        # Video input components
        self.patchify_proj = torch.nn.Linear(in_channels, self.inner_dim, bias=True)
        if caption_projection is not None:
            self.caption_projection = caption_projection

        self.adaln_single = AdaLayerNormSingle(self.inner_dim, embedding_coefficient=self._adaln_embedding_coefficient)

        self.prompt_adaln_single = (
            AdaLayerNormSingle(self.inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        # Video output components
        self.scale_shift_table = torch.nn.Parameter(torch.empty(2, self.inner_dim))
        self.norm_out = torch.nn.LayerNorm(self.inner_dim, elementwise_affine=False, eps=norm_eps)
        self.proj_out = torch.nn.Linear(self.inner_dim, out_channels)

    def _init_audio(
        self,
        in_channels: int,
        out_channels: int,
        norm_eps: float,
        caption_projection: torch.nn.Module | None = None,
    ) -> None:
        """Initialize audio-specific components."""

        # Audio input components
        self.audio_patchify_proj = torch.nn.Linear(in_channels, self.audio_inner_dim, bias=True)
        if caption_projection is not None:
            self.audio_caption_projection = caption_projection

        self.audio_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=self._adaln_embedding_coefficient,
        )

        self.audio_prompt_adaln_single = (
            AdaLayerNormSingle(self.audio_inner_dim, embedding_coefficient=2) if self.cross_attention_adaln else None
        )

        # Audio output components
        self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(2, self.audio_inner_dim))
        self.audio_norm_out = torch.nn.LayerNorm(self.audio_inner_dim, elementwise_affine=False, eps=norm_eps)
        self.audio_proj_out = torch.nn.Linear(self.audio_inner_dim, out_channels)

    def _init_audio_video(
        self,
        num_scale_shift_values: int,
    ) -> None:
        """Initialize audio-video cross-attention components."""
        self.av_ca_video_scale_shift_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_audio_scale_shift_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=num_scale_shift_values,
        )

        self.av_ca_a2v_gate_adaln_single = AdaLayerNormSingle(
            self.inner_dim,
            embedding_coefficient=1,
        )

        self.av_ca_v2a_gate_adaln_single = AdaLayerNormSingle(
            self.audio_inner_dim,
            embedding_coefficient=1,
        )

    def _init_preprocessors(
        self,
        cross_pe_max_pos: int | None = None,
    ) -> None:
        """Initialize preprocessors for LTX."""

        if self.model_type.is_video_enabled() and self.model_type.is_audio_enabled():
            self.video_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                cross_scale_shift_adaln=self.av_ca_video_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_a2v_gate_adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
            self.audio_args_preprocessor = MultiModalTransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                cross_scale_shift_adaln=self.av_ca_audio_scale_shift_adaln_single,
                cross_gate_adaln=self.av_ca_v2a_gate_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                cross_pe_max_pos=cross_pe_max_pos,
                use_middle_indices_grid=self.use_middle_indices_grid,
                audio_cross_attention_dim=self.audio_cross_attention_dim,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                av_ca_timestep_scale_multiplier=self.av_ca_timestep_scale_multiplier,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )
        elif self.model_type.is_video_enabled():
            self.video_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.patchify_proj,
                adaln=self.adaln_single,
                inner_dim=self.inner_dim,
                max_pos=self.positional_embedding_max_pos,
                num_attention_heads=self.num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "caption_projection", None),
                prompt_adaln=getattr(self, "prompt_adaln_single", None),
            )
        elif self.model_type.is_audio_enabled():
            self.audio_args_preprocessor = TransformerArgsPreprocessor(
                patchify_proj=self.audio_patchify_proj,
                adaln=self.audio_adaln_single,
                inner_dim=self.audio_inner_dim,
                max_pos=self.audio_positional_embedding_max_pos,
                num_attention_heads=self.audio_num_attention_heads,
                use_middle_indices_grid=self.use_middle_indices_grid,
                timestep_scale_multiplier=self.timestep_scale_multiplier,
                double_precision_rope=self.double_precision_rope,
                positional_embedding_theta=self.positional_embedding_theta,
                rope_type=self.rope_type,
                caption_projection=getattr(self, "audio_caption_projection", None),
                prompt_adaln=getattr(self, "audio_prompt_adaln_single", None),
            )

    def _init_transformer_blocks(
        self,
        num_layers: int,
        attention_head_dim: int,
        cross_attention_dim: int,
        audio_attention_head_dim: int,
        audio_cross_attention_dim: int,
        norm_eps: float,
        attention_type: AttentionFunction | AttentionCallable,
        apply_gated_attention: bool,
    ) -> None:
        """Initialize transformer blocks for LTX."""
        video_config = (
            TransformerConfig(
                dim=self.inner_dim,
                heads=self.num_attention_heads,
                d_head=attention_head_dim,
                context_dim=cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_video_enabled()
            else None
        )
        audio_config = (
            TransformerConfig(
                dim=self.audio_inner_dim,
                heads=self.audio_num_attention_heads,
                d_head=audio_attention_head_dim,
                context_dim=audio_cross_attention_dim,
                apply_gated_attention=apply_gated_attention,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self.model_type.is_audio_enabled()
            else None
        )
        self.transformer_blocks = torch.nn.ModuleList(
            [
                CausalAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=self.config.norm_eps,
                    attention_function=attention_type,
                )
                for idx in range(num_layers)
            ]
        )

    def set_gradient_checkpointing(self, enable: bool) -> None:
        """Enable or disable gradient checkpointing for transformer blocks.
        Gradient checkpointing trades compute for memory by recomputing activations
        during the backward pass instead of storing them. This can significantly
        reduce memory usage at the cost of ~20-30% slower training.
        Args:
            enable: Whether to enable gradient checkpointing
        """
        self._enable_gradient_checkpointing = enable

    def _process_transformer_blocks(
        self,
        video: CausalTransformerArgs | None,
        audio: CausalTransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[CausalTransformerArgs, CausalTransformerArgs]:
        """Process transformer blocks for LTXAV."""

        # Process transformer blocks
        for block in self.transformer_blocks:
            if self._enable_gradient_checkpointing and self.training:
                # Use gradient checkpointing to save memory during training.
                # With use_reentrant=False, we can pass dataclasses directly -
                # PyTorch will track all tensor leaves in the computation graph.
                video, audio = torch.utils.checkpoint.checkpoint(
                    block,
                    video,
                    audio,
                    perturbations,
                    use_reentrant=False,
                )
            else:
                video, audio = block(
                    video=video,
                    audio=audio,
                    perturbations=perturbations,
                )

        # === Strip sink tokens before output processing ===
        audio_out_x = audio.x
        num_sink = self.num_audio_sink_tokens
        if num_sink > 0:
            audio_out_x = audio_out_x[:, num_sink:]
        audio = replace(audio, x=audio_out_x)


        return video, audio

    def _process_output(
        self,
        scale_shift_table: torch.Tensor,
        norm_out: torch.nn.LayerNorm,
        proj_out: torch.nn.Linear,
        x: torch.Tensor,
        embedded_timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Process output for LTXV."""
        # Apply scale-shift modulation
        scale_shift_values = (
            scale_shift_table[None, None].to(device=x.device, dtype=x.dtype) + embedded_timestep[:, :, None]
        )
        shift, scale = scale_shift_values[:, :, 0], scale_shift_values[:, :, 1]

        x = norm_out(x)
        x = x * (1 + scale) + shift
        x = proj_out(x)
        return x

    def _prepare_modality_causal_interactions(
        self, video: CausalTransformerArgs | None, audio: CausalTransformerArgs | None,
        video_modality: Modality | None, audio_modality: Modality | None,
    ) -> tuple[CausalTransformerArgs | None, CausalTransformerArgs | None]:
        """
        Prepare modality-interactions for LTXAV.
        Returns:
            Processed video and audio arguments with causal mask / log scales
        """
        if video is None or audio is None:
            return video, audio

        # === Patch Embedding ===
        # Video: [B, F, C, H, W] → patchify → [B, T, C] → project → [B, T, D]
        device = video.x.device
        hidden_dtype = video.x.dtype
        B_v, F_v, C_v, H_v, W_v = video.original_shape.batch, \
                                  video.original_shape.frames, \
                                  video.original_shape.channels, \
                                  video.original_shape.height, \
                                  video.original_shape.width
        video_grid_sizes = torch.tensor([F_v, H_v, W_v], device=device).unsqueeze(0)

        # Audio: [B, F_a, C] -> [B, F_a, audio_dim]
        F_a_original = audio.x.shape[1]
        audio_grid_sizes = torch.tensor([F_a_original], device=device).unsqueeze(0)

        # sink related setting 
        # 1. expand audio x with sink
        num_sink = self.config.num_audio_sink_tokens
        if num_sink > 0:
            sink_expanded = self.audio_sink_tokens.expand(B_v, -1, -1).to(audio.x.dtype)
            audio_latent_with_sink = torch.cat([sink_expanded, audio.x], dim=1)

        # 2. Expand audio timestep with sink entries (same as first frame's timestep)
        if num_sink > 0: #  and audio.timesteps.ndim == 2
            # audio_ts is [B, F_a], prepend num_sink copies of first frame's value
            sink_ts = audio_modality.timesteps[:, :1].repeat(1, num_sink, 1)  # [B, num_sink]
            audio_ts_expanded = torch.cat([sink_ts, audio_modality.timesteps], dim=1)  # [B, num_sink + F_a]

        # 3. Save original audio embedded timestep (without sinks) for output
        audio_timestep_6d, audio_embedded_ts_full = self.audio_args_preprocessor.simple_preprocessor._prepare_timestep(
            audio_ts_expanded, self.audio_args_preprocessor.simple_preprocessor.adaln, B_v, hidden_dtype
        )

        # Strip sink entries from embedded timestep for output processing
        if num_sink > 0 and audio_embedded_ts_full.shape[1] > 1:
            audio_embedded_ts = audio_embedded_ts_full[:, num_sink:]
        else:
            audio_embedded_ts = audio_embedded_ts_full

        audio_cross_ss, audio_cross_gate = self.audio_args_preprocessor._prepare_cross_attention_timestep(
            timestep=audio_ts_expanded,
            timestep_scale_multiplier=self.audio_args_preprocessor.simple_preprocessor.timestep_scale_multiplier,
            batch_size=audio.x.shape[0],
            hidden_dtype=hidden_dtype,
        )

        # audio_cross_ss, audio_cross_gate = self.audio_args_preprocessor._prepare_cross_attention_timestep(
        #     audio_ts_expanded,
        #     self.av_ca_audio_scale_shift_adaln_single,
        #     self.av_ca_v2a_gate_adaln_single,
        #     B_v, hidden_dtype,
        # )

        # === Expand per-frame timestep embeddings to per-token ===
        # When timesteps is [B, F_v] (per-frame), AdaLN output is [B, F_v, *]
        # but transformer blocks need [B, F_v*H*W, *] (per-token).
        # When timesteps is [B] (scalar), output is [B, 1, *] which broadcasts.
        # Audio has 1 token/frame so no expansion needed.
        # frame_seqlen = H_v * W_v
        # video_timestep_6d = self._expand_per_frame_to_per_token(video_timestep_6d, frame_seqlen)
        # video_embedded_ts = self._expand_per_frame_to_per_token(video_embedded_ts, frame_seqlen)
        # video_cross_ss = self._expand_per_frame_to_per_token(video_cross_ss, frame_seqlen)
        # video_cross_gate = self._expand_per_frame_to_per_token(video_cross_gate, frame_seqlen)

        # Reuse wrapper-provided masks whenever available. The wrapper already
        # builds them with num_audio_sink_tokens=num_sink, so sink tokens do not
        # require an unconditional rebuild here.
        video_frame_seqlen = H_v * W_v # self.config.video_frame_seqlen # TODO: multiple mask config

        def draw_attention_mask(block_mask):
            import cv2
            cv2.imwrite("test.jpg", block_mask.to_dense()[0].permute(1,2,0).detach().cpu().numpy()*255)

        if video_frame_seqlen not in self.caual_masks:
            num_video_frames = video_grid_sizes[0, 0].item()
            num_audio_frames = audio_grid_sizes[0, 0].item()  # Original count (without sinks)
            mask_config = CausalMaskConfig(
                video_frame_seqlen=video_frame_seqlen,
                num_frame_per_block=self.config.num_frame_per_block,
                num_audio_sink_tokens=num_sink,
            )
            masks = build_all_causal_masks(
                num_video_frames, num_audio_frames,
                config=mask_config,
                device=device,
            )
            self.caual_masks[video_frame_seqlen] = masks

        # === Compute log-ratio scales for causal attention rescaling ===
        log_scales = None
        if self.config.enable_causal_log_rescale:
            blocks = compute_av_blocks(
                F_v, self.config.num_frame_per_block,
            )
            log_scales = compute_causal_log_scales(
                blocks,
                video_frame_seqlen=video_frame_seqlen,
                audio_frame_seqlen=self.config.audio_frame_seqlen,
                device=device,
                num_audio_sink_tokens=num_sink,
            )

        # === Precompute RoPE ===
        video_pe = causal_precompute_freqs_cis(
            video_grid_sizes, self.config.video_d_head * self.config.video_heads,
            theta=self.config.pe_theta, max_pos=list(self.config.pe_max_pos),
            start_frame=0, rope_type=self.config.rope_type,
            device=device, dtype=audio.x.dtype,
        )

        audio_pe = causal_precompute_freqs_cis(
            audio_grid_sizes, self.config.audio_d_head * self.config.audio_heads,
            theta=self.config.pe_theta, max_pos=list(self.config.audio_pe_max_pos),
            start_frame=0, rope_type=self.config.rope_type,
            device=device, dtype=audio.x.dtype,
            is_audio=True,
        )

        # Prepend identity RoPE for sink tokens (cos=1, sin=0 → no rotation)
        if num_sink > 0:
            audio_rope_dim = audio_pe[0].shape[-1]
            sink_cos = torch.ones(1, num_sink, audio_rope_dim, device=device, dtype=audio_pe[0].dtype)
            sink_sin = torch.zeros(1, num_sink, audio_rope_dim, device=device, dtype=audio_pe[1].dtype)
            audio_pe = (
                torch.cat([sink_cos, audio_pe[0]], dim=1),
                torch.cat([sink_sin, audio_pe[1]], dim=1),
            )

        # === Cross-attention RoPE ===
        # Original uses 1D temporal-only positions at audio_cross_attention_dim (2048).
        # cross_pe_max_pos = max(pe_max_pos[0], audio_pe_max_pos[0]) = max(20, 20) = 20
        cross_pe_max_pos = max(
            self.config.pe_max_pos[0],
            self.config.audio_pe_max_pos[0],
        )
        # Video cross-PE: temporal positions from video grid (1D, video temporal)
        video_temporal_grid = torch.tensor(
            [[F_v]], device=device, dtype=torch.long
        )  # [1, 1]
        video_cross_pe = causal_precompute_freqs_cis(
            video_temporal_grid,
            self.config.audio_cross_attention_dim,
            theta=self.config.pe_theta,
            max_pos=[cross_pe_max_pos],
            start_frame=0,
            rope_type=self.config.rope_type,
            device=device, dtype=video.x.dtype,
            is_audio=False,  # Video temporal conversion
        )
        # Expand temporal PE to full video sequence: each frame's tokens share same temporal PE
        # video_cross_pe: [B, F_v, D] → need [B, F_v*H*W, D]
        video_cross_pe = (
            video_cross_pe[0].unsqueeze(2).expand(-1, -1, video_frame_seqlen, -1)
            .reshape(1, -1, video_cross_pe[0].shape[-1]),
            video_cross_pe[1].unsqueeze(2).expand(-1, -1, video_frame_seqlen, -1)
            .reshape(1, -1, video_cross_pe[1].shape[-1]),
        )

        # Audio cross-PE: temporal positions from audio grid (1D, audio temporal)
        # Use original audio frame count (without sinks) for cross-PE computation
        audio_temporal_grid = torch.tensor(
            [[F_a_original]], device=device, dtype=torch.long
        )  # [1, 1]
        audio_cross_pe = causal_precompute_freqs_cis(
            audio_temporal_grid,
            self.config.audio_cross_attention_dim,
            theta=self.config.pe_theta,
            max_pos=[cross_pe_max_pos],
            start_frame=0,
            rope_type=self.config.rope_type,
            device=device, dtype=audio.x.dtype,
            is_audio=True,  # Audio temporal conversion
        )

        # Prepend identity RoPE for sink tokens in cross-PE
        if num_sink > 0:
            cross_rope_dim = audio_cross_pe[0].shape[-1]
            sink_cross_cos = torch.ones(1, num_sink, cross_rope_dim, device=device, dtype=audio_cross_pe[0].dtype)
            sink_cross_sin = torch.zeros(1, num_sink, cross_rope_dim, device=device, dtype=audio_cross_pe[1].dtype)
            audio_cross_pe = (
                torch.cat([sink_cross_cos, audio_cross_pe[0]], dim=1),
                torch.cat([sink_cross_sin, audio_cross_pe[1]], dim=1),
            )

        video = replace(
            video,
            positional_embeddings=video_pe,
            cross_positional_embeddings=video_cross_pe,
            block_mask=None, # self.caual_masks[video_frame_seqlen].get('video_self'), # 
            cross_causal_mask=None, # self.caual_masks[video_frame_seqlen].get('a2v'), # 
            self_attn_log_scale=log_scales['video_self_scale'].to(hidden_dtype) if log_scales else None, # 
            cross_attn_log_scale=log_scales['a2v_scale'].to(hidden_dtype) if log_scales else None, # 
        )

        audio = replace(
            audio,
            x=audio_latent_with_sink,
            timesteps=audio_timestep_6d,
            cross_scale_shift_timestep=audio_cross_ss,
            cross_gate_timestep=audio_cross_gate,
            positional_embeddings=audio_pe,
            cross_positional_embeddings=audio_cross_pe,
            block_mask=None, # self.caual_masks[video_frame_seqlen].get('audio_self'), # 
            cross_causal_mask=None, # self.caual_masks[video_frame_seqlen].get('v2a'), # 
            self_attn_log_scale=log_scales['audio_self_scale'].to(hidden_dtype) if log_scales else None, # 
            cross_attn_log_scale=log_scales['v2a_scale'].to(hidden_dtype) if log_scales else None, # 
        )

        return video, audio

    def forward(
        self, video: Modality | None, audio: Modality | None, perturbations: BatchedPerturbationConfig
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for LTX models.
        Returns:
            Processed output tensors
        """
        if not self.model_type.is_video_enabled() and video is not None:
            raise ValueError("Video is not enabled for this model")
        if not self.model_type.is_audio_enabled() and audio is not None:
            raise ValueError("Audio is not enabled for this model")

        video_args = self.video_args_preprocessor.prepare(video, audio) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None

        # get modality-interactions for causal inference
        if type(video_args) == CausalTransformerArgs and type(audio_args) == CausalTransformerArgs:
            video_args, audio_args = self._prepare_modality_causal_interactions(video_args, audio_args, video, audio)

        # Process transformer blocks
        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
        )

        # Process output
        vx = (
            self._process_output(
                self.scale_shift_table, self.norm_out, self.proj_out, video_out.x, video_out.embedded_timestep
            )
            if video_out is not None
            else None
        )
        ax = (
            self._process_output(
                self.audio_scale_shift_table,
                self.audio_norm_out,
                self.audio_proj_out,
                audio_out.x,
                audio_out.embedded_timestep,
            )
            if audio_out is not None
            else None
        )
        return vx, ax


class CausalLegacyX0Model(torch.nn.Module):
    """
    Legacy X0 model implementation.
    Returns fully denoised output based on the velocities produced by the base model.
    """

    def __init__(self, velocity_model: CausalLTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
        sigma: float,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, sigma) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, sigma) if ax is not None else None
        return denoised_video, denoised_audio


class CausalX0Model(torch.nn.Module):
    """
    X0 model implementation.
    Returns fully denoised outputs based on the velocities produced by the base model.
    Applies scaled denoising to the video and audio according to the timesteps = sigma * denoising_mask.
    """

    def __init__(self, velocity_model: CausalLTXModel):
        super().__init__()
        self.velocity_model = velocity_model

    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, audio, perturbations)
        denoised_video = to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None
        return denoised_video, denoised_audio
