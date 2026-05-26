from enum import Enum
from typing import Any, Dict, Optional, Tuple

import torch

from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.model.transformer.adaln import AdaLayerNormSingle, adaln_embedding_coefficient
from ltx_core.model.transformer.attention import AttentionCallable, AttentionFunction
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.causal_transformer import CausalBasicAVTransformerBlock, TransformerConfig
from ltx_core.model.transformer.transformer_args import (
    MultiModalTransformerArgsPreprocessor,
    TransformerArgs,
    TransformerArgsPreprocessor,
)
from ltx_core.utils import to_denoised

import torch.distributed as dist
import math
from torch.nn.attention.flex_attention import create_block_mask
from torch.nn.attention.flex_attention import BlockMask
from dataclasses import replace
from ltx_core.model.transformer.modality import KVCache
from ltx_causal.attention.mask_builder import AVCausalMaskBuilder, build_all_causal_masks
from ltx_causal.config import CausalMaskConfig

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
        attention_type: AttentionFunction | AttentionCallable | None = None,  # AttentionFunction.DEFAULT
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
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        double_precision_rope: bool = False,
        apply_gated_attention: bool = False,
        caption_projection: torch.nn.Module | None = None,
        audio_caption_projection: torch.nn.Module | None = None,
        cross_attention_adaln: bool = False,
    ):
        super().__init__()
        self._enable_gradient_checkpointing = False
        self.cross_attention_adaln = cross_attention_adaln
        self.use_middle_indices_grid = use_middle_indices_grid
        self.rope_type = rope_type
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
        # Initialize transformer blocks masks
        # self.block_mask_video = {}
        # self.block_mask_audio = {}
        # self.block_mask_a2v = {}
        # self.block_mask_v2a = {}
        self.block_mask = {}

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
        attention_type: AttentionFunction | AttentionCallable | None,
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
                CausalBasicAVTransformerBlock(
                    idx=idx,
                    video=video_config,
                    audio=audio_config,
                    rope_type=self.rope_type,
                    norm_eps=norm_eps,
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
        video: TransformerArgs | None,
        audio: TransformerArgs | None,
        perturbations: BatchedPerturbationConfig,
        kv_cache_list=None,
        kv_cache_snapshot: Dict[str, int] | None = None,
    ) -> tuple[TransformerArgs, TransformerArgs]:
        """Process transformer blocks for LTXAV."""

        # Process transformer blocks
        for block_index, block in enumerate(self.transformer_blocks):
            if kv_cache_list is not None:
                blk_video_self = kv_cache_list[block_index]["video_self_attn_kv_cache"]
                blk_video_cross = kv_cache_list[block_index]["video_cross_attn_kv_cache"]
                blk_audio_self = kv_cache_list[block_index]["audio_self_attn_kv_cache"]
                blk_audio_cross = kv_cache_list[block_index]["audio_cross_attn_kv_cache"]
                blk_a2v = kv_cache_list[block_index]["a2v_cross_attn_kv_cache"]
                blk_v2a = kv_cache_list[block_index]["v2a_cross_attn_kv_cache"]

                # Per-block gradient checkpointing also for the KV-cache path during
                # training. Without it, all 48 blocks' activations stay resident and
                # the recompute under the outer checkpoint OOMs. The cache writes
                # inside causal_attention run under torch.no_grad() and use
                # detached k/v, so replaying the forward writes the same values to
                # the same positions — safe to recompute.
                #
                # use_reentrant=False matches both the no-cache branch above and the
                # outer rollout-block checkpoint in inference_with_persistent_kv_cache.
                # Reentrant flavor cannot accept dataclass inputs (TransformerArgs).
                if self._enable_gradient_checkpointing and self.training:
                    def _block_call(v_in, a_in, perts, _blk=block,
                                    _snap=kv_cache_snapshot,
                                    _vs=blk_video_self, _vc=blk_video_cross,
                                    _as=blk_audio_self, _ac=blk_audio_cross,
                                    _a2v=blk_a2v, _v2a=blk_v2a):
                        return _blk(
                            video=v_in,
                            audio=a_in,
                            perturbations=perts,
                            kv_cache_snapshot=_snap,
                            video_self_attn_kv_cache=_vs,
                            video_cross_attn_kv_cache=_vc,
                            audio_self_attn_kv_cache=_as,
                            audio_cross_attn_kv_cache=_ac,
                            a2v_cross_attn_kv_cache=_a2v,
                            v2a_cross_attn_kv_cache=_v2a,
                        )

                    video, audio = torch.utils.checkpoint.checkpoint(
                        _block_call,
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
                        kv_cache_snapshot=kv_cache_snapshot,
                        video_self_attn_kv_cache=blk_video_self,
                        video_cross_attn_kv_cache=blk_video_cross,
                        audio_self_attn_kv_cache=blk_audio_self,
                        audio_cross_attn_kv_cache=blk_audio_cross,
                        a2v_cross_attn_kv_cache=blk_a2v,
                        v2a_cross_attn_kv_cache=blk_v2a,
                    )
            else:
                # ode training
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
                        kv_cache_snapshot=None,
                        video_self_attn_kv_cache=None,
                        video_cross_attn_kv_cache=None,        
                        audio_self_attn_kv_cache=None,
                        audio_cross_attn_kv_cache=None,
                        a2v_cross_attn_kv_cache=None,
                        v2a_cross_attn_kv_cache=None,
                    )
        # if kv_cache is not None:
        #     kv_cache.is_initialized = True
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

    def forward(
        self, video: Modality | None, audio: Modality | None, perturbations: BatchedPerturbationConfig,
        spatial_compression_ratio: int = 32, temporal_compression_ratio: int = 8, resolution: tuple[int, int] = (768, 512),
        num_frame_per_block: int = 3,
        kv_cache_list=None,
        kv_cache_snapshot: Dict[str, int] | None = None,
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
        
        # video_block_mask
        video_frame_seqlen_cur = resolution[0] * resolution[1]  // (spatial_compression_ratio * spatial_compression_ratio)
        audio_frame_seqlen_cur = 25 / num_frame_per_block
        video_block_seqlen_cur = video_frame_seqlen_cur * num_frame_per_block
        audio_block_seqlen_cur = 25 # 8.333  # count
        
        # import cv2
        # cv2.imwrite("test_video_selfmask.jpg", self.block_mask_video[video_frame_seqlen_cur].to_dense()[0].permute(1,2,0).detach().cpu().numpy()*255)
        # cv2.imwrite("test_audio_selfmask.jpg", self.block_mask_audio[video_frame_seqlen_cur].to_dense()[0].permute(1,2,0).detach().cpu().numpy()*255)
        # cv2.imwrite("test_a2v_selfmask.jpg", self.block_mask_a2v[video_frame_seqlen_cur].to_dense()[0].permute(1,2,0).detach().cpu().numpy()*255)
        # cv2.imwrite("test_v2a_selfmask.jpg", self.block_mask_v2a[video_frame_seqlen_cur].to_dense()[0].permute(1,2,0).detach().cpu().numpy()*255)

        if kv_cache_list is None:
            # using attention mask to achieve causal attention
            if video_frame_seqlen_cur not in self.block_mask:
                # self.block_mask_video[video_frame_seqlen_cur] = self._prepare_block_mask_video(
                #     video.latent.device, num_frames=video.latent.shape[1] // video_frame_seqlen_cur,
                #     block_seqlen=video_block_seqlen_cur,
                #     num_frame_per_block=num_frame_per_block, max_token_size=video.latent.shape[1],
                # )
                # self.block_mask_audio[video_frame_seqlen_cur] = self._prepare_block_mask_audio(
                #     audio.latent.device, num_frames=video.latent.shape[1] // video_frame_seqlen_cur,
                #     block_seqlen=audio_block_seqlen_cur,
                #     num_frame_per_block=num_frame_per_block, max_token_size=audio.latent.shape[1],
                # )
                # self.block_mask_a2v[video_frame_seqlen_cur] = self._prepare_block_mask_cross(
                #     video.latent.device, num_frames=video.latent.shape[1] // video_frame_seqlen_cur,
                #     block_seqlen_q=video_block_seqlen_cur, block_seqlen_kv=audio_block_seqlen_cur,
                #     block_seqlen_firstblock_q=video_frame_seqlen_cur, block_seqlen_firstblock_kv=1,
                #     num_frame_per_block=num_frame_per_block, max_token_size_q=video.latent.shape[1], max_token_size_kv=audio.latent.shape[1],
                # )
                # self.block_mask_v2a[video_frame_seqlen_cur] = self._prepare_block_mask_cross(
                #     audio.latent.device, num_frames=video.latent.shape[1] // video_frame_seqlen_cur,
                #     block_seqlen_q=audio_block_seqlen_cur, block_seqlen_kv=video_block_seqlen_cur,
                #     block_seqlen_firstblock_q=1, block_seqlen_firstblock_kv=video_frame_seqlen_cur,
                #     num_frame_per_block=num_frame_per_block, max_token_size_q=audio.latent.shape[1], max_token_size_kv=video.latent.shape[1],
                #     a2v=False,
                # )
                mask_builder = AVCausalMaskBuilder(
                    video_frame_seqlen=video_frame_seqlen_cur,
                    audio_frame_seqlen=1,
                    num_frame_per_block=3,
                    num_frame_per_block_first=4,
                )

                num_video_frames = video.latent.shape[1] // video_frame_seqlen_cur # video_grid_sizes[0, 0].item()
                num_audio_frames = audio.latent.shape[1] # audio_grid_sizes[0, 0].item()
                mask_config = CausalMaskConfig(
                    video_frame_seqlen=video_frame_seqlen_cur,
                    num_frame_per_block=3,
                    num_frame_per_block_first=4,
                )
                self.block_mask[video_frame_seqlen_cur] = build_all_causal_masks(
                    num_video_frames, num_audio_frames,
                    config=mask_config,
                    device=video.latent.device,
                )


        video_args = self.video_args_preprocessor.prepare(video, audio) if video is not None else None
        audio_args = self.audio_args_preprocessor.prepare(audio, video) if audio is not None else None

        if kv_cache_list is None:
            # video_args = replace(video_args, self_attention_mask=self.block_mask_video[video_frame_seqlen_cur], multimodel_attention_mask=self.block_mask_a2v[video_frame_seqlen_cur])
            # audio_args = replace(audio_args, self_attention_mask=self.block_mask_audio[video_frame_seqlen_cur], multimodel_attention_mask=self.block_mask_v2a[video_frame_seqlen_cur])
            video_args = replace(video_args, 
                                 self_attention_mask=self.block_mask[video_frame_seqlen_cur]['video_self'], 
                                 multimodel_attention_mask=self.block_mask[video_frame_seqlen_cur]['a2v'])
            audio_args = replace(audio_args, 
                                 self_attention_mask=self.block_mask[video_frame_seqlen_cur]['audio_self'], 
                                 multimodel_attention_mask=self.block_mask[video_frame_seqlen_cur]['v2a'])

        # Process transformer blocks
        video_out, audio_out = self._process_transformer_blocks(
            video=video_args,
            audio=audio_args,
            perturbations=perturbations,
            kv_cache_list=kv_cache_list,
            kv_cache_snapshot=kv_cache_snapshot,
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
    
    
    @staticmethod
    def _prepare_block_mask_video(
            device: torch.device | str, 
            num_frames: int = 22,          # 例如 N=7 时，1 + 3*7 = 22帧
            block_seqlen: int = 1560, 
            num_frame_per_block: int = 3,
            max_token_size: int = -1,
        ) -> "BlockMask":
            
        # 【修复点 1】: 提前计算单帧的 token 长度 (例如 1560 // 3 = 520)
        frame_seqlen = block_seqlen // num_frame_per_block
            
        # 如果未传入 max_token_size，按帧数计算
        if max_token_size == -1:
            total_length = num_frames * frame_seqlen
        else:
            total_length = max_token_size
            
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        # Initialize step and region mappings for valid tokens only
        block_id = torch.zeros(total_length + padded_length, device=device, dtype=torch.long) - 1

        # 【修复点 2: 重新设计 Block 划分逻辑】
        # 第 0 帧独立成为 idx == 0 的 block
        block_id[0 : frame_seqlen] = 0
        
        # 计算除了第 0 帧外，后续还需要划分出多少个 block
        num_subsequent_blocks = math.ceil((num_frames - 1) / num_frame_per_block)
        
        for b in range(1, num_subsequent_blocks + 1):
            # start 起点：第 0 帧长度 + 前面已排完的 (b-1) 个完整 block 长度
            start = frame_seqlen + (b - 1) * block_seqlen
            
            # 处理最后一个 block：计算当前 block 实际包含几帧（防止最后一组不满 num_frame_per_block）
            remain_frames = num_frames - 1 - (b - 1) * num_frame_per_block
            cur_block_frames = min(num_frame_per_block, remain_frames)
            
            # end 终点：起点 + 当前 block 的实际长度
            end = start + cur_block_frames * frame_seqlen
            
            # 安全机制：防止 max_token_size 截断导致的越界
            if start >= total_length:
                break
            end = min(end, total_length)
            
            block_id[start:end] = b

        def attention_mask(b, h, q_idx, kv_idx):
            q_block, kv_block = block_id[q_idx], block_id[kv_idx]

            # 1. Self-attention always allowed
            eye = (q_idx == kv_idx)

            # 2. Block self-attention allowed
            block_self_allowed = (q_block == kv_block)

            # 3. 全局锚点：后续每个 block 都能看到 idx==0 和 idx==1 的 block
            global_anchor_allowed = (kv_block == 0) | (kv_block == 1)

            # 4. attention to the prev block
            prev_block_allowed = (kv_block == q_block - 1)

            # 5. attention to the prev prev block
            prev_prev_block_allowed = (kv_block == q_block - 2)

            # 合并所有允许的 Attention 规则
            allowed_attention = eye | block_self_allowed | global_anchor_allowed | prev_block_allowed | prev_prev_block_allowed
            
            # 防止 padding 区域出现越界 attention，保留原有 -1 的屏蔽
            valid_tokens = (q_block != -1) & (kv_block != -1)

            return allowed_attention & valid_tokens

        block_mask = create_block_mask(
            attention_mask, B=None, H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False, device=device,
            BLOCK_SIZE=64,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(block_mask)

        return block_mask

    # def _prepare_block_mask_video(
    #     device: torch.device | str, num_frames: int = 21,
    #     frame_seqlen: int = 1560, num_frame_per_block=3,
    #     debug=False, max_token_size: int = -1,
    # ) -> "BlockMask":
    #     total_length = max_token_size
    #     padded_length = math.ceil(total_length / 128) * 128 - total_length

    #     # Initialize step and region mappings for valid tokens only
    #     block_id = torch.zeros(total_length + padded_length, device=device, dtype=torch.long) - 1

    #     for f in range(num_frames): # num_frame_per_block needs to be times of BLOCK_SIZE
    #         cur_block_id = f // num_frame_per_block
    #         start, end = f * frame_seqlen, (f + 1) * frame_seqlen
    #         block_id[start:end] = cur_block_id

    #     def attention_mask(b, h, q_idx, kv_idx):
    #         q_block, kv_block = block_id[q_idx], block_id[kv_idx]

    #         # Self-attention always allowed
    #         eye = (q_idx == kv_idx)

    #         # block self-attention allowed
    #         block_self_allowed = (q_block == kv_block)

    #         # always attention to first block(except first block itself)
    #         first_block_allowed = (kv_block == 0)  &  (q_block != num_frames // num_frame_per_block) # (q_block == (kv_block - num_frames // num_frame_per_block))

    #         # attention to the exact prev block before cur block (with teacher forcing)
    #         prev_block_allowed = (kv_block > 0) & (kv_block == q_block - 1)

    #         # attention to the prev prev block before cur block
    #         prev_prev_block_allowed = (kv_block > 0) & (kv_block == q_block - 2)

    #         # Apply blocking rule: remove attention to last clean frame for step 2 queries
    #         allowed_attention = eye | block_self_allowed | first_block_allowed | prev_block_allowed | prev_prev_block_allowed

    #         return allowed_attention

    #     block_mask = create_block_mask(
    #         attention_mask, B=None, H=None,
    #         Q_LEN=total_length + padded_length,
    #         KV_LEN=total_length + padded_length,
    #         _compile=False, device=device,
    #         BLOCK_SIZE=64,
    #     )

    #     if not dist.is_initialized() or dist.get_rank() == 0:
    #         print(block_mask)

    #     return block_mask
    
    @staticmethod
    def _prepare_block_mask_audio(
        device: torch.device | str, num_frames: int = 22,
        block_seqlen: int = 75, num_frame_per_block=3,
        max_token_size: int = -1,
    ) -> "BlockMask":

        # 计算基准单帧 token 长度 (向下取整，仅用于最后填不满的尾部块)
        frame_seqlen = block_seqlen // num_frame_per_block

        # 剥离第 0 帧后，计算后续的常规帧数
        regular_frames = num_frames - 1
        
        # 核心修复 1：计算有多少个完整的 block，以及尾部剩下几帧
        full_blocks = regular_frames // num_frame_per_block
        rem_frames = regular_frames % num_frame_per_block

        # 核心修复 2：如果未传入 max_token_size，按完整块与碎块精确拼接总长
        if max_token_size == -1:
            # 总长 = 第0个Token(1) + (完整块数 * block_seqlen) + (尾部剩余帧 * 单帧长度)
            total_length = 1 + (full_blocks * block_seqlen) + (rem_frames * frame_seqlen)
        else:
            total_length = max_token_size
            
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        # 初始化映射
        block_id = torch.zeros(total_length + padded_length, device=device, dtype=torch.long) - 1

        # 第 0 个 block 只有 1 个 token
        block_id[0 : 1] = 0
        
        num_subsequent_blocks = math.ceil(regular_frames / num_frame_per_block)
        
        for b in range(1, num_subsequent_blocks + 1):
            # start 依然按标准的 block_seqlen 跨步
            start = 1 + (b - 1) * block_seqlen
            
            remain_frames = regular_frames - (b - 1) * num_frame_per_block
            cur_block_frames = min(num_frame_per_block, remain_frames)
            
            # 核心修复 3：动态计算当前 block 的终点
            if cur_block_frames == num_frame_per_block:
                # 如果是完整的 block，无视整除误差，强行占据整个 block_seqlen
                end = start + block_seqlen
            else:
                # 如果是尾部填不满的 block，按实际帧数计算
                end = start + cur_block_frames * frame_seqlen
            
            # 安全越界保护
            if start >= total_length:
                break
            end = min(end, total_length)
            
            block_id[start:end] = b

        def attention_mask(b, h, q_idx, kv_idx):
            q_block, kv_block = block_id[q_idx], block_id[kv_idx]

            # 1. Self-attention
            eye = (q_idx == kv_idx)

            # 2. Block self-attention
            block_self_allowed = (q_block == kv_block)

            # 3. 全局锚点 (idx==0 和 idx==1)
            global_anchor_allowed = (kv_block == 0) | (kv_block == 1)

            # 4. Prev block
            prev_block_allowed = (kv_block == q_block - 1)

            # 5. Prev prev block
            prev_prev_block_allowed = (kv_block == q_block - 2)

            allowed_attention = eye | block_self_allowed | global_anchor_allowed | prev_block_allowed | prev_prev_block_allowed
            valid_tokens = (q_block != -1) & (kv_block != -1)

            return allowed_attention & valid_tokens

        block_mask = create_block_mask(
            attention_mask, B=None, H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False, device=device,
            BLOCK_SIZE=1,
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(block_mask)

        return block_mask

    @staticmethod
    def _prepare_block_mask_cross(
        device: torch.device | str, 
        num_frames: int = 22, 
        block_seqlen_q: int = 1560, 
        block_seqlen_kv: int = 1560, 
        block_seqlen_firstblock_q: int = 1560,
        block_seqlen_firstblock_kv: int = 1560,
        num_frame_per_block: int = 3,
        max_token_size_q: int = -1, 
        max_token_size_kv: int = -1, 
        a2v: bool = True,  # 保留参数以防破坏上游接口调用，但内部不再需要区分
    ) -> "BlockMask":
        
        # 计算基准单帧 token 长度 (向下取整，仅用于最后填不满的尾部块)
        frame_seqlen_q = block_seqlen_q // num_frame_per_block
        frame_seqlen_kv = block_seqlen_kv // num_frame_per_block

        # 剥离第 0 帧后，计算后续的常规帧数
        regular_frames = num_frames - 1
        
        # 计算有多少个完整的 block，以及尾部剩下几帧
        full_blocks = regular_frames // num_frame_per_block
        rem_frames = regular_frames % num_frame_per_block

        # 1. 完全对称且精确地计算序列总长度 (处理不能整除的问题)
        if max_token_size_q == -1:
            total_length_q = block_seqlen_firstblock_q + (full_blocks * block_seqlen_q) + (rem_frames * frame_seqlen_q)
        else:
            total_length_q = max_token_size_q
            
        if max_token_size_kv == -1:
            total_length_kv = block_seqlen_firstblock_kv + (full_blocks * block_seqlen_kv) + (rem_frames * frame_seqlen_kv)
        else:
            total_length_kv = max_token_size_kv

        padded_length_q = math.ceil(total_length_q / 128) * 128 - total_length_q
        padded_length_kv = math.ceil(total_length_kv / 128) * 128 - total_length_kv

        # 2. 初始化映射 Tensor (-1 表示 Padding 区域)
        block_id_q = torch.zeros(total_length_q + padded_length_q, device=device, dtype=torch.long) - 1
        block_id_kv = torch.zeros(total_length_kv + padded_length_kv, device=device, dtype=torch.long) - 1

        # 3. 完全对称地构建 Q 和 KV 的 Block ID 映射
        # 第 0 帧独立成为 idx == 0 的 block，使用传入的 firstblock 长度
        block_id_q[0 : block_seqlen_firstblock_q] = 0
        block_id_kv[0 : block_seqlen_firstblock_kv] = 0
        
        # 计算后续需要划分出多少个 block
        num_subsequent_blocks = math.ceil(regular_frames / num_frame_per_block)

        for b in range(1, num_subsequent_blocks + 1):
            # 当前 block 实际包含几帧（防尾部越界）
            cur_block_frames = min(num_frame_per_block, regular_frames - (b - 1) * num_frame_per_block)
            
            # ================= 处理 Q 侧 =================
            start_q = block_seqlen_firstblock_q + (b - 1) * block_seqlen_q
            if cur_block_frames == num_frame_per_block:
                # 完整的 block，无视整除误差，强行占据整个 block_seqlen_q
                end_q = start_q + block_seqlen_q
            else:
                # 尾部填不满的 block，按实际帧数计算
                end_q = start_q + cur_block_frames * frame_seqlen_q
                
            if start_q < total_length_q:
                end_q = min(end_q, total_length_q)
                block_id_q[start_q:end_q] = b
                
            # ================= 处理 KV 侧 =================
            start_kv = block_seqlen_firstblock_kv + (b - 1) * block_seqlen_kv
            if cur_block_frames == num_frame_per_block:
                # 完整的 block，无视整除误差，强行占据整个 block_seqlen_kv
                end_kv = start_kv + block_seqlen_kv
            else:
                # 尾部填不满的 block，按实际帧数计算
                end_kv = start_kv + cur_block_frames * frame_seqlen_kv
                
            if start_kv < total_length_kv:
                end_kv = min(end_kv, total_length_kv)
                block_id_kv[start_kv:end_kv] = b
        
        # 4. 定义跨模态 Attention 规则
        def attention_mask(b, h, q_idx, kv_idx):
            q_block, kv_block = block_id_q[q_idx], block_id_kv[kv_idx]

            # 跨模态全局锚点：看到另一个模态的 idx==0 和 idx==1
            global_anchor_allowed = (kv_block == 0) | (kv_block == 1)

            # 跨模态【当前】同步：看到对应的当前 Block
            current_block_allowed = (q_block == kv_block)

            # 跨模态【前面】：看到前一个 Block
            prev_block_allowed = (kv_block == q_block - 1)

            # 跨模态【前前面】：看到前前一个 Block
            prev_prev_block_allowed = (kv_block == q_block - 2)

            # 规则合并
            allowed_attention = global_anchor_allowed | current_block_allowed | prev_block_allowed | prev_prev_block_allowed
            
            # 过滤 padding 区域
            valid_tokens = (q_block != -1) & (kv_block != -1)

            return allowed_attention & valid_tokens

        # 5. 生成底层的 BlockMask
        block_mask = create_block_mask(
            attention_mask, B=None, H=None,
            Q_LEN=total_length_q + padded_length_q,
            KV_LEN=total_length_kv + padded_length_kv,
            _compile=False, device=device,
            BLOCK_SIZE=1, 
        )

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(block_mask)

        return block_mask

    # def _prepare_block_mask_cross(
    #     device: torch.device | str, num_frames: int = 21,
    #     frame_seqlen_q: int = 1560, frame_seqlen_kv: int = 1560, num_frame_per_block=3,
    #     debug=False, max_token_size_q: int = -1, max_token_size_kv: int = -1, a2v: bool = True,
    # ) -> "BlockMask":
    #     total_length_q = max_token_size_q
    #     total_length_kv = max_token_size_kv
    #     padded_length_q = math.ceil(total_length_q / 128) * 128 - total_length_q # need padding
    #     padded_length_kv = math.ceil(total_length_kv / 128) * 128 - total_length_kv # need padding

    #     # Initialize step and region mappings for valid tokens only
    #     block_id_q = torch.zeros(total_length_q + padded_length_q, device=device, dtype=torch.long) - 1
    #     block_id_kv = torch.zeros(total_length_kv + padded_length_kv, device=device, dtype=torch.long) - 1

    #     if a2v:
    #         for f in range(num_frames): # num_frame_per_block needs to be times of BLOCK_SIZE
    #             cur_block_id = f // num_frame_per_block
    #             start, end = f * frame_seqlen_q, (f + 1) * frame_seqlen_q
    #             block_id_q[start:end] = cur_block_id

    #         for f in range(num_frames): # num_frame_per_block needs to be times of BLOCK_SIZE
    #             cur_block_id = f // num_frame_per_block
    #             if f == 0:
    #                 start, end = 0, 1
    #             else:
    #                 start, end = math.floor(f * frame_seqlen_kv - 7), math.floor((f + 1) * frame_seqlen_kv - 7)  # 7 is because of the first block
    #             block_id_kv[start:end] = cur_block_id
    #     else:
    #         for f in range(num_frames): # num_frame_per_block needs to be times of BLOCK_SIZE
    #             cur_block_id = f // num_frame_per_block
    #             if f == 0:
    #                 start, end = 0, 1
    #             else:
    #                 start, end = math.floor(f * frame_seqlen_q - 7), math.floor((f + 1) * frame_seqlen_q - 7)  # 7 is because of the first block
    #             block_id_q[start:end] = cur_block_id
    #         for f in range(num_frames): # num_frame_per_block needs to be times of BLOCK_SIZE
    #             cur_block_id = f // num_frame_per_block
    #             start, end = f * frame_seqlen_kv, (f + 1) * frame_seqlen_kv
    #             block_id_kv[start:end] = cur_block_id
        
    #     def attention_mask(b, h, q_idx, kv_idx):
    #         q_block, kv_block = block_id_q[q_idx], block_id_kv[kv_idx]

    #         # Self-attention always allowed
    #         eye = (q_idx == kv_idx)

    #         # block self-attention allowed
    #         block_self_allowed = (q_block == kv_block)

    #         # always attention to first block(except first block itself)
    #         first_block_allowed = (kv_block == 0)  &  (q_block != num_frames // num_frame_per_block) # (q_block == (kv_block - num_frames // num_frame_per_block))

    #         # attention to the exact prev block before cur block (with teacher forcing)
    #         prev_block_allowed = (kv_block > 0) & (kv_block == q_block - 1)

    #         # attention to the prev prev block before cur block
    #         prev_prev_block_allowed = (kv_block > 0) & (kv_block == q_block - 2)

    #         # Apply blocking rule: remove attention to last clean frame for step 2 queries
    #         allowed_attention = eye | block_self_allowed | first_block_allowed | prev_block_allowed | prev_prev_block_allowed

    #         return allowed_attention

    #     block_mask = create_block_mask(
    #         attention_mask, B=None, H=None,
    #         Q_LEN=total_length_q + padded_length_q,
    #         KV_LEN=total_length_kv + padded_length_kv,
    #         _compile=False, device=device,
    #         BLOCK_SIZE=1,
    #     )

    #     if not dist.is_initialized() or dist.get_rank() == 0:
    #         print(block_mask)

    #     return block_mask


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
        kv_cache_list=None,
        kv_cache_snapshot: Dict[str, int] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """
        Denoise the video and audio according to the sigma.
        Returns:
            Denoised video and audio
        """
        vx, ax = self.velocity_model(video, 
                                     audio, 
                                     perturbations, 
                                     kv_cache_list=kv_cache_list, 
                                     kv_cache_snapshot=kv_cache_snapshot)
        denoised_video = to_denoised(video.latent, vx, video.timesteps) if vx is not None else None
        denoised_audio = to_denoised(audio.latent, ax, audio.timesteps) if ax is not None else None
        return denoised_video, denoised_audio
