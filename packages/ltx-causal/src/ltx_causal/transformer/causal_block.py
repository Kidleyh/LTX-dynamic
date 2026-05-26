from dataclasses import dataclass, replace

import torch
import torch.nn as nn

from ltx_core.guidance.perturbations import BatchedPerturbationConfig, PerturbationType
from ltx_core.model.transformer.adaln import adaln_embedding_coefficient
from ltx_core.model.transformer.attention import Attention, AttentionCallable, AttentionFunction
# from ltx_core.model.transformer.feed_forward import FeedForward
from ltx_core.model.transformer.rope import LTXRopeType
from ltx_core.model.transformer.transformer_args import TransformerArgs
from ltx_core.utils import rms_norm

from ltx_causal.attention.causal_attention import CausalLTXAttention
# from ltx_causal.transformer.causal_block import (
#     CausalAVTransformerBlock,
#     CausalTransformerArgs,
#     TransformerConfig,
#     rms_norm,
# )
from ltx_causal.rope.causal_rope import CausalRopeType, causal_precompute_freqs_cis
from ltx_causal.transformer.compat import FeedForward

# Try to import BlockMask type
try:
    from torch.nn.attention.flex_attention import BlockMask
except ImportError:
    BlockMask = None

from ltx_core.model.transformer.adaln import AdaLayerNormSingle
from ltx_core.model.transformer.modality import Modality
from ltx_core.model.transformer.rope import (
    generate_freq_grid_np,
    generate_freq_grid_pytorch,
    precompute_freqs_cis,
)

from typing import Optional, Tuple
from ltx_core.types import AudioLatentShape, VideoLatentShape

@dataclass
class TransformerConfig:
    dim: int
    heads: int
    d_head: int
    context_dim: int
    apply_gated_attention: bool = False
    cross_attention_adaln: bool = False


@dataclass
class CausalTransformerArgs:
    """
    Arguments for causal transformer forward pass (training only).
    """
    x: torch.Tensor                           # Hidden states [B, L, D]
    timesteps: torch.Tensor                   # Timestep embeddings for AdaLN
    embedded_timestep: torch.Tensor
    positional_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None  # RoPE
    context: Optional[torch.Tensor] = None    # Text context
    context_mask: Optional[torch.Tensor] = None
    enabled: bool = True

    # Cross-attention RoPE (for A2V/V2A)
    cross_positional_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    cross_scale_shift_timestep: Optional[torch.Tensor] = None
    cross_gate_timestep: Optional[torch.Tensor] = None

    prompt_timestep: torch.Tensor | None = None
    self_attention_mask: torch.Tensor | None = (
        None  # Additive log-space self-attention bias (B, 1, T, T), None = full attention
    )

    # Causal masks (training only)
    block_mask: Optional["BlockMask"] = None  # For self-attention
    cross_causal_mask: Optional[torch.Tensor] = None  # For cross-attention

    # Log-ratio scales for causal attention output (entropy-aligned rescaling)
    self_attn_log_scale: Optional[torch.Tensor] = None   # [1, L, 1]
    cross_attn_log_scale: Optional[torch.Tensor] = None  # [1, L, 1]
    original_shape: VideoLatentShape | AudioLatentShape | None = None


class CausalAVTransformerBlock(torch.nn.Module):
    def __init__(
        self,
        idx: int,
        video: TransformerConfig | None = None,
        audio: TransformerConfig | None = None,
        rope_type: LTXRopeType | CausalRopeType = None,
        norm_eps: float = 1e-6,
        attention_function: AttentionFunction | AttentionCallable = AttentionFunction.DEFAULT,
        # Kept in signature for backward-compatible construction but unused
        local_attn_size: int = 16,
        sink_size: int = 1,
    ):
        super().__init__()

        self.idx = idx

        self._store_gate_stats = True
        self._gate_stats = {}
        # Curriculum learning: skip A2V/V2A cross-modal attention
        self.skip_cross_modal_attention = False

        if video is not None:
            self.attn1 = CausalLTXAttention(
                query_dim=video.dim,
                heads=video.heads,
                dim_head=video.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.attn2 = CausalLTXAttention(
                query_dim=video.dim,
                context_dim=video.context_dim,
                heads=video.heads,
                dim_head=video.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )
            self.ff = FeedForward(video.dim, dim_out=video.dim)
            video_sst_size = adaln_embedding_coefficient(video.cross_attention_adaln)
            self.scale_shift_table = torch.nn.Parameter(torch.empty(video_sst_size, video.dim))

        if audio is not None:
            self.audio_attn1 = CausalLTXAttention(
                query_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                context_dim=None,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_attn2 = CausalLTXAttention(
                query_dim=audio.dim,
                context_dim=audio.context_dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )
            self.audio_ff = FeedForward(audio.dim, dim_out=audio.dim)
            audio_sst_size = adaln_embedding_coefficient(audio.cross_attention_adaln)
            self.audio_scale_shift_table = torch.nn.Parameter(torch.empty(audio_sst_size, audio.dim))

        if audio is not None and video is not None:
            # Q: Video, K,V: Audio
            self.audio_to_video_attn = CausalLTXAttention(
                query_dim=video.dim,
                context_dim=audio.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=video.apply_gated_attention,
            )

            # Q: Audio, K,V: Video
            self.video_to_audio_attn = CausalLTXAttention(
                query_dim=audio.dim,
                context_dim=video.dim,
                heads=audio.heads,
                dim_head=audio.d_head,
                rope_type=rope_type,
                norm_eps=norm_eps,
                attention_function=attention_function,
                apply_gated_attention=audio.apply_gated_attention,
            )

            self.scale_shift_table_a2v_ca_audio = torch.nn.Parameter(torch.empty(5, audio.dim))
            self.scale_shift_table_a2v_ca_video = torch.nn.Parameter(torch.empty(5, video.dim))

        # [NOTE]: ltx2.3 new
        self.cross_attention_adaln = (video is not None and video.cross_attention_adaln) or (
            audio is not None and audio.cross_attention_adaln
        )
        if self.cross_attention_adaln and video is not None:
            self.prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, video.dim))
        if self.cross_attention_adaln and audio is not None:
            self.audio_prompt_scale_shift_table = torch.nn.Parameter(torch.empty(2, audio.dim))
        self.norm_eps = norm_eps

    def get_ada_values(
        self, scale_shift_table: torch.Tensor, batch_size: int, timestep: torch.Tensor, indices: slice
    ) -> tuple[torch.Tensor, ...]:
        num_ada_params = scale_shift_table.shape[0]

        ada_values = (
            scale_shift_table[indices].unsqueeze(0).unsqueeze(0).to(device=timestep.device, dtype=timestep.dtype)
            + timestep.reshape(batch_size, timestep.shape[1], num_ada_params, -1)[:, :, indices, :]
        ).unbind(dim=2)
        return ada_values

    def get_av_ca_ada_values(
        self,
        scale_shift_table: torch.Tensor,
        batch_size: int,
        scale_shift_timestep: torch.Tensor,
        gate_timestep: torch.Tensor,
        scale_shift_indices: slice,
        num_scale_shift_values: int = 4,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale_shift_ada_values = self.get_ada_values(
            scale_shift_table[:num_scale_shift_values, :], batch_size, scale_shift_timestep, scale_shift_indices
        )
        gate_ada_values = self.get_ada_values(
            scale_shift_table[num_scale_shift_values:, :], batch_size, gate_timestep, slice(None, None)
        )

        scale, shift = (t.squeeze(2) for t in scale_shift_ada_values)
        (gate,) = (t.squeeze(2) for t in gate_ada_values)

        return scale, shift, gate

    def _apply_text_cross_attention(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        attn: AttentionCallable,
        scale_shift_table: torch.Tensor,
        prompt_scale_shift_table: torch.Tensor | None,
        timestep: torch.Tensor,
        prompt_timestep: torch.Tensor | None,
        context_mask: torch.Tensor | None,
        cross_attention_adaln: bool = False,
    ) -> torch.Tensor:
        """Apply text cross-attention, with optional AdaLN modulation."""
        if cross_attention_adaln:
            shift_q, scale_q, gate = self.get_ada_values(scale_shift_table, x.shape[0], timestep, slice(6, 9))
            return apply_cross_attention_adaln(
                x,
                context,
                attn,
                shift_q,
                scale_q,
                gate,
                prompt_scale_shift_table,
                prompt_timestep,
                context_mask,
                self.norm_eps,
            )
        return attn(rms_norm(x, eps=self.norm_eps), context=context, mask=context_mask)

    def forward(  # noqa: PLR0915
        self,
        video: CausalTransformerArgs | TransformerArgs | None,
        audio: CausalTransformerArgs | TransformerArgs | None,
        perturbations: BatchedPerturbationConfig | None = None,
    ) -> tuple[CausalTransformerArgs | None, CausalTransformerArgs | None, CausalTransformerArgs | TransformerArgs | None]:
        if video is None and audio is None:
            raise ValueError("At least one of video or audio must be provided")

        batch_size = (video or audio).x.shape[0]

        if perturbations is None:
            perturbations = BatchedPerturbationConfig.empty(batch_size)

        vx = video.x if video is not None else None
        ax = audio.x if audio is not None else None

        run_vx = video is not None and video.enabled and vx.numel() > 0
        run_ax = audio is not None and audio.enabled and ax.numel() > 0

        run_a2v = run_vx and (audio is not None and ax.numel() > 0)
        run_v2a = run_ax and (video is not None and vx.numel() > 0)

        if run_vx:
            vshift_msa, vscale_msa, vgate_msa = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(0, 3)
            )
            norm_vx = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_msa) + vshift_msa

            all_perturbed = perturbations.all_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx)
            none_perturbed = not perturbations.any_in_batch(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx)
            v_mask = (
                perturbations.mask_like(PerturbationType.SKIP_VIDEO_SELF_ATTN, self.idx, vx)
                if not all_perturbed and not none_perturbed
                else None
            )

            if hasattr(video, "block_mask") and video.block_mask is not None:
                vx_attn = self.attn1(
                    norm_vx,
                    pe=video.positional_embeddings,
                    block_mask=video.block_mask,
                    logit_log_scale=video.self_attn_log_scale,
                )
            else:
                vx_attn = self.attn1(
                    norm_vx,
                    pe=video.positional_embeddings,
                    mask=video.self_attention_mask,
                    perturbation_mask=v_mask,
                    all_perturbed=all_perturbed,
                )

            if self._store_gate_stats:
                with torch.no_grad():
                    self._gate_stats['vgate_msa_mean'] = vgate_msa.detach().float().mean().item()
                    self._gate_stats['vgate_msa_std'] = vgate_msa.detach().float().std().item()
                    self._gate_stats['vscale_msa_mean'] = vscale_msa.detach().float().mean().item()
                    self._gate_stats['vscale_msa_std'] = vscale_msa.detach().float().std().item()
                    self._gate_stats['vshift_msa_mean'] = vshift_msa.detach().float().mean().item()
                    self._gate_stats['vx_attn_norm'] = vx_attn.detach().float().norm().item()
                    self._gate_stats['vx_self_attn_out_norm'] = vx_attn.detach().float().norm().item()
                    self._gate_stats['vx_self_attn_out_absmax'] = vx_attn.detach().float().abs().max().item()
                if vx_attn.requires_grad:
                    vx_attn.register_hook(lambda g, s=self: s._record_grad('vx_self_attn', g))

            vx = vx + vx_attn * vgate_msa

            del norm_vx, v_mask, vshift_msa, vscale_msa, vgate_msa

            vx_text_attn = self._apply_text_cross_attention(
                vx,
                video.context,
                self.attn2,
                self.scale_shift_table,
                getattr(self, "prompt_scale_shift_table", None),
                video.timesteps,
                video.prompt_timestep,
                video.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self._store_gate_stats:
                with torch.no_grad():
                    self._gate_stats['vx_text_attn_out_norm'] = vx_text_attn.detach().float().norm().item()
                    self._gate_stats['vx_text_attn_out_absmax'] = vx_text_attn.detach().float().abs().max().item()
                if vx_text_attn.requires_grad:
                    vx_text_attn.register_hook(lambda g, s=self: s._record_grad('vx_text_attn', g))
            
            vx = vx + vx_text_attn

        if run_ax:
            ashift_msa, ascale_msa, agate_msa = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(0, 3)
            )

            norm_ax = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_msa) + ashift_msa
            all_perturbed = perturbations.all_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx)
            none_perturbed = not perturbations.any_in_batch(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx)
            a_mask = (
                perturbations.mask_like(PerturbationType.SKIP_AUDIO_SELF_ATTN, self.idx, ax)
                if not all_perturbed and not none_perturbed
                else None
            )

            if hasattr(audio, "block_mask") and audio.block_mask is not None:
                ax_attn = self.audio_attn1(
                    norm_ax,
                    pe=audio.positional_embeddings,
                    block_mask=audio.block_mask,
                    logit_log_scale=audio.self_attn_log_scale,
                )
            else:
                ax_attn = self.audio_attn1(
                    norm_ax,
                    pe=audio.positional_embeddings,
                    mask=audio.self_attention_mask,
                    perturbation_mask=a_mask,
                    all_perturbed=all_perturbed,
                )

            # Collect audio gate stats
            if self._store_gate_stats:
                with torch.no_grad():
                    self._gate_stats['agate_msa_mean'] = agate_msa.detach().float().mean().item()
                    self._gate_stats['agate_msa_std'] = agate_msa.detach().float().std().item()
                    self._gate_stats['ax_attn_norm'] = ax_attn.detach().float().norm().item()
                    self._gate_stats['ax_self_attn_out_norm'] = ax_attn.detach().float().norm().item()
                    self._gate_stats['ax_self_attn_out_absmax'] = ax_attn.detach().float().abs().max().item()
                if ax_attn.requires_grad:
                    ax_attn.register_hook(lambda g, s=self: s._record_grad('ax_self_attn', g))
            ax = ax + ax_attn * agate_msa
            del agate_msa, norm_ax, a_mask, ashift_msa, ascale_msa

            ax_text_attn = self._apply_text_cross_attention(
                ax,
                audio.context,
                self.audio_attn2,
                self.audio_scale_shift_table,
                getattr(self, "audio_prompt_scale_shift_table", None),
                audio.timesteps,
                audio.prompt_timestep,
                audio.context_mask,
                cross_attention_adaln=self.cross_attention_adaln,
            )
            if self._store_gate_stats:
                with torch.no_grad():
                    self._gate_stats['ax_text_attn_out_norm'] = ax_text_attn.detach().float().norm().item()
                    self._gate_stats['ax_text_attn_out_absmax'] = ax_text_attn.detach().float().abs().max().item()
                if ax_text_attn.requires_grad:
                    ax_text_attn.register_hook(lambda g, s=self: s._record_grad('ax_text_attn', g))
            ax = ax + ax_text_attn

        # Audio - Video cross attention.
        if run_a2v or run_v2a:
            vx_norm3 = rms_norm(vx, eps=self.norm_eps)
            ax_norm3 = rms_norm(ax, eps=self.norm_eps)

            if run_a2v and not perturbations.all_in_batch(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx):
                scale_ca_video_a2v, shift_ca_video_a2v, gate_out_a2v = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(0, 2),
                )
                vx_scaled = vx_norm3 * (1 + scale_ca_video_a2v) + shift_ca_video_a2v
                del scale_ca_video_a2v, shift_ca_video_a2v

                scale_ca_audio_a2v, shift_ca_audio_a2v, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(0, 2),
                )
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_a2v) + shift_ca_audio_a2v
                del scale_ca_audio_a2v, shift_ca_audio_a2v
                a2v_mask = perturbations.mask_like(PerturbationType.SKIP_A2V_CROSS_ATTN, self.idx, vx)
                if hasattr(video, "cross_causal_mask") and video.cross_causal_mask is not None:
                    a2v_out = self.audio_to_video_attn(
                        vx_scaled,
                        context=ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                        cross_causal_mask=video.cross_causal_mask,  # A2V timestamp mask
                        logit_log_scale=video.cross_attn_log_scale,
                    )
                else:
                    a2v_out = self.audio_to_video_attn(
                        vx_scaled,
                        context=ax_scaled,
                        pe=video.cross_positional_embeddings,
                        k_pe=audio.cross_positional_embeddings,
                    )
                if self._store_gate_stats:
                    with torch.no_grad():
                        self._gate_stats['gate_a2v_mean'] = gate_out_a2v.detach().float().mean().item()
                        self._gate_stats['a2v_out_norm'] = a2v_out.detach().float().norm().item()
                        self._gate_stats['a2v_attn_out_norm'] = a2v_out.detach().float().norm().item()
                        self._gate_stats['a2v_attn_out_absmax'] = a2v_out.detach().float().abs().max().item()
                    if a2v_out.requires_grad:
                        a2v_out.register_hook(lambda g, s=self: s._record_grad('a2v_attn', g))

                vx = vx + a2v_out * gate_out_a2v * a2v_mask

                del gate_out_a2v, a2v_mask, vx_scaled, ax_scaled

            if run_v2a and not perturbations.all_in_batch(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx):
                scale_ca_audio_v2a, shift_ca_audio_v2a, gate_out_v2a = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_audio,
                    ax.shape[0],
                    audio.cross_scale_shift_timestep,
                    audio.cross_gate_timestep,
                    slice(2, 4),
                )
                ax_scaled = ax_norm3 * (1 + scale_ca_audio_v2a) + shift_ca_audio_v2a
                del scale_ca_audio_v2a, shift_ca_audio_v2a
                scale_ca_video_v2a, shift_ca_video_v2a, _ = self.get_av_ca_ada_values(
                    self.scale_shift_table_a2v_ca_video,
                    vx.shape[0],
                    video.cross_scale_shift_timestep,
                    video.cross_gate_timestep,
                    slice(2, 4),
                )
                vx_scaled = vx_norm3 * (1 + scale_ca_video_v2a) + shift_ca_video_v2a
                del scale_ca_video_v2a, shift_ca_video_v2a
                v2a_mask = perturbations.mask_like(PerturbationType.SKIP_V2A_CROSS_ATTN, self.idx, ax)

                if hasattr(audio, "cross_causal_mask") and audio.cross_causal_mask is not None:
                    v2a_out = self.video_to_audio_attn(
                        ax_scaled,
                        context=vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                        cross_causal_mask=audio.cross_causal_mask,  # V2A timestamp mask
                        logit_log_scale=audio.cross_attn_log_scale,
                    )
                else:
                    v2a_out = self.video_to_audio_attn(
                        ax_scaled,
                        context=vx_scaled,
                        pe=audio.cross_positional_embeddings,
                        k_pe=video.cross_positional_embeddings,
                    )
                if self._store_gate_stats:
                    with torch.no_grad():
                        self._gate_stats['gate_v2a_mean'] = gate_out_v2a.detach().float().mean().item()
                        self._gate_stats['v2a_out_norm'] = v2a_out.detach().float().norm().item()
                        self._gate_stats['v2a_attn_out_norm'] = v2a_out.detach().float().norm().item()
                        self._gate_stats['v2a_attn_out_absmax'] = v2a_out.detach().float().abs().max().item()
                    if v2a_out.requires_grad:
                        v2a_out.register_hook(lambda g, s=self: s._record_grad('v2a_attn', g))

                ax = ax + v2a_out * gate_out_v2a * v2a_mask

                del gate_out_v2a, v2a_mask, ax_scaled, vx_scaled

            del vx_norm3, ax_norm3

        if run_vx:
            vshift_mlp, vscale_mlp, vgate_mlp = self.get_ada_values(
                self.scale_shift_table, vx.shape[0], video.timesteps, slice(3, 6)
            )
            vx_scaled = rms_norm(vx, eps=self.norm_eps) * (1 + vscale_mlp) + vshift_mlp
            vx = vx + self.ff(vx_scaled) * vgate_mlp

            if self._store_gate_stats:
                with torch.no_grad():
                    self._gate_stats['vgate_mlp_mean'] = vgate_mlp.detach().float().mean().item()
                    self._gate_stats['vgate_mlp_std'] = vgate_mlp.detach().float().std().item()

            del vshift_mlp, vscale_mlp, vgate_mlp, vx_scaled

        if run_ax:
            ashift_mlp, ascale_mlp, agate_mlp = self.get_ada_values(
                self.audio_scale_shift_table, ax.shape[0], audio.timesteps, slice(3, 6)
            )
            ax_scaled = rms_norm(ax, eps=self.norm_eps) * (1 + ascale_mlp) + ashift_mlp
            ax = ax + self.audio_ff(ax_scaled) * agate_mlp

            if self._store_gate_stats:
                with torch.no_grad():
                    self._gate_stats['agate_mlp_mean'] = agate_mlp.detach().float().mean().item()
                    self._gate_stats['agate_mlp_std'] = agate_mlp.detach().float().std().item()

            del ashift_mlp, ascale_mlp, agate_mlp, ax_scaled

        return replace(video, x=vx) if video is not None else None, replace(audio, x=ax) if audio is not None else None


class TransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        use_middle_indices_grid: bool,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: LTXRopeType | CausalRopeType,
        caption_projection: torch.nn.Module | None = None,
        prompt_adaln: AdaLayerNormSingle | None = None,
    ) -> None:
        self.patchify_proj = patchify_proj
        self.adaln = adaln
        self.inner_dim = inner_dim
        self.max_pos = max_pos
        self.num_attention_heads = num_attention_heads
        self.use_middle_indices_grid = use_middle_indices_grid
        self.timestep_scale_multiplier = timestep_scale_multiplier
        self.double_precision_rope = double_precision_rope
        self.positional_embedding_theta = positional_embedding_theta
        self.rope_type = rope_type
        self.caption_projection = caption_projection
        self.prompt_adaln = prompt_adaln

    def _prepare_timestep(
        self, timestep: torch.Tensor, adaln: AdaLayerNormSingle, batch_size: int, hidden_dtype: torch.dtype
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare timestep embeddings."""
        timestep_scaled = timestep * self.timestep_scale_multiplier
        timestep, embedded_timestep = adaln(
            timestep_scaled.flatten(),
            hidden_dtype=hidden_dtype,
        )
        # Second dimension is 1 or number of tokens (if timestep_per_token)
        timestep = timestep.view(batch_size, -1, timestep.shape[-1])
        embedded_timestep = embedded_timestep.view(batch_size, -1, embedded_timestep.shape[-1])

        return timestep, embedded_timestep

    def _prepare_context(
        self,
        context: torch.Tensor,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Prepare context for transformer blocks."""
        if self.caption_projection is not None:
            context = self.caption_projection(context)
        batch_size = x.shape[0]
        return context.view(batch_size, -1, x.shape[-1])

    def _prepare_attention_mask(self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype) -> torch.Tensor | None:
        """Prepare attention mask."""
        if attention_mask is None or torch.is_floating_point(attention_mask):
            return attention_mask

        return (attention_mask - 1).to(x_dtype).reshape(
            (attention_mask.shape[0], 1, -1, attention_mask.shape[-1])
        ) * torch.finfo(x_dtype).max

    def _prepare_self_attention_mask(
        self, attention_mask: torch.Tensor | None, x_dtype: torch.dtype
    ) -> torch.Tensor | None:
        """Prepare self-attention mask by converting [0,1] values to additive log-space bias.
        Input shape: (B, T, T) with values in [0, 1].
        Output shape: (B, 1, T, T) with 0.0 for full attention and a large negative value
        for masked positions.
        Positions with attention_mask <= 0 are fully masked (mapped to the dtype's minimum
        representable value). Strictly positive entries are converted via log-space for
        smooth attenuation, with small values clamped for numerical stability.
        Returns None if input is None (no masking).
        """
        if attention_mask is None:
            return None

        # Convert [0, 1] attention mask to additive log-space bias:
        #   1.0 -> log(1.0) = 0.0  (no bias, full attention)
        #   0.0 -> finfo.min        (fully masked)
        finfo = torch.finfo(x_dtype)
        eps = finfo.tiny

        bias = torch.full_like(attention_mask, finfo.min, dtype=x_dtype)
        positive = attention_mask > 0
        if positive.any():
            bias[positive] = torch.log(attention_mask[positive].clamp(min=eps)).to(x_dtype)

        return bias.unsqueeze(1)  # (B, 1, T, T) for head broadcast

    def _prepare_positional_embeddings(
        self,
        positions: torch.Tensor,
        inner_dim: int,
        max_pos: list[int],
        use_middle_indices_grid: bool,
        num_attention_heads: int,
        x_dtype: torch.dtype,
    ) -> torch.Tensor:
        """Prepare positional embeddings."""
        freq_grid_generator = generate_freq_grid_np if self.double_precision_rope else generate_freq_grid_pytorch
        pe = precompute_freqs_cis(
            positions,
            dim=inner_dim,
            out_dtype=x_dtype,
            theta=self.positional_embedding_theta,
            max_pos=max_pos,
            use_middle_indices_grid=use_middle_indices_grid,
            num_attention_heads=num_attention_heads,
            rope_type=self.rope_type,
            freq_grid_generator=freq_grid_generator,
        )
        return pe

    def prepare(
        self,
        modality: Modality,
        cross_modality: Modality | None = None,  # noqa: ARG002
    ) -> CausalTransformerArgs:
        x = self.patchify_proj(modality.latent)
        batch_size = x.shape[0]
        timestep, embedded_timestep = self._prepare_timestep(
            modality.timesteps, self.adaln, batch_size, modality.latent.dtype
        )
        prompt_timestep = None
        if self.prompt_adaln is not None:
            prompt_timestep, _ = self._prepare_timestep(
                modality.sigma, self.prompt_adaln, batch_size, modality.latent.dtype
            )
        context = self._prepare_context(modality.context, x)
        attention_mask = self._prepare_attention_mask(modality.context_mask, modality.latent.dtype)
        pe = self._prepare_positional_embeddings(
            positions=modality.positions,
            inner_dim=self.inner_dim,
            max_pos=self.max_pos,
            use_middle_indices_grid=self.use_middle_indices_grid,
            num_attention_heads=self.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )
        self_attention_mask = self._prepare_self_attention_mask(modality.attention_mask, modality.latent.dtype)
        return CausalTransformerArgs(
            x=x,
            context=context,
            context_mask=attention_mask,
            timesteps=timestep,
            embedded_timestep=embedded_timestep,
            positional_embeddings=pe,
            cross_positional_embeddings=None,
            cross_scale_shift_timestep=None,
            cross_gate_timestep=None,
            enabled=modality.enabled,
            prompt_timestep=prompt_timestep,
            self_attention_mask=self_attention_mask,
            original_shape=modality.original_shape,
        )


class MultiModalTransformerArgsPreprocessor:
    def __init__(  # noqa: PLR0913
        self,
        patchify_proj: torch.nn.Linear,
        adaln: AdaLayerNormSingle,
        cross_scale_shift_adaln: AdaLayerNormSingle,
        cross_gate_adaln: AdaLayerNormSingle,
        inner_dim: int,
        max_pos: list[int],
        num_attention_heads: int,
        cross_pe_max_pos: int,
        use_middle_indices_grid: bool,
        audio_cross_attention_dim: int,
        timestep_scale_multiplier: int,
        double_precision_rope: bool,
        positional_embedding_theta: float,
        rope_type: CausalRopeType | LTXRopeType,
        av_ca_timestep_scale_multiplier: int,
        caption_projection: torch.nn.Module | None = None,
        prompt_adaln: AdaLayerNormSingle | None = None,
    ) -> None:
        self.simple_preprocessor = TransformerArgsPreprocessor(
            patchify_proj=patchify_proj,
            adaln=adaln,
            inner_dim=inner_dim,
            max_pos=max_pos,
            num_attention_heads=num_attention_heads,
            use_middle_indices_grid=use_middle_indices_grid,
            timestep_scale_multiplier=timestep_scale_multiplier,
            double_precision_rope=double_precision_rope,
            positional_embedding_theta=positional_embedding_theta,
            rope_type=rope_type,
            caption_projection=caption_projection,
            prompt_adaln=prompt_adaln,
        )
        self.cross_scale_shift_adaln = cross_scale_shift_adaln
        self.cross_gate_adaln = cross_gate_adaln
        self.cross_pe_max_pos = cross_pe_max_pos
        self.audio_cross_attention_dim = audio_cross_attention_dim
        self.av_ca_timestep_scale_multiplier = av_ca_timestep_scale_multiplier

    def prepare(
        self,
        modality: Modality,
        cross_modality: Modality | None = None,
    ) -> CausalTransformerArgs:
        transformer_args = self.simple_preprocessor.prepare(modality)
        if cross_modality is None:
            return transformer_args

        if cross_modality.sigma.numel() > 1:
            if cross_modality.sigma.shape[0] != modality.timesteps.shape[0]:
                raise ValueError("Cross modality sigma must have the same batch size as the modality")
            if cross_modality.sigma.ndim != 1:
                raise ValueError("Cross modality sigma must be a 1D tensor")

        cross_timestep = cross_modality.sigma.view(
            modality.timesteps.shape[0], 1, *[1] * len(modality.timesteps.shape[2:])
        )

        cross_pe = self.simple_preprocessor._prepare_positional_embeddings(
            positions=modality.positions[:, 0:1, :],
            inner_dim=self.audio_cross_attention_dim,
            max_pos=[self.cross_pe_max_pos],
            use_middle_indices_grid=True,
            num_attention_heads=self.simple_preprocessor.num_attention_heads,
            x_dtype=modality.latent.dtype,
        )

        cross_scale_shift_timestep, cross_gate_timestep = self._prepare_cross_attention_timestep(
            timestep=cross_timestep,
            timestep_scale_multiplier=self.simple_preprocessor.timestep_scale_multiplier,
            batch_size=transformer_args.x.shape[0],
            hidden_dtype=modality.latent.dtype,
        )

        return replace(
            transformer_args,
            cross_positional_embeddings=cross_pe,
            cross_scale_shift_timestep=cross_scale_shift_timestep,
            cross_gate_timestep=cross_gate_timestep,
        )

    def _prepare_cross_attention_timestep(
        self,
        timestep: torch.Tensor | None,
        timestep_scale_multiplier: int,
        batch_size: int,
        hidden_dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Prepare cross attention timestep embeddings."""
        timestep = timestep * timestep_scale_multiplier

        av_ca_factor = self.av_ca_timestep_scale_multiplier / timestep_scale_multiplier

        scale_shift_timestep, _ = self.cross_scale_shift_adaln(
            timestep.flatten(),
            hidden_dtype=hidden_dtype,
        )
        scale_shift_timestep = scale_shift_timestep.view(batch_size, -1, scale_shift_timestep.shape[-1])
        gate_noise_timestep, _ = self.cross_gate_adaln(
            timestep.flatten() * av_ca_factor,
            hidden_dtype=hidden_dtype,
        )
        gate_noise_timestep = gate_noise_timestep.view(batch_size, -1, gate_noise_timestep.shape[-1])

        return scale_shift_timestep, gate_noise_timestep



def apply_cross_attention_adaln(
    x: torch.Tensor,
    context: torch.Tensor,
    attn: AttentionCallable,
    q_shift: torch.Tensor,
    q_scale: torch.Tensor,
    q_gate: torch.Tensor,
    prompt_scale_shift_table: torch.Tensor,
    prompt_timestep: torch.Tensor,
    context_mask: torch.Tensor | None = None,
    norm_eps: float = 1e-6,
) -> torch.Tensor:
    batch_size = x.shape[0]
    shift_kv, scale_kv = (
        prompt_scale_shift_table[None, None].to(device=x.device, dtype=x.dtype)
        + prompt_timestep.reshape(batch_size, prompt_timestep.shape[1], 2, -1)
    ).unbind(dim=2)
    attn_input = rms_norm(x, eps=norm_eps) * (1 + q_scale) + q_shift
    encoder_hidden_states = context * (1 + scale_kv) + shift_kv
    return attn(attn_input, context=encoder_hidden_states, mask=context_mask) * q_gate
