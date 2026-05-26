"""
CausalLTXAttention: Causal attention module with Flexattention for training.

This module implements:
- Training mode: Flexattention with BlockMask for efficient block-wise causal attention
- Weight-compatible with original LTX-2 Attention module

Key Design Decisions:
1. Same projection layer structure as original Attention (to_q, to_k, to_v, to_out)
2. Same normalization (q_norm, k_norm with RMSNorm)
3. BlockMask for causal self-attention, dense mask for cross-attention
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn

from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb

from ltx_causal.attention.flex_attention_utils import (
    FLEX_ATTENTION_AVAILABLE,
    flex_attention_forward,
    standard_attention_forward,
)
from ltx_causal.rope.causal_rope import (
    CausalRopeType,
    apply_interleaved_rotary_emb,
)

# Import BlockMask type for annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from torch.nn.attention.flex_attention import BlockMask

# ltx2.3
from enum import Enum
from typing import Protocol

memory_efficient_attention = None
flash_attn_interface = None
try:
    from xformers.ops import memory_efficient_attention
except ImportError:
    memory_efficient_attention = None
try:
    # FlashAttention3 and XFormersAttention cannot be used together
    if memory_efficient_attention is None:
        import flash_attn_interface
except ImportError:
    flash_attn_interface = None

from ltx_core.model.transformer.attention import AttentionFunction, AttentionCallable

# class AttentionCallable(Protocol):
#     def __call__(
#         self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
#     ) -> torch.Tensor: ...


# class PytorchAttention(AttentionCallable):
#     def __call__(
#         self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
#     ) -> torch.Tensor:
#         b, _, dim_head = q.shape
#         dim_head //= heads
#         q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

#         if mask is not None:
#             # add a batch dimension if there isn't already one
#             if mask.ndim == 2:
#                 mask = mask.unsqueeze(0)
#             # add a heads dimension if there isn't already one
#             if mask.ndim == 3:
#                 mask = mask.unsqueeze(1)

#         out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
#         out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
#         return out


# class XFormersAttention(AttentionCallable):
#     def __call__(
#         self,
#         q: torch.Tensor,
#         k: torch.Tensor,
#         v: torch.Tensor,
#         heads: int,
#         mask: torch.Tensor | None = None,
#     ) -> torch.Tensor:
#         if memory_efficient_attention is None:
#             raise RuntimeError("XFormersAttention was selected but `xformers` is not installed.")

#         b, _, dim_head = q.shape
#         dim_head //= heads

#         # xformers expects [B, M, H, K]
#         q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

#         if mask is not None:
#             # add a singleton batch dimension
#             if mask.ndim == 2:
#                 mask = mask.unsqueeze(0)
#             # add a singleton heads dimension
#             if mask.ndim == 3:
#                 mask = mask.unsqueeze(1)
#             # pad to a multiple of 8
#             pad = 8 - mask.shape[-1] % 8
#             # the xformers docs says that it's allowed to have a mask of shape (1, Nq, Nk)
#             # but when using separated heads, the shape has to be (B, H, Nq, Nk)
#             # in flux, this matrix ends up being over 1GB
#             # here, we create a mask with the same batch/head size as the input mask (potentially singleton or full)
#             mask_out = torch.empty(
#                 [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device
#             )

#             mask_out[..., : mask.shape[-1]] = mask
#             # doesn't this remove the padding again??
#             mask = mask_out[..., : mask.shape[-1]]
#             mask = mask.expand(b, heads, -1, -1)

#         out = memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, attn_bias=mask, p=0.0)
#         out = out.reshape(b, -1, heads * dim_head)
#         return out


# class FlashAttention3(AttentionCallable):
#     def __call__(
#         self,
#         q: torch.Tensor,
#         k: torch.Tensor,
#         v: torch.Tensor,
#         heads: int,
#         mask: torch.Tensor | None = None,
#     ) -> torch.Tensor:
#         if flash_attn_interface is None:
#             raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")

#         b, _, dim_head = q.shape
#         dim_head //= heads

#         q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

#         if mask is not None:
#             raise NotImplementedError("Mask is not supported for FlashAttention3")

#         out = flash_attn_interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
#         out = out.reshape(b, -1, heads * dim_head)
#         return out


# class AttentionFunction(Enum):
#     PYTORCH = "pytorch"
#     XFORMERS = "xformers"
#     FLASH_ATTENTION_3 = "flash_attention_3"
#     DEFAULT = "default"

#     def to_callable(self) -> AttentionCallable:
#         """Resolve to a concrete callable. Use this at module init time so that
#         torch.compile can trace through the attention call without graph breaks."""
#         if self is AttentionFunction.PYTORCH:
#             return PytorchAttention()
#         elif self is AttentionFunction.XFORMERS:
#             return XFormersAttention()
#         elif self is AttentionFunction.FLASH_ATTENTION_3:
#             return FlashAttention3()
#         else:
#             # Default behavior: XFormers if installed else - PyTorch
#             return XFormersAttention() if memory_efficient_attention is not None else PytorchAttention()


class CausalLTXAttention(nn.Module):
    """
    Causal attention module for LTX-2.

    This module is weight-compatible with the original LTX-2 Attention:
    - Same linear projections (to_q, to_k, to_v, to_out)
    - Same RMSNorm for Q/K normalization
    - Supports both self-attention and cross-attention

    Causal Features:
    - Uses Flexattention with BlockMask for efficient causal attention
    - Dense mask for cross-modal causal attention (A2V, V2A)

    Args:
        query_dim: Dimension of query input
        context_dim: Dimension of context input (None for self-attention)
        heads: Number of attention heads
        dim_head: Dimension per head
        norm_eps: Epsilon for RMSNorm
        rope_type: Type of RoPE (INTERLEAVED only)
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: Optional[int] = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType | CausalRopeType = CausalRopeType.INTERLEAVED,
        attention_function: AttentionCallable | AttentionFunction = AttentionFunction.DEFAULT,
        apply_gated_attention: bool = False,
        # Kept in signature for backward-compatible construction but unused
        local_attn_size: int = -1,
        sink_size: int = 1,
    ) -> None:
        super().__init__()

        self.rope_type = rope_type
        self.attention_function = (
            attention_function.to_callable()
            if isinstance(attention_function, AttentionFunction)
            else attention_function
        )
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head
        self.is_cross_attention = context_dim is not None
        context_dim = query_dim if context_dim is None else context_dim

        # === Projection Layers (Weight-Compatible with Original) ===
        self.to_q = nn.Linear(query_dim, self.inner_dim, bias=True)
        self.to_k = nn.Linear(context_dim, self.inner_dim, bias=True)
        self.to_v = nn.Linear(context_dim, self.inner_dim, bias=True)

        # Q/K Normalization
        self.q_norm = nn.RMSNorm(self.inner_dim, eps=norm_eps)
        self.k_norm = nn.RMSNorm(self.inner_dim, eps=norm_eps)

        # Optional per-head gating
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, query_dim, bias=True),
            nn.Identity(),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        pe: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        k_pe: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
        # === Causal Training Parameters ===
        block_mask: Optional["BlockMask"] = None,
        cross_causal_mask: Optional[torch.Tensor] = None,
        logit_log_scale: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass for training with causal masks.

        Args:
            x: Query input [B, L, D]
            context: Context for cross-attention [B, L_ctx, D_ctx] (None for self-attn)
            mask: Optional attention mask (for non-causal attention, e.g. text)
            pe: RoPE frequencies for Q (cos, sin)
            k_pe: RoPE frequencies for K (if different from Q)
            block_mask: BlockMask for flexattention (causal self-attention)
            cross_causal_mask: Dense mask for cross-attention causality (A2V/V2A)
            logit_log_scale: Per-position log-ratio scale [1, L_q, 1] applied to Q
                before attention, making QK^T = (Q * scale) K^T. Acts as a
                position-dependent temperature: tokens seeing fewer KV tokens
                get scale < 1, softening their attention distribution.

        Returns:
            Attention output [B, L, D]
        """
        B, L, _ = x.shape
        context = x if context is None else context

        use_attention = not all_perturbed
        v = self.to_v(context)
        if not use_attention:
            out = v
            # Apply per-head gating if enabled
            if self.to_gate_logits is not None:
                gate_logits = self.to_gate_logits(x)  # (B, T, H)
                b, t, _ = out.shape
                # Reshape to (B, T, H, D) for per-head gating
                out = out.view(b, t, self.heads, self.dim_head)
                # Apply gating: 2 * sigmoid(x) so that zero-init gives identity (2 * 0.5 = 1.0)
                gates = 2.0 * torch.sigmoid(gate_logits)  # (B, T, H)
                out = out * gates.unsqueeze(-1)  # (B, T, H, D) * (B, T, H, 1)
                # Reshape back to (B, T, H*D)
                out = out.view(b, t, self.heads * self.dim_head)
        else:
            # Projections
            q = self.to_q(x)
            k = self.to_k(context)

            # Q/K Normalization
            q = self.q_norm(q)
            k = self.k_norm(k)

            # Apply RoPE if provided
            # [NOTE]: causal attention's rope is INTERLEAVED
            if pe is not None:
                q = self._apply_rope(q, pe)
                k = self._apply_rope(k, pe if k_pe is None else k_pe)
            # if pe is not None:
            #     q = apply_rotary_emb(q, pe, self.rope_type)
            #     k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

            # Apply log-ratio scaling to Q (PaLM-style Log-N Scaling)
            # This scales QK^T by a position-dependent factor, acting as a
            # per-token temperature that softens attention for early causal blocks.
            # scale = log(1 + visible) / log(1 + total), applied BEFORE reshape
            # so it broadcasts across all heads: [1, L, 1] * [B, L, inner_dim]
            # """
            if logit_log_scale is not None:
                q = q * logit_log_scale

            # Reshape for attention: [B, L, H, D]
            q = q.view(B, -1, self.heads, self.dim_head)
            k = k.view(B, -1, self.heads, self.dim_head)
            v = v.view(B, -1, self.heads, self.dim_head)

            # Apply attention
            if block_mask is not None:
                if not FLEX_ATTENTION_AVAILABLE:
                    raise RuntimeError(
                        "block_mask provided but flex_attention is not available. "
                        "PyTorch 2.2+ with CUDA is required for causal self-attention."
                    )
                # === Flexattention Path (Self-Attention with BlockMask) ===
                out = flex_attention_forward(q, k, v, block_mask)

            elif cross_causal_mask is not None:
                # === Standard Attention with Dense Causal Mask (Cross-Attention) ===
                # [TODO]: change to the latest attention implementation 
                out = standard_attention_forward(q, k, v, cross_causal_mask)

            elif mask is not None:
                # === Standard Attention with Provided Mask (no temperature) ===
                out = standard_attention_forward(q, k, v, mask)

            else:
                # === Standard Attention (No Mask, no temperature) ===
                out = standard_attention_forward(q, k, v)
        
            # out: [B, L, H, D]
            # Apply per-head gating if enabled
            if self.to_gate_logits is not None:
                gate_logits = self.to_gate_logits(x)  # (B, T, H)
                # Apply gating: 2 * sigmoid(x) so that zero-init gives identity (2 * 0.5 = 1.0)
                gates = 2.0 * torch.sigmoid(gate_logits)  # (B, T, H)
                out = out * gates.unsqueeze(-1)  # (B, T, H, D) * (B, T, H, 1)
                # Reshape back to (B, T, H*D)
                out = out.view(B, -1, self.inner_dim)
            else:
                # Reshape and project output
                out = out.reshape(B, -1, self.inner_dim)
        # """
        return self.to_out(out)

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     context: torch.Tensor | None = None,
    #     mask: torch.Tensor | None = None,
    #     pe: torch.Tensor | None = None,
    #     k_pe: torch.Tensor | None = None,
    #     perturbation_mask: torch.Tensor | None = None,
    #     all_perturbed: bool = False,
    # ) -> torch.Tensor:
    #     """Multi-head attention with optional RoPE, perturbation masking, and per-head gating.
    #     When ``perturbation_mask`` is all zeros, the expensive query/key path
    #     (linear projections, RMSNorm, RoPE) is skipped entirely and only the
    #     value projection is used as a pass-through.
    #     Args:
    #         x: Query input tensor of shape ``(B, T, query_dim)``.
    #         context: Key/value context tensor of shape ``(B, S, context_dim)``.
    #             Falls back to ``x`` (self-attention) when *None*.
    #         mask: Optional attention mask. Interpretation depends on the attention
    #             backend (additive bias for xformers/PyTorch SDPA).
    #         pe: Rotary positional embeddings applied to both ``q`` and ``k``.
    #         k_pe: Separate rotary positional embeddings for ``k`` only. When
    #             *None*, ``pe`` is reused for keys.
    #         perturbation_mask: Optional mask in ``[0, 1]`` that
    #             blends the attention output with the raw value projection:
    #             ``out = attn_out * mask + v * (1 - mask)``.
    #             **1** keeps the full attention output, **0** bypasses attention
    #             and passes the value projection through unchanged.
    #             *None* or all-ones means standard attention; all-zeros skips
    #             the query/key path entirely for efficiency.
    #         all_perturbed: Whether all perturbations are active for this block.
    #     Returns:
    #         Output tensor of shape ``(B, T, query_dim)``.
    #     """
    #     context = x if context is None else context
    #     use_attention = not all_perturbed

    #     v = self.to_v(context)

    #     if not use_attention:
    #         out = v
    #     else:
    #         q = self.to_q(x)
    #         k = self.to_k(context)

    #         q = self.q_norm(q)
    #         k = self.k_norm(k)

    #         if pe is not None:
    #             q = apply_rotary_emb(q, pe, self.rope_type)
    #             k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)

    #         out = self.attention_function(q, k, v, self.heads, mask)  # (B, T, H*D)

    #         if perturbation_mask is not None:
    #             out = out * perturbation_mask + v * (1 - perturbation_mask)

    #     # Apply per-head gating if enabled
    #     if self.to_gate_logits is not None:
    #         gate_logits = self.to_gate_logits(x)  # (B, T, H)
    #         b, t, _ = out.shape
    #         # Reshape to (B, T, H, D) for per-head gating
    #         out = out.view(b, t, self.heads, self.dim_head)
    #         # Apply gating: 2 * sigmoid(x) so that zero-init gives identity (2 * 0.5 = 1.0)
    #         gates = 2.0 * torch.sigmoid(gate_logits)  # (B, T, H)
    #         out = out * gates.unsqueeze(-1)  # (B, T, H, D) * (B, T, H, 1)
    #         # Reshape back to (B, T, H*D)
    #         out = out.view(b, t, self.heads * self.dim_head)

    #     return self.to_out(out)

    def _apply_rope(
        self,
        x: torch.Tensor,
        freqs_cis: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        """Apply RoPE to input tensor. Only INTERLEAVED mode is supported."""
        if self.rope_type != CausalRopeType.INTERLEAVED:
            raise ValueError(
                f"Only CausalRopeType.INTERLEAVED is supported, got {self.rope_type}. "
                f"SPLIT mode is not implemented correctly for causal generation."
            )
        cos_freqs, sin_freqs = freqs_cis
        return apply_interleaved_rotary_emb(x, cos_freqs, sin_freqs)


# ============================================================================
# Factory Functions
# ============================================================================

def create_causal_attention(
    query_dim: int,
    context_dim: Optional[int] = None,
    heads: int = 32,
    dim_head: int = 128,
    **kwargs,
) -> CausalLTXAttention:
    """
    Factory function to create CausalLTXAttention with LTX-2 defaults.

    Args:
        query_dim: Query dimension
        context_dim: Context dimension (None for self-attention)
        heads: Number of attention heads
        dim_head: Dimension per head
    Returns:
        Configured CausalLTXAttention instance
    """
    return CausalLTXAttention(
        query_dim=query_dim,
        context_dim=context_dim,
        heads=heads,
        dim_head=dim_head,
        **kwargs,
    )


def create_video_self_attention(
    dim: int = 4096,
    heads: int = 32,
    dim_head: int = 128,
    **kwargs,
) -> CausalLTXAttention:
    """Create video self-attention module with LTX-2 19B dimensions."""
    return create_causal_attention(
        query_dim=dim,
        context_dim=None,
        heads=heads,
        dim_head=dim_head,
        **kwargs,
    )


def create_audio_self_attention(
    dim: int = 2048,
    heads: int = 32,
    dim_head: int = 64,
    **kwargs,
) -> CausalLTXAttention:
    """Create audio self-attention module with LTX-2 19B dimensions."""
    return create_causal_attention(
        query_dim=dim,
        context_dim=None,
        heads=heads,
        dim_head=dim_head,
        **kwargs,
    )


def create_cross_attention(
    query_dim: int,
    context_dim: int,
    heads: int = 32,
    dim_head: int = 64,
    **kwargs,
) -> CausalLTXAttention:
    """Create cross-attention module (A2V or V2A)."""
    return create_causal_attention(
        query_dim=query_dim,
        context_dim=context_dim,
        heads=heads,
        dim_head=dim_head,
        **kwargs,
    )
