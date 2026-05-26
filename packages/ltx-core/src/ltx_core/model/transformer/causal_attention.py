from enum import Enum
from typing import Protocol

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask
from ltx_core.model.transformer.rope import LTXRopeType, apply_rotary_emb
from torch.nn.attention.flex_attention import BlockMask

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

from ltx_core.model.transformer.modality import KVCache

class AttentionCallable(Protocol):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor: ...


class PytorchAttention(AttentionCallable):
    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        b, _, dim_head = q.shape
        dim_head //= heads
        q, k, v = (t.view(b, -1, heads, dim_head).transpose(1, 2) for t in (q, k, v))

        if mask is not None:
            # add a batch dimension if there isn't already one
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a heads dimension if there isn't already one
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(b, -1, heads * dim_head)
        return out


class XFormersAttention(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_efficient_attention is None:
            raise RuntimeError("XFormersAttention was selected but `xformers` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        # xformers expects [B, M, H, K]
        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            # add a singleton batch dimension
            if mask.ndim == 2:
                mask = mask.unsqueeze(0)
            # add a singleton heads dimension
            if mask.ndim == 3:
                mask = mask.unsqueeze(1)
            # pad to a multiple of 8
            pad = 8 - mask.shape[-1] % 8
            # the xformers docs says that it's allowed to have a mask of shape (1, Nq, Nk)
            # but when using separated heads, the shape has to be (B, H, Nq, Nk)
            # in flux, this matrix ends up being over 1GB
            # here, we create a mask with the same batch/head size as the input mask (potentially singleton or full)
            mask_out = torch.empty(
                [mask.shape[0], mask.shape[1], q.shape[1], mask.shape[-1] + pad], dtype=q.dtype, device=q.device
            )

            mask_out[..., : mask.shape[-1]] = mask
            # doesn't this remove the padding again??
            mask = mask_out[..., : mask.shape[-1]]
            mask = mask.expand(b, heads, -1, -1)

        out = memory_efficient_attention(q.to(v.dtype), k.to(v.dtype), v, attn_bias=mask, p=0.0)
        out = out.reshape(b, -1, heads * dim_head)
        return out


class FlashAttention3(AttentionCallable):
    def __call__(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        heads: int,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if flash_attn_interface is None:
            raise RuntimeError("FlashAttention3 was selected but `FlashAttention3` is not installed.")

        b, _, dim_head = q.shape
        dim_head //= heads

        q, k, v = (t.view(b, -1, heads, dim_head) for t in (q, k, v))

        if mask is not None:
            raise NotImplementedError("Mask is not supported for FlashAttention3")

        out = flash_attn_interface.flash_attn_func(q.to(v.dtype), k.to(v.dtype), v)
        out = out.reshape(b, -1, heads * dim_head)
        return out

    
class FlexAttention(AttentionCallable):
    def __call__(self, q, k, v, heads: int, mask: BlockMask | None = None):
        b, _, dim = q.shape
        dim_head = dim // heads

        # total len

        padded_length_q = mask.shape[-2] - q.shape[1]
        padded_length_k = mask.shape[-1] - k.shape[1]
        padded_length_v = mask.shape[-1] - v.shape[1]
        padded_q = torch.cat(
            [q, torch.zeros([q.shape[0], padded_length_q, q.shape[2]],
            device=q.device, dtype=v.dtype)],
            dim=1
        )

        padded_k = torch.cat(
            [k, torch.zeros([k.shape[0], padded_length_k, k.shape[2]],
                                    device=k.device, dtype=v.dtype)],
            dim=1
        )

        padded_v = torch.cat(
            [v, torch.zeros([v.shape[0], padded_length_v, v.shape[2]],
                            device=v.device, dtype=v.dtype)],
            dim=1
        )

        padded_q = padded_q.view(b, -1, heads, dim_head).permute(0, 2, 1, 3)
        padded_k = padded_k.view(b, -1, heads, dim_head).permute(0, 2, 1, 3)
        padded_v = padded_v.view(b, -1, heads, dim_head).permute(0, 2, 1, 3)

        if padded_length_q > 0: 
            out = flex_attention(padded_q, padded_k, padded_v, block_mask=mask)[:, :, :-padded_length_q]
        else:
            out = flex_attention(padded_q, padded_k, padded_v, block_mask=mask)

        out = out.permute(0, 2, 1, 3).reshape(b, -1, dim)

        return out


class AttentionFunction(Enum):
    PYTORCH = "pytorch"
    XFORMERS = "xformers"
    FLASH_ATTENTION_3 = "flash_attention_3"
    FLEX = "flex"
    DEFAULT = "default"

    def to_callable(self) -> AttentionCallable:
        """Resolve to a concrete callable. Use this at module init time so that
        torch.compile can trace through the attention call without graph breaks."""
        if self is AttentionFunction.PYTORCH:
            return PytorchAttention()
        elif self is AttentionFunction.XFORMERS:
            return XFormersAttention()
        elif self is AttentionFunction.FLASH_ATTENTION_3:
            return FlashAttention3()
        else:
            # Default behavior: XFormers if installed else - PyTorch
            return XFormersAttention() if memory_efficient_attention is not None else PytorchAttention()

    def __call__(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, heads: int, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if self is AttentionFunction.PYTORCH:
            return PytorchAttention()(q, k, v, heads, mask)
        elif self is AttentionFunction.XFORMERS:
            return XFormersAttention()(q, k, v, heads, mask)
        elif self is AttentionFunction.FLASH_ATTENTION_3:
            return FlashAttention3()(q, k, v, heads, mask)
        elif self is AttentionFunction.FLEX:
            if mask is None:
                raise ValueError("FlexAttention requires a blockmask")
            return FlexAttention()(q, k, v, heads, mask)
        else:
            # Default behavior: XFormers if installed else - PyTorch
            return (
                XFormersAttention()(q, k, v, heads, mask)
                if memory_efficient_attention is not None
                else PytorchAttention()(q, k, v, heads, mask)
            )


class CausalAttention(torch.nn.Module):
    def __init__(
        self,
        query_dim: int,
        context_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        norm_eps: float = 1e-6,
        rope_type: LTXRopeType = LTXRopeType.INTERLEAVED,
        # always use flex attention when using causal attention
        attention_function: AttentionCallable | AttentionFunction | None = None,
        apply_gated_attention: bool = False,
    ) -> None:
        super().__init__()
        self.rope_type = rope_type
        if attention_function is None:
            attention_function = AttentionFunction.FLEX
        else:
            self.attention_function = (
                attention_function.to_callable()
                if isinstance(attention_function, AttentionFunction)
                else attention_function
            )

        inner_dim = dim_head * heads
        context_dim = query_dim if context_dim is None else context_dim

        self.heads = heads
        self.dim_head = dim_head

        self.q_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)
        self.k_norm = torch.nn.RMSNorm(inner_dim, eps=norm_eps)

        self.to_q = torch.nn.Linear(query_dim, inner_dim, bias=True)
        self.to_k = torch.nn.Linear(context_dim, inner_dim, bias=True)
        self.to_v = torch.nn.Linear(context_dim, inner_dim, bias=True)

        # Optional per-head gating
        if apply_gated_attention:
            self.to_gate_logits = torch.nn.Linear(query_dim, heads, bias=True)
        else:
            self.to_gate_logits = None

        self.to_out = torch.nn.Sequential(torch.nn.Linear(inner_dim, query_dim, bias=True), torch.nn.Identity())

    # @profile
    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        pe: torch.Tensor | None = None,
        k_pe: torch.Tensor | None = None,
        perturbation_mask: torch.Tensor | None = None,
        all_perturbed: bool = False,
        kv_cache=None,
        kv_cache_name: str | None = None,
        kv_cache_snapshot: dict | None = None,
        dtype=torch.bfloat16,
        use_text_encoder_cache: bool = False,
    ) -> torch.Tensor:
        """Multi-head attention with optional RoPE, perturbation masking, and per-head gating.
        When ``perturbation_mask`` is all zeros, the expensive query/key path
        (linear projections, RMSNorm, RoPE) is skipped entirely and only the
        value projection is used as a pass-through.
        Args:
            x: Query input tensor of shape ``(B, T, query_dim)``.
            context: Key/value context tensor of shape ``(B, S, context_dim)``.
                Falls back to ``x`` (self-attention) when *None*.
            mask: Optional attention mask. Interpretation depends on the attention
                backend (additive bias for xformers/PyTorch SDPA).
            pe: Rotary positional embeddings applied to both ``q`` and ``k``.
            k_pe: Separate rotary positional embeddings for ``k`` only. When
                *None*, ``pe`` is reused for keys.
            perturbation_mask: Optional mask in ``[0, 1]`` that
                blends the attention output with the raw value projection:
                ``out = attn_out * mask + v * (1 - mask)``.
                **1** keeps the full attention output, **0** bypasses attention
                and passes the value projection through unchanged.
                *None* or all-ones means standard attention; all-zeros skips
                the query/key path entirely for efficiency.
            all_perturbed: Whether all perturbations are active for this block.
        Returns:
            Output tensor of shape ``(B, T, query_dim)``.
        """
        if kv_cache is None:
            context = x if context is None else context
            dtype = torch.bfloat16
            x = x.to(dtype)
            context = context.to(dtype)
            use_attention = not all_perturbed

            v = self.to_v(context)

            if not use_attention:
                out = v
            else:
                q = self.to_q(x)
                k = self.to_k(context)

                q = self.q_norm(q)
                k = self.k_norm(k)

                if pe is not None:
                    q = apply_rotary_emb(q, pe, self.rope_type)
                    k = apply_rotary_emb(k, pe if k_pe is None else k_pe, self.rope_type)
                out = self.attention_function(q.to(dtype), k.to(dtype), v.to(dtype), self.heads, mask)  # (B, T, H*D)

                if perturbation_mask is not None:
                    out = out * perturbation_mask + v * (1 - perturbation_mask)

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

            return self.to_out(out)
        else:
            context = x if context is None else context
            x = x.to(dtype)
            context = context.to(dtype)
            use_attention = not all_perturbed

            is_text_cross_attn = (kv_cache_name == "video_cross_attn_kv_cache" or kv_cache_name == "audio_cross_attn_kv_cache")
            current_video_kv_cache_start = kv_cache_snapshot["current_video_kv_cache_start"]
            current_audio_kv_cache_start = kv_cache_snapshot["current_audio_kv_cache_start"]
            current_video_kv_cache_end = kv_cache_snapshot["current_video_kv_cache_end"]
            current_audio_kv_cache_end = kv_cache_snapshot["current_audio_kv_cache_end"]
            current_video_kv_cache_current_seqlen = kv_cache_snapshot["current_video_kv_cache_current_seqlen"]
            current_audio_kv_cache_current_seqlen = kv_cache_snapshot["current_audio_kv_cache_current_seqlen"]

            current_video_kv_cache_adj_seqlen = kv_cache_snapshot["current_video_kv_cache_adj_seqlen"]
            current_audio_kv_cache_adj_seqlen = kv_cache_snapshot["current_audio_kv_cache_adj_seqlen"]
            current_video_kv_cache_sink_seqlen = kv_cache_snapshot["current_video_kv_cache_sink_seqlen"]
            current_audio_kv_cache_sink_seqlen = kv_cache_snapshot["current_audio_kv_cache_sink_seqlen"]

            v_max_rope_end = kv_cache_snapshot["v_max_rope_end"]
            a_max_rope_end = kv_cache_snapshot["a_max_rope_end"]

            sigma_idx = kv_cache_snapshot["sigma_idx"]
            roll_kv = kv_cache_snapshot["roll_kv"]
            # pe_start may differ from kv_cache_start when running multi-segment inference:
            # kv_cache uses global token offsets, but pe is indexed within the current segment.
            video_pe_start = kv_cache_snapshot.get("video_pe_start", current_video_kv_cache_start)
            audio_pe_start = kv_cache_snapshot.get("audio_pe_start", current_audio_kv_cache_start)

            video_sink_perlatent_seqlen = current_video_kv_cache_sink_seqlen // 4
            audio_sink_perlatent_seqlen = 25

            # for text cross-attention, avoid recomputation of kv if kv cache is initialized
            if is_text_cross_attn:
                if kv_cache["is_init"] and use_text_encoder_cache:
                    v = kv_cache['v'][sigma_idx]
                else:
                    v = self.to_v(context)
            else:
                v = self.to_v(context)

            if not use_attention:
                out = v
            else:
                q = self.to_q(x)
                q = self.q_norm(q)
                
                # for text cross-attention, avoid recomputation of kv if kv cache is initialized
                if is_text_cross_attn:
                    if kv_cache["is_init"] and use_text_encoder_cache:
                        k = kv_cache['k'][sigma_idx]
                    else:
                        k = self.to_k(context)
                        k = self.k_norm(k)
                else:
                    k = self.to_k(context)
                    k = self.k_norm(k)

                q_pe = pe
                k_pe = pe if k_pe is None else k_pe
                
                # num_frame_per_block = 3, pre_block = 2
                k_cache = kv_cache['k']
                v_cache = kv_cache['v']
                
                if current_video_kv_cache_start > v_max_rope_end:
                    v_rope_start = v_max_rope_end # TODO: v_max_rope_end rename
                    a_rope_start = a_max_rope_end
                else:
                    v_rope_start = current_video_kv_cache_start
                    a_rope_start = current_audio_kv_cache_start

                with torch.no_grad():
                    v_rope_end = v_rope_start + current_video_kv_cache_current_seqlen
                    a_rope_end = a_rope_start + current_audio_kv_cache_current_seqlen
                    if is_text_cross_attn:
                        # for text cross-attention, we have enabled kv cache update now for the reason that
                        # the text kv cache is influenced by the timestep, have to do the following modifications:
                        # 1. save kv cache of all timesteps of all transformer blocks;
                        # 2. identify initialization of the kv cache for different transformer blocks,
                        #     avoid recomputation for the initialized kv cache;
                        k_cache[sigma_idx][:] = k.detach()
                        v_cache[sigma_idx][:] = v.detach()
                        if sigma_idx == len(kv_cache) - 1:
                            kv_cache["is_init"] = True
                    elif "video" in kv_cache_name or "v2a" in kv_cache_name:
                        k_cache[:, v_rope_start:v_rope_end] = k.detach()
                        v_cache[:, v_rope_start:v_rope_end] = v.detach()
                    else:
                        k_cache[:, a_rope_start:a_rope_end] = k.detach()
                        v_cache[:, a_rope_start:a_rope_end] = v.detach()

                if ("video" in kv_cache_name or "v2a" in kv_cache_name) and not is_text_cross_attn:
                    # 刚开始滑窗中间没有空隙
                    continuous_window = v_rope_start - current_video_kv_cache_adj_seqlen - current_video_kv_cache_sink_seqlen <= 0
                    if continuous_window: # current_video_kv_cache_start // v.shape[1] <= 3:
                        rope_start = 0
                        rope_end = video_pe_start + current_video_kv_cache_current_seqlen
                        k_pe = (k_pe[0][:, :, rope_start:rope_end],
                                k_pe[1][:, :, rope_start:rope_end])

                        # --- 预分配优化 ---
                        hist_len = v_rope_start
                        curr_len = current_video_kv_cache_current_seqlen
                        total_len = hist_len + curr_len
                        new_k = torch.empty((k.shape[0], total_len, *k.shape[2:]), dtype=k.dtype, device=k.device)
                        new_v = torch.empty((v.shape[0], total_len, *v.shape[2:]), dtype=v.dtype, device=v.device)
                        new_k[:, :hist_len] = k_cache[:, :v_rope_start]
                        new_k[:, hist_len:] = k
                        new_v[:, :hist_len] = v_cache[:, :v_rope_start]
                        new_v[:, hist_len:] = v
                        k, v = new_k, new_v
                        # k = torch.cat([k_cache[:, :v_rope_start], k], dim=1)
                        # v = torch.cat([v_cache[:, :v_rope_start], v], dim=1)
                    # 滑窗中间有空隙
                    else:  # current_video_kv_cache_start // v.shape[1] > 3:
                        # cache: 0 1 2 3 | 4 5 | 6
                        # pe index uses pe-local coords; cache index uses global kv cache coords
                        pe_n_adj_rope_start = video_pe_start - current_video_kv_cache_adj_seqlen
                        pe_n_adj_rope_end = video_pe_start + current_video_kv_cache_current_seqlen
                        cache_n_adj_rope_start = v_rope_start - current_video_kv_cache_adj_seqlen
                        sink_rope_start = 0
                        sink_rope_end = current_video_kv_cache_sink_seqlen
                        k_pe = (torch.cat([k_pe[0][:,:,sink_rope_start:sink_rope_end], k_pe[0][:,:,pe_n_adj_rope_start:pe_n_adj_rope_end]], dim=2),
                                torch.cat([k_pe[1][:,:,sink_rope_start:sink_rope_end], k_pe[1][:,:,pe_n_adj_rope_start:pe_n_adj_rope_end]], dim=2))
                        # k_sink = k_cache[:, sink_rope_start:sink_rope_end]
                        # v_sink = v_cache[:, sink_rope_start:sink_rope_end]

                        # pre-allocate
                        sink_len = sink_rope_end - sink_rope_start
                        adj_len = current_video_kv_cache_adj_seqlen
                        curr_len = current_video_kv_cache_current_seqlen
                        total_len = sink_len + adj_len + curr_len

                        new_k = torch.empty((k.shape[0], total_len, *k.shape[2:]), dtype=k.dtype, device=k.device)
                        new_v = torch.empty((v.shape[0], total_len, *v.shape[2:]), dtype=v.dtype, device=v.device)
                        new_k[:, :sink_len] = k_cache[:, sink_rope_start:sink_rope_end]
                        new_v[:, :sink_len] = v_cache[:, sink_rope_start:sink_rope_end]
                        # 写入 adj
                        new_k[:, sink_len : sink_len + adj_len] = k_cache[:, cache_n_adj_rope_start:v_rope_start]
                        new_v[:, sink_len : sink_len + adj_len] = v_cache[:, cache_n_adj_rope_start:v_rope_start]
                        # 写入 current
                        new_k[:, sink_len + adj_len :] = k
                        new_v[:, sink_len + adj_len :] = v

                        k, v = new_k, new_v
                        # k = torch.cat([k_sink, k_cache[:, current_n_adj_rope_start:v_rope_start], k],  dim=1)
                        # v = torch.cat([v_sink, v_cache[:, current_n_adj_rope_start:v_rope_start], v],  dim=1)

                elif not is_text_cross_attn:
                    continuous_window = a_rope_start - current_audio_kv_cache_adj_seqlen - current_audio_kv_cache_sink_seqlen <= 0
                    if continuous_window: # current_audio_kv_cache_start / v.shape[1] <= 3:
                        rope_start = 0
                        rope_end = audio_pe_start + current_audio_kv_cache_current_seqlen
                        k_pe = (k_pe[0][:, :, rope_start:rope_end],
                                k_pe[1][:, :, rope_start:rope_end])

                        # --- 预分配优化 ---
                        hist_len = a_rope_start
                        curr_len = current_audio_kv_cache_current_seqlen
                        total_len = hist_len + curr_len

                        new_k = torch.empty((k.shape[0], total_len, *k.shape[2:]), dtype=k.dtype, device=k.device)
                        new_v = torch.empty((v.shape[0], total_len, *v.shape[2:]), dtype=v.dtype, device=v.device)

                        new_k[:, :hist_len] = k_cache[:, :a_rope_start]
                        new_k[:, hist_len:] = k
                        new_v[:, :hist_len] = v_cache[:, :a_rope_start]
                        new_v[:, hist_len:] = v

                        k, v = new_k, new_v
                        # k = torch.cat([k_cache[:, :a_rope_start], k], dim=1)
                        # v = torch.cat([v_cache[:, :a_rope_start], v], dim=1)
                    else:  # if current_audio_kv_cache_start / v.shape[1] > 3:
                        pe_n_adj_rope_start = audio_pe_start - current_audio_kv_cache_adj_seqlen
                        pe_n_adj_rope_end = audio_pe_start + current_audio_kv_cache_current_seqlen
                        cache_n_adj_rope_start = a_rope_start - current_audio_kv_cache_adj_seqlen
                        sink_rope_start = 0
                        sink_rope_end = current_audio_kv_cache_sink_seqlen
                        k_pe = (torch.cat([k_pe[0][:,:,sink_rope_start:sink_rope_end], k_pe[0][:,:,pe_n_adj_rope_start:pe_n_adj_rope_end]], dim=2),
                                torch.cat([k_pe[1][:,:,sink_rope_start:sink_rope_end], k_pe[1][:,:,pe_n_adj_rope_start:pe_n_adj_rope_end]], dim=2))
                        # k_sink = k_cache[:, sink_rope_start:sink_rope_end]
                        # v_sink = v_cache[:, sink_rope_start:sink_rope_end]

                        # --- 预分配优化 ---
                        sink_len = sink_rope_end - sink_rope_start
                        adj_len = a_rope_start - cache_n_adj_rope_start
                        curr_len = k.shape[1]
                        total_len = sink_len + adj_len + curr_len

                        new_k = torch.empty((k.shape[0], total_len, *k.shape[2:]), dtype=k.dtype, device=k.device)
                        new_v = torch.empty((v.shape[0], total_len, *v.shape[2:]), dtype=v.dtype, device=v.device)

                        new_k[:, :sink_len] = k_cache[:, sink_rope_start:sink_rope_end]
                        new_v[:, :sink_len] = v_cache[:, sink_rope_start:sink_rope_end]

                        # 写入 adj
                        new_k[:, sink_len : sink_len + adj_len] = k_cache[:, cache_n_adj_rope_start:a_rope_start]
                        new_v[:, sink_len : sink_len + adj_len] = v_cache[:, cache_n_adj_rope_start:a_rope_start]

                        # 写入 current
                        new_k[:, sink_len + adj_len :] = k
                        new_v[:, sink_len + adj_len :] = v

                        k, v = new_k, new_v
                
                # q rope
                if ("video" in kv_cache_name or "a2v" in kv_cache_name) and not is_text_cross_attn:
                    q_pe_start = video_pe_start
                    q_pe_end = video_pe_start + current_video_kv_cache_current_seqlen
                    q_pe = (q_pe[0][:, :, q_pe_start:q_pe_end],
                            q_pe[1][:, :, q_pe_start:q_pe_end])
                elif ("audio" in kv_cache_name or "v2a" in kv_cache_name) and not is_text_cross_attn:
                    q_pe_start = audio_pe_start
                    q_pe_end = audio_pe_start + current_audio_kv_cache_current_seqlen
                    q_pe = (q_pe[0][:, :, q_pe_start:q_pe_end],
                            q_pe[1][:, :, q_pe_start:q_pe_end])

                if pe is not None:
                    q = apply_rotary_emb(q, q_pe, self.rope_type)
                    k = apply_rotary_emb(k, k_pe, self.rope_type)

                out = self.attention_function(q.to(dtype), k.to(dtype), v.to(dtype), self.heads, mask)  # (B, T, H*D)
                
                if perturbation_mask is not None:
                    out = out * perturbation_mask + v * (1 - perturbation_mask)

                # roll kv cache
                if roll_kv: 
                    if is_text_cross_attn:
                        pass
                    elif "video" in kv_cache_name or "v2a" in kv_cache_name:
                        # roll cache rule:
                        # roll the 'adj' and 'current' cache backward, 
                        # in the distance that equals the length of the 'current' cache
                        cache_from_idx_start = v_rope_start - current_video_kv_cache_adj_seqlen
                        cache_from_idx_end = v_rope_start + current_video_kv_cache_current_seqlen
                        cache_to_idx_start = cache_from_idx_start - current_video_kv_cache_current_seqlen
                        cache_to_idx_end = cache_from_idx_end - current_video_kv_cache_current_seqlen
                        # roll
                        k_cache[:, cache_to_idx_start:cache_to_idx_end] = k_cache[:, cache_from_idx_start:cache_from_idx_end].clone()
                        v_cache[:, cache_to_idx_start:cache_to_idx_end] = v_cache[:, cache_from_idx_start:cache_from_idx_end].clone()
                    else:
                        cache_from_idx_start = a_rope_start - current_audio_kv_cache_adj_seqlen
                        cache_from_idx_end = a_rope_start + current_audio_kv_cache_current_seqlen
                        cache_to_idx_start = cache_from_idx_start - current_audio_kv_cache_current_seqlen
                        cache_to_idx_end = cache_from_idx_end - current_audio_kv_cache_current_seqlen
                        # roll
                        k_cache[:, cache_to_idx_start:cache_to_idx_end] = k_cache[:, cache_from_idx_start:cache_from_idx_end].clone()
                        v_cache[:, cache_to_idx_start:cache_to_idx_end] = v_cache[:, cache_from_idx_start:cache_from_idx_end].clone()

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

            return self.to_out(out)