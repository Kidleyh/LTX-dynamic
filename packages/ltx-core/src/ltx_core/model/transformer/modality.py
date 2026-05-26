from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import torch

from ltx_core.types import VideoLatentShape, AudioLatentShape

@dataclass(frozen=True)
class Modality:
    """
    Input data for a single modality (video or audio) in the transformer.
    Bundles the latent tokens, timestep embeddings, positional information,
    and text conditioning context for processing by the diffusion transformer.
    Attributes:
        latent: Patchified latent tokens, shape ``(B, T, D)`` where *B* is
            the batch size, *T* is the total number of tokens (noisy +
            conditioning), and *D* is the input dimension.
        timesteps: Per-token timestep embeddings, shape ``(B, T)``.
        positions: Positional coordinates, shape ``(B, 3, T)`` for video
            (time, height, width) or ``(B, 1, T)`` for audio.
        context: Text conditioning embeddings from the prompt encoder.
        enabled: Whether this modality is active in the current forward pass.
        context_mask: Optional mask for the text context tokens.
        attention_mask: Optional 2-D self-attention mask, shape ``(B, T, T)``.
            Values in ``[0, 1]`` where ``1`` = full attention and ``0`` = no
            attention. ``None`` means unrestricted (full) attention between
            all tokens. Built incrementally by conditioning items; see
            :class:`~ltx_core.conditioning.types.attention_strength_wrapper.ConditioningItemAttentionStrengthWrapper`.
    """

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
    context_mask: torch.Tensor | None = None
    attention_mask: torch.Tensor | None = None
    original_shape: VideoLatentShape | AudioLatentShape | None = None

    def split(self, sizes: list[int]) -> list[Modality]:
        """Split along the batch dimension into chunks of the given sizes."""
        n = len(sizes)
        split_fields: dict[str, list[torch.Tensor | None] | list[bool]] = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if isinstance(value, torch.Tensor):
                split_fields[f.name] = list(value.split(sizes, dim=0))
            elif value is None or isinstance(value, bool):
                split_fields[f.name] = [value] * n
            elif isinstance(value, (VideoLatentShape, AudioLatentShape)):
                split_fields[f.name] = [value] * n
            else:
                raise TypeError(f"Cannot split field {f.name!r}: unsupported type {type(value)}")
        return [Modality(**{name: parts[i] for name, parts in split_fields.items()}) for i in range(n)]

@dataclass(frozen=False)
class KVCache:
    """
    Cache for all key-value pairs in the transformer.
    Attributes:
        video_self_attn_cache: Cache for video self-attention.
        audio_self_attn_cache: Cache for audio self-attention.
        video_cross_attn_cache: Cache for video cross-attention.
        audio_cross_attn_cache: Cache for audio cross-attention.
        a2v_cross_attn_cache: Cache for audio-to-video cross-attention.
        v2a_cross_attn_cache: Cache for video-to-audio cross-attention.s
    """
    # different types of cache
    video_self_attn_cache: Optional[List[dict]] = None
    audio_self_attn_cache: Optional[List[dict]] = None
    video_cross_attn_cache: Optional[List[dict]] = None
    audio_cross_attn_cache: Optional[List[dict]] = None
    a2v_cross_attn_cache: Optional[List[dict]] = None
    v2a_cross_attn_cache: Optional[List[dict]] = None
    # start index of the current block cache
    current_video_kv_cache_start: int | None = None # to be initialized
    current_audio_kv_cache_start: int | None = None # to be initialized
    # end index of the current block cache
    current_video_kv_cache_end: int | None = None # to be initialized
    current_audio_kv_cache_end: int | None = None # to be initialized
    # use current start and current seqlen to locate the current cache
    current_video_kv_cache_current_seqlen: int | None = None # to be initialized
    current_audio_kv_cache_current_seqlen: int | None = None # to be initialized
    # use current start and adj seqlen to locate the adjcent cache
    # use position 0 and the sink seqlen to locate the sink cache
    # two types of cache: sink and adjcent tokens
    current_video_kv_cache_adj_seqlen: int | None = None # to be initialized
    current_video_kv_cache_sink_seqlen: int | None = None # to be initialized
    current_audio_kv_cache_adj_seqlen: int | None = None # to be initialized
    current_audio_kv_cache_sink_seqlen: int | None = None # to be initialized
    # used for tracking the current transformer block index
    current_transformer_block_index: int | None = None
    # if the text cache is initialized or not
    is_initialized: bool = False
    v_max_rope_end: int = 10000000
    a_max_rope_end: int = 10000000