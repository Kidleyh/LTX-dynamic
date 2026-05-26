"""
Causal benchmark inference pipeline for LTX-2 AV generation.

This pipeline mirrors the ODE benchmark's prefix-rerun autoregressive strategy
instead of relying on the unfinished KV-cache runtime path in the tracked
causal wrapper.
"""

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import importlib
import os

from ltx_causal.attention.mask_builder import (
    compute_aligned_audio_frames,
    compute_av_blocks,
)
# from ltx_distillation.inference.bidirectional_pipeline import BidirectionalAVInferencePipeline
from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT, GENERAL_NOSPEECH_PROMPT
from ltx_pipelines.utils.types import PipelineComponents
from ltx_core.components.patchifiers import get_pixel_coords
from dataclasses import dataclass, replace
from ltx_core.model.transformer.modality import Modality
from accelerate.utils import broadcast
from PIL import Image
from torchvision import transforms
from einops import rearrange
from ltx_pipelines.utils.media_io import encode_video as save_video
from tqdm import tqdm
import random 
import datetime

class LTX23CausalAVInferencePipeline:
    """
    Prefix-rerun autoregressive pipeline for causal AV benchmark inference.

    `use_kv_cache` is kept for config compatibility, but the current causal
    wrapper does not expose a runnable KV-cache runtime API. We therefore always
    execute the prefix-rerun path, which matches the ODE benchmark semantics.
    """

    def __init__(
        self,
        generator: nn.Module,
        add_noise_fn,
        denoising_sigmas: torch.Tensor,
        num_frame_per_block: int = 3,
        num_audio_token_per_block: int = 25,
        device=None,
        dtype=None,
        use_kv_cache: bool = False,
        clear_cuda_cache_per_round: bool = True,
        accelerator=None,
        text_encoder=None,
        video_vae=None,
        audio_vae=None,
        text_encoder_device=None,
    ):
        if denoising_sigmas.ndim != 1 or denoising_sigmas.numel() < 2:
            raise ValueError(
                "denoising_sigmas must be a 1D tensor with at least 2 entries"
            )

        self.generator = generator
        self.add_noise_fn = add_noise_fn
        self.denoising_sigmas = denoising_sigmas
        self.num_frame_per_block = max(1, int(num_frame_per_block))
        self.use_kv_cache_requested = bool(use_kv_cache)
        self.clear_cuda_cache_per_round = bool(clear_cuda_cache_per_round)
        self.num_audio_token_per_block = max(1, int(num_audio_token_per_block))

        self.device = device
        self.dtype = dtype
        self.accelerator = accelerator
        self.text_encoder = text_encoder
        self.video_vae = video_vae
        self.audio_vae = audio_vae
        self.torch_rng = torch.Generator().manual_seed(42)
        self.text_encoder_device = text_encoder_device

    def _get_bootstrap_generator(self) -> nn.Module:
        get_delegate = getattr(self.generator, "_get_bidirectional_delegate", None)
        if callable(get_delegate):
            delegate = get_delegate()
            device, dtype = self._module_device_dtype(self.generator)
            return delegate.to(device=device, dtype=dtype)
        return self.generator

    def _release_bootstrap_generator(self, bootstrap_generator: nn.Module) -> None:
        if bootstrap_generator is self.generator:
            return
        bootstrap_generator.to(device="cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @staticmethod
    def _module_device_dtype(module: nn.Module) -> Tuple[torch.device, torch.dtype]:
        param = next(module.parameters())
        return param.device, param.dtype

    # @staticmethod
    # def _zeros_sigma(
    #     batch_size: int,
    #     frames: int,
    #     device: torch.device,
    #     dtype: torch.dtype,
    # ) -> torch.Tensor:
    #     return torch.zeros((batch_size, frames), device=device, dtype=dtype)

    # @staticmethod
    # def _full_sigma(
    #     sigma: torch.Tensor,
    #     batch_size: int,
    #     frames: int,
    #     device: torch.device,
    #     dtype: torch.dtype,
    # ) -> torch.Tensor:
    #     sigma_value = sigma.to(device=device, dtype=dtype)
    #     return sigma_value.expand(batch_size, frames)

    # def _renoise_block(self, clean_block: torch.Tensor, next_sigma: torch.Tensor) -> torch.Tensor:
    #     if clean_block is None:
    #         return None

    #     batch_size = clean_block.shape[0]
    #     num_frames = clean_block.shape[1]
    #     sigma = self._full_sigma(
    #         next_sigma,
    #         batch_size=batch_size,
    #         frames=num_frames,
    #         device=clean_block.device,
    #         dtype=clean_block.dtype,
    #     )
    #     return self.add_noise_fn(
    #         clean_block,
    #         torch.randn_like(clean_block),
    #         sigma,
    #     )

    @staticmethod
    def _merge_bootstrap_blocks(blocks):
            if len(blocks) < 2 or blocks[0].video_frames != 1:
                return blocks

            # 合并前两个块
            bootstrap = type(blocks[0])(
                block_idx=0,
                video_start=blocks[0].video_start,
                video_end=blocks[1].video_end,
                audio_start=blocks[0].audio_start,
                audio_end=blocks[1].audio_end,
            )

            # 重新映射后续块的 block_idx，从 1 开始累加
            # enumerate(blocks[2:], start=1) 会生成 (1, block), (2, block)...
            rest_blocks = [
                type(b)(
                    block_idx=i,
                    video_start=b.video_start,
                    video_end=b.video_end,
                    audio_start=b.audio_start,
                    audio_end=b.audio_end,
                ) 
                for i, b in enumerate(blocks[2:], start=1)
            ]

            return [bootstrap] + rest_blocks

    def generate_and_sync_list_accelerate(self, num_denoising_steps):
        # accelerator.is_main_process 表示当前进程是不是 “主进程” （rank = 0）
        device = self.accelerator.device

        # 在所有进程上先创建一个 empty 初始化张量（或者零张量）
        indices = torch.zeros(1, dtype=torch.long, device=device)

        if self.accelerator.is_main_process:
            temp = torch.randint(
                low=0,
                high=num_denoising_steps,
                size=(1,),
                device=device
            )
            indices = temp

        # 使用 accelerate 提供的 broadcast，把主进程的 indices 广播出去
        # 注意 broadcast 函数必须在所有进程都调用，一起同步
        indices = broadcast(indices)

        return indices.tolist()

    def _set_kv_cache(
        self,
        blocks,
        block_idx,
        video_seqlen_frame,
        roll_kv: bool = False,
        video_pe_start: int | None = None,
        audio_pe_start: int | None = None,
        ):

        # update
        self.generator.kv_cache.current_video_kv_cache_start = blocks[block_idx].video_start * video_seqlen_frame # block内的start/end是latent数序号
        self.generator.kv_cache.current_audio_kv_cache_start = blocks[block_idx].audio_start # audio已经是实际token长度了
        self.generator.kv_cache.current_video_kv_cache_end = blocks[block_idx].video_end * video_seqlen_frame
        self.generator.kv_cache.current_audio_kv_cache_end = blocks[block_idx].audio_end
        self.generator.kv_cache.current_video_kv_cache_current_seqlen = (blocks[block_idx].video_end - blocks[block_idx].video_start) * video_seqlen_frame
        self.generator.kv_cache.current_audio_kv_cache_current_seqlen = blocks[block_idx].audio_end - blocks[block_idx].audio_start

        # fixed
        self.generator.kv_cache.current_video_kv_cache_adj_seqlen = 2 * 3 * video_seqlen_frame
        self.generator.kv_cache.current_audio_kv_cache_adj_seqlen = 2 * self.num_audio_token_per_block
        self.generator.kv_cache.current_video_kv_cache_sink_seqlen = (1 + 3) * video_seqlen_frame
        self.generator.kv_cache.current_audio_kv_cache_sink_seqlen = 1 + self.num_audio_token_per_block

        kv_cache_snapshot = dict(
            current_video_kv_cache_start=self.generator.kv_cache.current_video_kv_cache_start,
            current_audio_kv_cache_start=self.generator.kv_cache.current_audio_kv_cache_start,
            current_video_kv_cache_end=self.generator.kv_cache.current_video_kv_cache_end,
            current_audio_kv_cache_end=self.generator.kv_cache.current_audio_kv_cache_end,
            current_video_kv_cache_current_seqlen=self.generator.kv_cache.current_video_kv_cache_current_seqlen,
            current_audio_kv_cache_current_seqlen=self.generator.kv_cache.current_audio_kv_cache_current_seqlen,
            current_video_kv_cache_adj_seqlen=self.generator.kv_cache.current_video_kv_cache_adj_seqlen,
            current_audio_kv_cache_adj_seqlen=self.generator.kv_cache.current_audio_kv_cache_adj_seqlen,
            current_video_kv_cache_sink_seqlen=self.generator.kv_cache.current_video_kv_cache_sink_seqlen,
            current_audio_kv_cache_sink_seqlen=self.generator.kv_cache.current_audio_kv_cache_sink_seqlen,
            v_max_rope_end=self.generator.kv_cache.v_max_rope_end,
            a_max_rope_end=self.generator.kv_cache.a_max_rope_end,
            roll_kv=roll_kv,
            video_pe_start=video_pe_start if video_pe_start is not None else self.generator.kv_cache.current_video_kv_cache_start,
            audio_pe_start=audio_pe_start if audio_pe_start is not None else self.generator.kv_cache.current_audio_kv_cache_start,
        )

        return kv_cache_snapshot


    def inference_with_trajectory(
        self,
        video_latent_state: LatentState,
        audio_latent_state: LatentState,
        video_latent_num_frames: int,
        text_context_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        device, dtype = self._module_device_dtype(self.generator)
        blocks = compute_av_blocks(
            total_video_latent_frames=video_latent_num_frames,
            num_frame_per_block=self.num_frame_per_block,
        )
        # blocks = self._merge_bootstrap_blocks(blocks)

        torch.cuda.empty_cache()
        batch_size, video_seqlen, feat_dim = video_latent_state.latent.shape 
        batch_size, audio_seqlen, feat_dim = audio_latent_state.latent.shape
        assert video_seqlen % video_latent_num_frames == 0
        num_blocks = len(blocks) # (video_latent_num_frames - 1) // self.num_frame_per_block # 第一帧和3(n==0)的block合为一个block
        video_seqlen_frame = video_seqlen // video_latent_num_frames

        # Step 1: Initialize KV cache to all zeros
        kv_cache_size = dict(
            video_self_attn_kv_cache_size=video_seqlen,
            video_cross_attn_kv_cache_size=1024,
            audio_self_attn_kv_cache_size=audio_seqlen,
            audio_cross_attn_kv_cache_size=1024,
            a2v_cross_attn_kv_cache_size=audio_seqlen,
            v2a_cross_attn_kv_cache_size=video_seqlen,
            num_sigmas=len(self.denoising_sigmas[:-1]),
        )

        self.generator._initialize_kv_cache(
            batch_size=batch_size,
            dtype=video_latent_state.latent.dtype,
            device=self.device,
            kv_cache_size=kv_cache_size,
        )

        # all_num_frames = [self.num_frame_per_block] * num_blocks
        num_denoising_steps = len(self.denoising_sigmas[:-1])
        exit_flags = self.generate_and_sync_list_accelerate(num_denoising_steps)
        ROLLOUT_CACHE_BLOCK_IDX = (video_latent_num_frames - 4) // 3 + 1 - 1 # 6

        def get_shuffled_list_torch(m, n, device='cpu'):
            """
            使用 torch.randperm 生成 m 到 n（包含）的随机打乱列表
            """
            # 计算区间内元素的总数
            num_blocks = n - m + 1
            
            # 生成 0 到 num_blocks-1 的随机排列，并加上偏移量 m
            # 然后直接转换为 Python 列表
            return torch.randperm(num_blocks, device=device) + m

        # randomly select 'num_select' block to backprop
        # num_select = 7
        # last_n = 7
        # last_block_idx = num_blocks - 1
        # start_block_idx = last_block_idx - last_n + 1
        # if self.accelerator.is_main_process:
        #     has_grad_block_index = get_shuffled_list_torch(start_block_idx, last_block_idx, device=self.device)[:num_select] # torch.randperm(num_blocks, device=self.device)[:num_select]
        # else:
        #     has_grad_block_index = torch.zeros(num_select, dtype=torch.long, device=self.device)
        # has_grad_block_index = broadcast(has_grad_block_index)
        # has_grad_block_index = set(has_grad_block_index.tolist())

        # video_output = []
        # audio_output = []

        video_output = torch.zeros(
            [batch_size, video_seqlen, feat_dim],
            device=video_latent_state.clean_latent.device,
            dtype=video_latent_state.clean_latent.dtype,
        )
        audio_output = torch.zeros(
            [batch_size, audio_seqlen, feat_dim],
            device=audio_latent_state.clean_latent.device,
            dtype=audio_latent_state.clean_latent.dtype,
        )

        for block in blocks:
            block_idx = block.block_idx
            has_grad_block = True # block_idx in has_grad_block_index

            # kv cache init
            kv_cache_snapshot = self._set_kv_cache(
                blocks,
                block_idx,
                video_seqlen_frame,
                roll_kv=False, # In-length training does not require rolling kv cache
            )
            kv_cache_list = self.generator.kv_cache_list

            video_noisy_input = video_latent_state.latent[:, \
                            self.generator.kv_cache.current_video_kv_cache_start:self.generator.kv_cache.current_video_kv_cache_end].to(device=self.device, dtype=self.dtype)
            audio_noisy_input = audio_latent_state.latent[:, \
                            self.generator.kv_cache.current_audio_kv_cache_start:self.generator.kv_cache.current_audio_kv_cache_end].to(device=self.device, dtype=self.dtype)

            if block_idx == 0:  # only for the first frame ti2av
                # mask
                first_block_denoise_mask = video_latent_state.denoise_mask[:, \
                                          self.generator.kv_cache.current_video_kv_cache_start:self.generator.kv_cache.current_video_kv_cache_end].to(device=self.device, dtype=self.dtype)
                first_block_clean_latent = video_latent_state.clean_latent[:, \
                                          self.generator.kv_cache.current_video_kv_cache_start:self.generator.kv_cache.current_video_kv_cache_end].to(device=self.device, dtype=self.dtype)
                video_noisy_input = video_noisy_input * first_block_denoise_mask + first_block_clean_latent * (1 - first_block_denoise_mask)

            for sigma_idx, sigma in enumerate(self.denoising_sigmas[:-1]):

                # all blocks exit at the same random sigma
                exit_flag = (sigma_idx == exit_flags[0]) 

                # update roll_kv flag
                kv_cache_snapshot['roll_kv'] = False # now do not move kv cache in training # (exit_flag and block_idx >= ROLLOUT_CACHE_BLOCK_IDX)
                kv_cache_snapshot['sigma_idx'] = sigma_idx

                video_latent_model_input = video_noisy_input
                audio_latent_model_input = audio_noisy_input

                if block_idx == 0:
                    video_timesteps = sigma * first_block_denoise_mask
                    video_latent_model_input = video_latent_model_input * first_block_denoise_mask + first_block_clean_latent * (1 - first_block_denoise_mask)
                else:
                    video_timesteps = sigma * torch.ones_like(video_latent_state.denoise_mask[:, self.generator.kv_cache.current_video_kv_cache_start:self.generator.kv_cache.current_video_kv_cache_end])
                audio_timesteps = sigma * torch.ones_like(audio_latent_state.denoise_mask[:, self.generator.kv_cache.current_audio_kv_cache_start:self.generator.kv_cache.current_audio_kv_cache_end])

                v_pos_start = self.generator.kv_cache.current_video_kv_cache_start
                v_pos_end = self.generator.kv_cache.current_video_kv_cache_end
                a_pos_start = self.generator.kv_cache.current_audio_kv_cache_start
                a_pos_end = self.generator.kv_cache.current_audio_kv_cache_end
                video_modality = Modality(
                    latent=video_latent_model_input.to(device=self.device, dtype=self.dtype),
                    sigma=sigma.repeat(batch_size).to(device=self.device, dtype=self.dtype),
                    timesteps=video_timesteps.to(device=self.device, dtype=self.dtype),
                    positions=video_latent_state.positions[:, :, v_pos_start:v_pos_end, :].to(device=self.device, dtype=self.dtype),
                    context=text_context_dict['v_context'].to(device=self.device, dtype=self.dtype),
                    enabled=True,
                    context_mask=None,
                    attention_mask=None,
                )
                audio_modality = Modality(
                    latent=audio_latent_model_input.to(device=self.device, dtype=self.dtype),
                    sigma=sigma.repeat(batch_size).to(device=self.device, dtype=self.dtype),
                    timesteps=audio_timesteps.to(device=self.device, dtype=self.dtype),
                    positions=audio_latent_state.positions[:, :, a_pos_start:a_pos_end, :].to(device=self.device, dtype=self.dtype),
                    context=text_context_dict['a_context'].to(device=self.device, dtype=self.dtype),
                    enabled=True,
                    context_mask=None,
                    attention_mask=None,
                )

                if not exit_flag:
                    self.generator.model.eval()
                    with torch.no_grad():
                        # already input the kv_cache after kv_cache initialization
                        pred_video, pred_audio = self.generator(
                            video_modality,
                            audio_modality,
                            perturbations=None,
                            kv_cache_list=kv_cache_list,
                            kv_cache_snapshot=kv_cache_snapshot,
                        )

                        next_sigma = self.denoising_sigmas[sigma_idx+1]
                        assert next_sigma > 0
                        fresh_noise_video = torch.randn_like(pred_video)
                        fresh_noise_audio = torch.randn_like(pred_audio)

                        next_video_sigma = next_sigma * torch.ones([batch_size, pred_video.shape[1]], device=self.device)
                        next_audio_sigma = next_sigma * torch.ones([batch_size, pred_audio.shape[1]], device=self.device)

                        video_noisy_input = self.add_noise_fn(
                            pred_video.flatten(0, 1),
                            fresh_noise_video.flatten(0, 1),
                            next_video_sigma.flatten(0, 1),
                        ).unflatten(0, (batch_size, pred_video.shape[1]))

                        audio_noisy_input = self.add_noise_fn(
                            pred_audio, fresh_noise_audio, next_audio_sigma
                        )
                else:
                    def _exit_forward(vm, am, _dummy,
                                      _kv=kv_cache_list, _snap=kv_cache_snapshot):

                        return self.generator(
                            video=vm,
                            audio=am,
                            perturbations=None,
                            kv_cache_list=_kv,
                            kv_cache_snapshot=_snap,
                            )
                    self.generator.model.train()
                    if has_grad_block:
                        pred_video, pred_audio = torch.utils.checkpoint.checkpoint(
                            _exit_forward, 
                            video_modality, audio_modality, 
                            torch.tensor(0.0, device=self.device, requires_grad=True),
                            use_reentrant=True,
                        )
                        
                    else:
                        self.generator.model.eval()
                        with torch.no_grad():
                            pred_video, pred_audio = self.generator(
                                video_modality,
                                audio_modality,
                                perturbations=None,
                                kv_cache_list=kv_cache_list,
                                kv_cache_snapshot=kv_cache_snapshot,
                            )
                    # only for the first frame ti2av
                    if block_idx == 0:
                        pred_video = pred_video * first_block_denoise_mask + first_block_clean_latent * (1 - first_block_denoise_mask)

                    break

            # video_output.append(pred_video)
            # audio_output.append(pred_audio)
            video_output[:, block.video_start*video_seqlen_frame:block.video_end*video_seqlen_frame, :] = pred_video
            audio_output[:, block.audio_start:block.audio_end, :] = pred_audio

            torch.cuda.empty_cache()

        rollout_log = {"dmdtrain_generator_sigma": sigma.detach().item()}

        # video_output = torch.cat(video_output, dim=1)
        # audio_output = torch.cat(audio_output, dim=1)

        return video_output, audio_output, rollout_log


    def inference_with_persistent_kv_cache(
        self,
        video_latent_state: LatentState,
        audio_latent_state: LatentState,
        video_latent_num_frames: int,
        text_context_dict: Dict[str, Any],
        persistent_kv_cache_list: Optional[list] = None,
        segment_video_offset: int = 0,
        compute_grad: bool = True,
        prev_video_seqlen_frame: Optional[int] = None,
        return_replay_state: bool = False,
        shared_exit_step: Optional[int] = None,
    ):
        """
        Run one segment of the self-forcing rollout, optionally reusing a KV cache
        populated by a previous segment call.

        Args:
            persistent_kv_cache_list: The kv_cache_list from the previous segment (detached
                tensors).  Pass None for the first segment.
            segment_video_offset: Global video-token offset of the first frame of this
                segment.  Used to restore kv_cache cursor positions.
            compute_grad: When False the entire segment runs under torch.no_grad() and
                the returned kv_cache tensors are detached from the graph.
            return_replay_state: When True, the rollout runs entirely under no_grad and
                this method additionally returns a list of `replay_states`, one per
                rollout block, that the caller can pass to `replay_block_exit_forward`
                to re-run each block's exit-step forward with autograd enabled. This
                bounds peak backward memory to one block (RELIC §4.4.2 Replayed
                Back-Propagation). When return_replay_state is True, compute_grad is
                forced to False because the outer rollout doesn't track gradients.

        Returns:
            (video_output, audio_output, rollout_log, updated_kv_cache_list[, replay_states])
        """
        # In return_replay_state mode, the rollout itself is no-grad; caller will
        # re-run per-block exit forwards via replay_block_exit_forward.
        if return_replay_state:
            compute_grad = False
        device, dtype = self._module_device_dtype(self.generator)
        blocks = compute_av_blocks(
            total_video_latent_frames=video_latent_num_frames,
            num_frame_per_block=self.num_frame_per_block,
        )

        torch.cuda.empty_cache()
        batch_size, video_seqlen, feat_dim = video_latent_state.latent.shape
        batch_size, audio_seqlen, _ = audio_latent_state.latent.shape
        assert video_seqlen % video_latent_num_frames == 0
        video_seqlen_frame = video_seqlen // video_latent_num_frames
        # Token-per-frame for the previous segment (may differ if spatial resolution changed).
        restore_seqlen_frame = prev_video_seqlen_frame if prev_video_seqlen_frame is not None else video_seqlen_frame

        # Buffer must cover both the previous-segment prefix (segment_video_offset frames)
        # and the current segment, because offset_block shifts write positions by the offset.
        total_video_latent_frames = video_latent_num_frames + segment_video_offset
        total_video_seqlen = video_seqlen + segment_video_offset * restore_seqlen_frame
        audio_offset = compute_aligned_audio_frames(
            total_video_latent_frames=segment_video_offset,
            num_frame_per_block=self.num_frame_per_block,
        ) if segment_video_offset > 0 else 0
        total_audio_seqlen = audio_seqlen + audio_offset

        # Build global positions covering [0, total_video_seqlen) so that causal_attention's
        # pe slicing with global offsets (video_pe_start = offset_block.video_start * seqlen_frame)
        # correctly finds sink and adj tokens in the pe tensor.
        # Tile the current-segment positions to cover the full global range: spatial (h,w) coords
        # repeat identically across segments, and the prefix KV is already stored in the cache.
        if segment_video_offset > 0:
            repeats = (total_video_seqlen + video_seqlen - 1) // video_seqlen
            global_video_positions = video_latent_state.positions.repeat(1, 1, repeats, 1)[:, :, :total_video_seqlen, :]
            global_audio_positions = audio_latent_state.positions.repeat(
                1, 1, (total_audio_seqlen + audio_seqlen - 1) // audio_seqlen, 1
            )[:, :, :total_audio_seqlen, :]
        else:
            global_video_positions = video_latent_state.positions
            global_audio_positions = audio_latent_state.positions

        kv_cache_size = dict(
            video_self_attn_kv_cache_size=total_video_seqlen,
            video_cross_attn_kv_cache_size=1024,
            audio_self_attn_kv_cache_size=total_audio_seqlen,
            audio_cross_attn_kv_cache_size=1024,
            a2v_cross_attn_kv_cache_size=total_audio_seqlen,
            v2a_cross_attn_kv_cache_size=total_video_seqlen,
            num_sigmas=len(self.denoising_sigmas[:-1]),
        )

        self.generator._initialize_kv_cache(
            batch_size=batch_size,
            dtype=video_latent_state.latent.dtype,
            device=self.device,
            kv_cache_size=kv_cache_size,
        )

        # Restore the KV cache content written by the previous segment.
        # Each previous-segment self-attention slot covers `segment_video_offset` tokens
        # at the beginning of the fresh cache buffers.
        if persistent_kv_cache_list is not None:
            prev_video_len = segment_video_offset
            # Use the token-per-frame count from the previous segment when provided;
            # the current segment may have a different spatial resolution.
            restore_seqlen_frame = prev_video_seqlen_frame if prev_video_seqlen_frame is not None else video_seqlen_frame
            prev_audio_len = compute_aligned_audio_frames(
                total_video_latent_frames=segment_video_offset,
                num_frame_per_block=self.num_frame_per_block,
            ) if segment_video_offset > 0 else 0
            for blk_idx, blk_cache in enumerate(persistent_kv_cache_list):
                cur = self.generator.kv_cache_list[blk_idx]
                if prev_video_len > 0:
                    cur["video_self_attn_kv_cache"]["k"][:, :prev_video_len * restore_seqlen_frame] = \
                        blk_cache["video_self_attn_kv_cache"]["k"][:, :prev_video_len * restore_seqlen_frame].detach()
                    cur["video_self_attn_kv_cache"]["v"][:, :prev_video_len * restore_seqlen_frame] = \
                        blk_cache["video_self_attn_kv_cache"]["v"][:, :prev_video_len * restore_seqlen_frame].detach()
                    # v2a is indexed by video positions (video queries attend to audio kv)
                    cur["v2a_cross_attn_kv_cache"]["k"][:, :prev_video_len * restore_seqlen_frame] = \
                        blk_cache["v2a_cross_attn_kv_cache"]["k"][:, :prev_video_len * restore_seqlen_frame].detach()
                    cur["v2a_cross_attn_kv_cache"]["v"][:, :prev_video_len * restore_seqlen_frame] = \
                        blk_cache["v2a_cross_attn_kv_cache"]["v"][:, :prev_video_len * restore_seqlen_frame].detach()
                if prev_audio_len > 0:
                    cur["audio_self_attn_kv_cache"]["k"][:, :prev_audio_len] = \
                        blk_cache["audio_self_attn_kv_cache"]["k"][:, :prev_audio_len].detach()
                    cur["audio_self_attn_kv_cache"]["v"][:, :prev_audio_len] = \
                        blk_cache["audio_self_attn_kv_cache"]["v"][:, :prev_audio_len].detach()
                    # a2v is indexed by audio positions (audio queries attend to video kv)
                    cur["a2v_cross_attn_kv_cache"]["k"][:, :prev_audio_len] = \
                        blk_cache["a2v_cross_attn_kv_cache"]["k"][:, :prev_audio_len].detach()
                    cur["a2v_cross_attn_kv_cache"]["v"][:, :prev_audio_len] = \
                        blk_cache["a2v_cross_attn_kv_cache"]["v"][:, :prev_audio_len].detach()
                # Clear each old per-block buffer in-place as soon as it is copied
                # so its storage can be reused while the next blocks are restored.
                # The caller still references persistent_kv_cache_list, so deleting
                # our local won't free it; we have to drop the tensors directly.
                blk_cache["video_self_attn_kv_cache"]["k"] = None
                blk_cache["video_self_attn_kv_cache"]["v"] = None
                blk_cache["v2a_cross_attn_kv_cache"]["k"] = None
                blk_cache["v2a_cross_attn_kv_cache"]["v"] = None
                blk_cache["audio_self_attn_kv_cache"]["k"] = None
                blk_cache["audio_self_attn_kv_cache"]["v"] = None
                blk_cache["a2v_cross_attn_kv_cache"]["k"] = None
                blk_cache["a2v_cross_attn_kv_cache"]["v"] = None
                if "video_cross_attn_kv_cache" in blk_cache:
                    blk_cache["video_cross_attn_kv_cache"]["k"] = None
                    blk_cache["video_cross_attn_kv_cache"]["v"] = None
                if "audio_cross_attn_kv_cache" in blk_cache:
                    blk_cache["audio_cross_attn_kv_cache"]["k"] = None
                    blk_cache["audio_cross_attn_kv_cache"]["v"] = None
            torch.cuda.empty_cache()

        num_denoising_steps = len(self.denoising_sigmas[:-1])
        if shared_exit_step is not None:
            exit_flags = [shared_exit_step]
        else:
            exit_flags = self.generate_and_sync_list_accelerate(num_denoising_steps)
        ROLLOUT_CACHE_BLOCK_IDX = (video_latent_num_frames - 4) // 3 + 1 - 1

        video_output = torch.zeros(
            [batch_size, video_seqlen, feat_dim],
            device=video_latent_state.clean_latent.device,
            dtype=video_latent_state.clean_latent.dtype,
        )
        audio_output = torch.zeros(
            [batch_size, audio_seqlen, feat_dim],
            device=audio_latent_state.clean_latent.device,
            dtype=audio_latent_state.clean_latent.dtype,
        )

        # When return_replay_state, capture the per-block exit-step inputs so the
        # caller can re-run only block `l`'s forward with autograd. The full rollout
        # runs under no_grad, so the captured inputs are detached.
        replay_states: list = [] if return_replay_state else None

        # When training the 2nd+ segment, freeze the RoPE pe_start at seg0's last
        # block so that every new block of seg1 rotates Q/K as if it lived at that
        # same temporal position — matching the streaming inference behavior where
        # v_max_rope_end pins the rolling window to the last in-length block.
        # (sink remains at PE position 0 via the existing prefix-token slice; adj
        # is sliced as `pe_start - adj_seqlen .. pe_start`, so the 6 frames just
        # before the pinned pe_start are reused for every seg1 block.)
        if segment_video_offset > 0:
            prev_blocks = compute_av_blocks(
                total_video_latent_frames=segment_video_offset,
                num_frame_per_block=self.num_frame_per_block,
            )
            seg0_last = prev_blocks[-1]
            # Use the current segment's token-per-frame for the PE index because
            # the model's PE tensor is computed from seg1's positions; the cache
            # cursor `current_video_kv_cache_start` is also strided in
            # `video_seqlen_frame` units, so v_max_rope_end must be in the same
            # unit for the `current > v_max_rope_end` comparison to fire. This
            # assumes seg1 shares (H, W) with seg0, which the trainer enforces by
            # pinning seg0_batch across segments.
            frozen_video_pe_start = seg0_last.video_start * video_seqlen_frame
            frozen_audio_pe_start = seg0_last.audio_start
            # Pin v_rope_start/a_rope_start to the seg0 last-block boundary so
            # cache writes stay in the same slot for every seg1 block (matches
            # the streaming-inference behavior past ROLLOUT_CACHE_BLOCK_IDX).
            # The `kv_cache_snapshot` built by `_set_kv_cache` reads these
            # attributes off the cache instance, so the override propagates
            # into every per-block snapshot below.
            self.generator.kv_cache.v_max_rope_end = frozen_video_pe_start
            self.generator.kv_cache.a_max_rope_end = frozen_audio_pe_start
        else:
            frozen_video_pe_start = None
            frozen_audio_pe_start = None

        sigma = None
        for block in blocks:
            block_idx = block.block_idx

            # Offset the kv_cache cursor by the previous segment so that attention
            # sees all previously generated tokens as context.
            offset_block = type(block)(
                block_idx=block.block_idx,
                video_start=block.video_start + segment_video_offset,
                video_end=block.video_end + segment_video_offset,
                audio_start=block.audio_start + (
                    compute_aligned_audio_frames(segment_video_offset, self.num_frame_per_block)
                    if segment_video_offset > 0 else 0
                ),
                audio_end=block.audio_end + (
                    compute_aligned_audio_frames(segment_video_offset, self.num_frame_per_block)
                    if segment_video_offset > 0 else 0
                ),
            )

            if frozen_video_pe_start is not None:
                cur_video_pe_start = frozen_video_pe_start
                cur_audio_pe_start = frozen_audio_pe_start
            else:
                cur_video_pe_start = offset_block.video_start * video_seqlen_frame
                cur_audio_pe_start = offset_block.audio_start

            kv_cache_snapshot = self._set_kv_cache(
                [offset_block],
                0,
                video_seqlen_frame,
                roll_kv=False,
                video_pe_start=cur_video_pe_start,
                audio_pe_start=cur_audio_pe_start,
            )
            kv_cache_list = self.generator.kv_cache_list

            video_noisy_input = video_latent_state.latent[:, \
                block.video_start * video_seqlen_frame:block.video_end * video_seqlen_frame].to(device=self.device, dtype=self.dtype)
            audio_noisy_input = audio_latent_state.latent[:, \
                block.audio_start:block.audio_end].to(device=self.device, dtype=self.dtype)

            if block_idx == 0:
                first_block_denoise_mask = video_latent_state.denoise_mask[:, \
                    block.video_start * video_seqlen_frame:block.video_end * video_seqlen_frame].to(device=self.device, dtype=self.dtype)
                first_block_clean_latent = video_latent_state.clean_latent[:, \
                    block.video_start * video_seqlen_frame:block.video_end * video_seqlen_frame].to(device=self.device, dtype=self.dtype)
                video_noisy_input = video_noisy_input * first_block_denoise_mask + first_block_clean_latent * (1 - first_block_denoise_mask)

            for sigma_idx, sigma in enumerate(self.denoising_sigmas[:-1]):
                exit_flag = (sigma_idx == exit_flags[0])
                # On seg1+: roll the adj/current KV cache at the exit step of each
                # uniform (3-frame) block so seg1 block_N's adj contains the most
                # recent two blocks (seg1_block_{N-1}, seg1_block_{N-2}) while
                # their PE positions stay pinned at seg0's pre-last two blocks.
                # Skip block 0 of seg1 (4-frame transition block) — rolling with
                # curr_seqlen=4 would clip the last frame of the sink slot.
                kv_cache_snapshot['roll_kv'] = (
                    exit_flag
                    and frozen_video_pe_start is not None
                    and block_idx > 0
                )
                kv_cache_snapshot['sigma_idx'] = sigma_idx

                if block_idx == 0:
                    video_timesteps = sigma * first_block_denoise_mask
                    video_latent_model_input = video_noisy_input * first_block_denoise_mask + first_block_clean_latent * (1 - first_block_denoise_mask)
                else:
                    v_local_start = block.video_start * video_seqlen_frame
                    v_local_end = block.video_end * video_seqlen_frame
                    video_timesteps = sigma * torch.ones_like(video_latent_state.denoise_mask[:, v_local_start:v_local_end])
                    video_latent_model_input = video_noisy_input
                a_local_start = block.audio_start
                a_local_end = block.audio_end
                audio_timesteps = sigma * torch.ones_like(audio_latent_state.denoise_mask[:, a_local_start:a_local_end])

                # Pass global positions [0, offset_block.end) so that the PE tensor is
                # large enough for video_pe_start (= offset_block.video_start * sf) to be
                # a valid index. The q_pe slice in causal_attention uses video_pe_start as
                # an absolute index into the PE, so the PE must cover the full prefix.
                video_modality = Modality(
                    latent=video_latent_model_input.to(device=self.device, dtype=self.dtype),
                    sigma=sigma.repeat(batch_size).to(device=self.device, dtype=self.dtype),
                    timesteps=video_timesteps.to(device=self.device, dtype=self.dtype),
                    positions=global_video_positions[:, :, :offset_block.video_end * video_seqlen_frame, :].to(device=self.device, dtype=self.dtype),
                    context=text_context_dict['v_context'].to(device=self.device, dtype=self.dtype),
                    enabled=True,
                    context_mask=None,
                    attention_mask=None,
                )
                audio_modality = Modality(
                    latent=audio_noisy_input.to(device=self.device, dtype=self.dtype),
                    sigma=sigma.repeat(batch_size).to(device=self.device, dtype=self.dtype),
                    timesteps=audio_timesteps.to(device=self.device, dtype=self.dtype),
                    positions=global_audio_positions[:, :, :offset_block.audio_end, :].to(device=self.device, dtype=self.dtype),
                    context=text_context_dict['a_context'].to(device=self.device, dtype=self.dtype),
                    enabled=True,
                    context_mask=None,
                    attention_mask=None,
                )

                if not exit_flag:
                    self.generator.model.eval()
                    with torch.no_grad():
                        pred_video, pred_audio = self.generator(
                            video_modality,
                            audio_modality,
                            perturbations=None,
                            kv_cache_list=kv_cache_list,
                            kv_cache_snapshot=kv_cache_snapshot,
                        )
                        next_sigma = self.denoising_sigmas[sigma_idx + 1]
                        assert next_sigma > 0
                        fresh_noise_video = torch.randn_like(pred_video)
                        fresh_noise_audio = torch.randn_like(pred_audio)
                        next_video_sigma = next_sigma * torch.ones([batch_size, pred_video.shape[1]], device=self.device)
                        next_audio_sigma = next_sigma * torch.ones([batch_size, pred_audio.shape[1]], device=self.device)
                        video_noisy_input = self.add_noise_fn(
                            pred_video.flatten(0, 1), fresh_noise_video.flatten(0, 1), next_video_sigma.flatten(0, 1),
                        ).unflatten(0, (batch_size, pred_video.shape[1]))
                        audio_noisy_input = self.add_noise_fn(pred_audio, fresh_noise_audio, next_audio_sigma)
                else:
                    def _exit_forward(vm, am, _dummy, _kv=kv_cache_list, _snap=kv_cache_snapshot):
                        return self.generator(vm, am, perturbations=None, kv_cache_list=_kv, kv_cache_snapshot=_snap)

                    if return_replay_state:
                        # Run the exit-step forward under no_grad and stash the
                        # inputs needed to replay block-by-block with autograd
                        # (RELIC §4.4.2 replayed back-propagation).
                        self.generator.model.eval()
                        with torch.no_grad():
                            pred_video, pred_audio = self.generator(
                                video_modality, audio_modality,
                                perturbations=None,
                                kv_cache_list=kv_cache_list,
                                kv_cache_snapshot=kv_cache_snapshot,
                            )
                        replay_states.append({
                            "block_idx": block_idx,
                            "video_start": block.video_start,
                            "video_end": block.video_end,
                            "audio_start": block.audio_start,
                            "audio_end": block.audio_end,
                            "video_modality": video_modality,
                            "audio_modality": audio_modality,
                            # snapshot is a dict of plain ints/bools — safe to keep
                            "kv_cache_snapshot": dict(kv_cache_snapshot),
                            "first_block_denoise_mask": first_block_denoise_mask if block_idx == 0 else None,
                            "first_block_clean_latent": first_block_clean_latent if block_idx == 0 else None,
                        })
                    elif compute_grad:
                        self.generator.model.train()
                        # Two-level checkpointing:
                        #   - Outer (this checkpoint): saves only the rollout-block
                        #     inputs; recompute rebuilds the per-block graph during
                        #     backward, so only one rollout block's intermediate
                        #     tensors live at a time across the rollout chain.
                        #   - Inner (per-transformer-block in _process_transformer_blocks):
                        #     during the outer recompute, saves only inputs to each
                        #     of the 48 transformer blocks, so peak activations are
                        #     one block's worth instead of all 48 stacked.
                        # Without the outer, all 7 rollout blocks' exit-step graphs
                        # live concurrently. Without the inner, the recompute itself
                        # OOMs from 48 stacked block activations.
                        pred_video, pred_audio = torch.utils.checkpoint.checkpoint(
                            _exit_forward,
                            video_modality, audio_modality,
                            torch.tensor(0.0, device=self.device, requires_grad=True),
                            use_reentrant=False,
                        )
                    else:
                        self.generator.model.eval()
                        with torch.no_grad():
                            pred_video, pred_audio = self.generator(
                                video_modality, audio_modality,
                                perturbations=None,
                                kv_cache_list=kv_cache_list,
                                kv_cache_snapshot=kv_cache_snapshot,
                            )

                    if block_idx == 0:
                        pred_video = pred_video * first_block_denoise_mask + first_block_clean_latent * (1 - first_block_denoise_mask)

                    break

            video_output[:, block.video_start * video_seqlen_frame:block.video_end * video_seqlen_frame] = pred_video
            audio_output[:, block.audio_start:block.audio_end] = pred_audio
            torch.cuda.empty_cache()

        rollout_log = {"dmdtrain_generator_sigma": sigma.detach().item() if sigma is not None else 0.0,
                       "exit_step": exit_flags[0]}

        # Snapshot the updated kv_cache_list (detach to avoid holding the graph).
        def _detach_kv(x):
            if isinstance(x, list):
                return [t.detach() for t in x]
            return x.detach()

        updated_kv_cache_list = [
            {
                k: (
                    {"k": _detach_kv(v["k"]), "v": _detach_kv(v["v"]),
                     **({"is_init": v["is_init"]} if "is_init" in v else {})}
                    if isinstance(v, dict) else v
                )
                for k, v in blk.items()
            }
            for blk in self.generator.kv_cache_list
        ]

        if return_replay_state:
            return video_output, audio_output, rollout_log, updated_kv_cache_list, replay_states
        return video_output, audio_output, rollout_log, updated_kv_cache_list

    def replay_block_exit_forward(self, replay_state: Dict[str, Any]):
        """Re-run one rollout block's exit-step generator forward with autograd
        enabled, reading from the already-populated KV cache.

        Used by RELIC §4.4.2 replayed back-propagation: rollout was run under
        no_grad and captured `replay_states`; this method reconstructs each
        block's clean prediction with a live autograd graph, so the caller can
        backprop a per-block loss and free the graph before the next block.

        The KV cache (`self.generator.kv_cache_list`) is assumed to already be
        populated by a preceding `inference_with_persistent_kv_cache(..., return_replay_state=True)`
        call. We update only the snapshot cursor so attention slices the right
        prefix; the cache contents are reused as-is. Cache writes inside
        causal_attention run under torch.no_grad() and use detached k/v, so
        replaying the forward overwrites the same positions with identical
        values — no autograd interference.
        """
        self.generator.model.train()
        snapshot = replay_state["kv_cache_snapshot"]
        kv_cache_list = self.generator.kv_cache_list
        # Mirror cursor state on self.generator.kv_cache so downstream code that
        # reads attributes off it (e.g. for sigma-based bookkeeping) sees the
        # right values, matching the rollout-time setup.
        kv = self.generator.kv_cache
        kv.current_video_kv_cache_start = snapshot["current_video_kv_cache_start"]
        kv.current_audio_kv_cache_start = snapshot["current_audio_kv_cache_start"]
        kv.current_video_kv_cache_end = snapshot["current_video_kv_cache_end"]
        kv.current_audio_kv_cache_end = snapshot["current_audio_kv_cache_end"]
        kv.current_video_kv_cache_current_seqlen = snapshot["current_video_kv_cache_current_seqlen"]
        kv.current_audio_kv_cache_current_seqlen = snapshot["current_audio_kv_cache_current_seqlen"]
        kv.current_video_kv_cache_adj_seqlen = snapshot["current_video_kv_cache_adj_seqlen"]
        kv.current_audio_kv_cache_adj_seqlen = snapshot["current_audio_kv_cache_adj_seqlen"]
        kv.current_video_kv_cache_sink_seqlen = snapshot["current_video_kv_cache_sink_seqlen"]
        kv.current_audio_kv_cache_sink_seqlen = snapshot["current_audio_kv_cache_sink_seqlen"]

        pred_video, pred_audio = self.generator(
            replay_state["video_modality"],
            replay_state["audio_modality"],
            perturbations=None,
            kv_cache_list=kv_cache_list,
            kv_cache_snapshot=snapshot,
        )
        if replay_state["block_idx"] == 0:
            mask = replay_state["first_block_denoise_mask"]
            clean = replay_state["first_block_clean_latent"]
            pred_video = pred_video * mask + clean * (1 - mask)
        return pred_video, pred_audio


    ######## below are inference pipeline code ########

    @torch.no_grad()
    def inference_with_trajectory_inference(
        self,
        video_latent_state: LatentState,
        audio_latent_state: LatentState,
        video_latent_num_frames: int,
        video_latent_num_frames_output: int,
        text_context_dict: Dict[str, Any],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # device, dtype = self._module_device_dtype(self.generator)
        blocks = compute_av_blocks(
            total_video_latent_frames=video_latent_num_frames_output,
            num_frame_per_block=self.num_frame_per_block,
        )
        # blocks = self._merge_bootstrap_blocks(blocks)

        torch.cuda.empty_cache()
        batch_size, video_seqlen, feat_dim = video_latent_state.latent.shape 
        batch_size, audio_seqlen, feat_dim = audio_latent_state.latent.shape
        assert video_seqlen % video_latent_num_frames == 0
        num_blocks = len(blocks) # (video_latent_num_frames - 1) // self.num_frame_per_block # 第一帧和3(n==0)的block合为一个block
        video_seqlen_frame = video_seqlen // video_latent_num_frames

        # Step 1: Initialize KV cache to all zeros
        kv_cache_size = dict(
            video_self_attn_kv_cache_size=video_seqlen,
            video_cross_attn_kv_cache_size=1024,
            audio_self_attn_kv_cache_size=audio_seqlen,
            audio_cross_attn_kv_cache_size=1024,
            a2v_cross_attn_kv_cache_size=audio_seqlen,
            v2a_cross_attn_kv_cache_size=video_seqlen,
            num_sigmas=len(self.denoising_sigmas[:-1]),
        )

        self.generator._initialize_kv_cache(
            batch_size=batch_size,
            dtype=video_latent_state.latent.dtype,
            device=self.device,
            kv_cache_size=kv_cache_size,
        )

        # all_num_frames = [self.num_frame_per_block] * num_blocks
        num_denoising_steps = len(self.denoising_sigmas[:-1])
        video_output = []
        audio_output = []
        ROLLOUT_CACHE_BLOCK_IDX = (video_latent_num_frames - 4) // 3 + 1 - 1 # 6

        for block in tqdm(blocks, desc="Inference blocks"):
            block_idx = block.block_idx

            # initialize kv cache
            kv_cache_list = self.generator.kv_cache_list
            kv_cache_snapshot = self._set_kv_cache(
                blocks,
                block_idx,
                video_seqlen_frame,
                roll_kv=False,
            )

            video_noisy_input = torch.randn_like(video_latent_state.latent[:, \
                            :self.generator.kv_cache.current_video_kv_cache_end-self.generator.kv_cache.current_video_kv_cache_start] \
                                ).to(device=self.device, dtype=self.dtype)
            audio_noisy_input = torch.randn_like(audio_latent_state.latent[:, \
                            :self.generator.kv_cache.current_audio_kv_cache_end-self.generator.kv_cache.current_audio_kv_cache_start] \
                                ).to(device=self.device, dtype=self.dtype)
            
            # video_noisy_input = video_latent_state.latent[:, \
            #                 self.generator.kv_cache.current_video_kv_cache_start:self.generator.kv_cache.current_video_kv_cache_end].to(device=self.device, dtype=self.dtype)
            # audio_noisy_input = audio_latent_state.latent[:, \
            #                 self.generator.kv_cache.current_audio_kv_cache_start:self.generator.kv_cache.current_audio_kv_cache_end].to(device=self.device, dtype=self.dtype)

            if block_idx == 0:  # only for the first frame ti2av
                # mask
                first_block_denoise_mask = video_latent_state.denoise_mask[:, \
                                          self.generator.kv_cache.current_video_kv_cache_start:self.generator.kv_cache.current_video_kv_cache_end].to(device=self.device, dtype=self.dtype)
                first_block_clean_latent = video_latent_state.clean_latent[:, \
                                          self.generator.kv_cache.current_video_kv_cache_start:self.generator.kv_cache.current_video_kv_cache_end].to(device=self.device, dtype=self.dtype)
                video_noisy_input = video_noisy_input * first_block_denoise_mask + first_block_clean_latent * (1 - first_block_denoise_mask)

            for sigma_idx, sigma in enumerate(self.denoising_sigmas[:-1]):
                # print(sigma)
                exit_flag = (sigma_idx == num_denoising_steps - 1)

                # update roll_kv flag
                kv_cache_snapshot['roll_kv'] = (exit_flag and block_idx >= ROLLOUT_CACHE_BLOCK_IDX)
                kv_cache_snapshot['sigma_idx'] = sigma_idx

                video_latent_model_input = video_noisy_input
                audio_latent_model_input = audio_noisy_input

                if block_idx == 0:
                    video_timesteps = sigma * first_block_denoise_mask
                    video_latent_model_input = video_latent_model_input * first_block_denoise_mask + first_block_clean_latent * (1 - first_block_denoise_mask)
                else:
                    video_timesteps = sigma * torch.ones_like(video_latent_state.denoise_mask[:, :self.generator.kv_cache.current_video_kv_cache_end-self.generator.kv_cache.current_video_kv_cache_start])
                audio_timesteps = sigma * torch.ones_like(audio_latent_state.denoise_mask[:, :self.generator.kv_cache.current_audio_kv_cache_end-self.generator.kv_cache.current_audio_kv_cache_start])

                v_pos_size = self.generator.kv_cache.current_video_kv_cache_end - self.generator.kv_cache.current_video_kv_cache_start
                a_pos_size = self.generator.kv_cache.current_audio_kv_cache_end - self.generator.kv_cache.current_audio_kv_cache_start
                video_modality = Modality(
                    latent=video_latent_model_input.to(device=self.device, dtype=self.dtype),
                    sigma=sigma.repeat(batch_size).to(device=self.device, dtype=self.dtype),
                    timesteps=video_timesteps.to(device=self.device, dtype=self.dtype),
                    positions=video_latent_state.positions[:, :, :v_pos_size, :].to(device=self.device, dtype=self.dtype),
                    context=text_context_dict['v_context'].to(device=self.device, dtype=self.dtype),
                    enabled=True,
                    context_mask=None,
                    attention_mask=None,
                )
                audio_modality = Modality(
                    latent=audio_latent_model_input.to(device=self.device, dtype=self.dtype),
                    sigma=sigma.repeat(batch_size).to(device=self.device, dtype=self.dtype),
                    timesteps=audio_timesteps.to(device=self.device, dtype=self.dtype),
                    positions=audio_latent_state.positions[:, :, :a_pos_size, :].to(device=self.device, dtype=self.dtype),
                    context=text_context_dict['a_context'].to(device=self.device, dtype=self.dtype),
                    enabled=True,
                    context_mask=None,
                    attention_mask=None,
                )

                if not exit_flag:
                    self.generator.model.eval()
                    # already input the kv_cache after kv_cache initialization
                    pred_video, pred_audio = self.generator(
                        video_modality,
                        audio_modality,
                        perturbations=None,
                        kv_cache_list=kv_cache_list,
                        kv_cache_snapshot=kv_cache_snapshot,
                    )

                    next_sigma = self.denoising_sigmas[sigma_idx+1]
                    assert next_sigma > 0
                    fresh_noise_video = torch.randn_like(pred_video)
                    fresh_noise_audio = torch.randn_like(pred_audio)

                    next_video_sigma = next_sigma * torch.ones([batch_size, pred_video.shape[1]], device=self.device)
                    next_audio_sigma = next_sigma * torch.ones([batch_size, pred_audio.shape[1]], device=self.device)

                    video_noisy_input = self.add_noise_fn(
                        pred_video.flatten(0, 1),
                        fresh_noise_video.flatten(0, 1),
                        next_video_sigma.flatten(0, 1),
                    ).unflatten(0, (batch_size, pred_video.shape[1]))

                    audio_noisy_input = self.add_noise_fn(
                        pred_audio, fresh_noise_audio, next_audio_sigma
                    )
                else:
                    self.generator.model.eval()
                    pred_video, pred_audio = self.generator(
                        video_modality,
                        audio_modality,
                        perturbations=None,
                        kv_cache_list=kv_cache_list,
                        kv_cache_snapshot=kv_cache_snapshot,
                    )
                    # only for the first frame ti2av
                    if block_idx == 0:
                        pred_video = pred_video * first_block_denoise_mask + first_block_clean_latent * (1 - first_block_denoise_mask)

                    break

            if block_idx == ROLLOUT_CACHE_BLOCK_IDX:
                self.generator.kv_cache.v_max_rope_end = self.generator.kv_cache.current_video_kv_cache_start
                self.generator.kv_cache.a_max_rope_end = self.generator.kv_cache.current_audio_kv_cache_start

            video_output.append(pred_video)
            audio_output.append(pred_audio)

        video_output = torch.cat(video_output, dim=1)
        audio_output = torch.cat(audio_output, dim=1)

        return video_output, audio_output

    def get_resolution(self, img, resolution: str):
        bucket_config_module = importlib.import_module("ltx_distillation.utils.misc")
        if resolution == '480p':
            bucket_config = getattr(bucket_config_module, 'ASPECT_RATIO_627')
        else:
            raise ValueError(f"Unknown resolution: {resolution}")
        src_h, src_w = img.height, img.width
        ratio = src_h / src_w
        closest_bucket = sorted(list(bucket_config.keys()), key=lambda x: abs(float(x)-ratio))[0]
        target_h, target_w = bucket_config[closest_bucket][0]
        return target_h, target_w

    def adapt_batch(
        self, 
        image_path, 
        prompt, 
        num_inference_frames=169, 
        resolution="480p", 
        frame_rate=24, 
        use_nospeech_prompt=False):
        torch.cuda.empty_cache()
        # self.text_encoder.to(self.device)
        # read image and resize
        img = Image.open(image_path).convert("RGB")
        height, width = self.get_resolution(img, resolution)

        # transform input image
        transform = transforms.Compose([
            transforms.Resize((height, width)),
            transforms.ToTensor(),  # PIL Image -> [C, H, W], float32, 0~1
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        img_tensor = transform(img).unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]

        # encode video stream
        with torch.no_grad():
            video_clean_latent = self.video_vae.encode(
                rearrange(img_tensor, "b f c h w -> b c f h w").to(device=self.device, dtype=self.dtype))

        torch.cuda.empty_cache()
        
        B, C, F, H, W = video_clean_latent.shape
        latent_num_frames = F
        token_per_frame = H * W
        video_clean_latent = rearrange(video_clean_latent, "b c t h w -> b (t h w) c")
        
        # only use the first frame, no regression loss
        num_frames = num_inference_frames
        latent_num_frames = num_frames//8 + 1
        seq_len = latent_num_frames * token_per_frame
        video_clean_latent = torch.cat([video_clean_latent[:, :token_per_frame], 
                                        torch.randn([video_clean_latent.shape[0], seq_len - token_per_frame, video_clean_latent.shape[2]], dtype=self.dtype, device=self.device)], dim=1)
        
        # clip audio to match video length
        video_pixel_shape = VideoPixelShape(batch=video_clean_latent.shape[0], frames=num_frames, width=width, height=height, fps=frame_rate)
        audio_latent_shape = AudioLatentShape.from_video_pixel_shape(video_pixel_shape)
        
        # only use the first frame, no regression loss
        audio_clean_latent = torch.randn([video_clean_latent.shape[0], audio_latent_shape.frames, video_clean_latent.shape[2]], dtype=self.dtype, device=self.device)
        
        # encode text prompt
        v_context_p_list = []
        a_context_p_list = []
        v_context_n_list = []
        a_context_n_list = []
        v_context_nospeech_list = []
        a_context_nospeech_list = []
        with torch.no_grad():
            ctx_p, ctx_n, ctx_nospeech = self.text_encoder(
                [prompt, DEFAULT_NEGATIVE_PROMPT, GENERAL_NOSPEECH_PROMPT],
                device=self.text_encoder_device,
            )
            v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
            v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding
            v_context_nospeech, a_context_nospeech = ctx_nospeech.video_encoding, ctx_nospeech.audio_encoding
            v_context_p_list.append(v_context_p.to(self.device))
            a_context_p_list.append(a_context_p.to(self.device))
            v_context_n_list.append(v_context_n.to(self.device))
            a_context_n_list.append(a_context_n.to(self.device))
            v_context_nospeech_list.append(v_context_nospeech.to(self.device))
            a_context_nospeech_list.append(a_context_nospeech.to(self.device))
        v_context_p = torch.cat(v_context_p_list, dim=0)
        a_context_p = torch.cat(a_context_p_list, dim=0)
        v_context_n = torch.cat(v_context_n_list, dim=0)
        a_context_n = torch.cat(a_context_n_list, dim=0)
        v_context_nospeech = torch.cat(v_context_nospeech_list, dim=0)
        a_context_nospeech = torch.cat(a_context_nospeech_list, dim=0)
        if use_nospeech_prompt:
            v_context_p = torch.cat([v_context_p, v_context_nospeech], dim=1)
            a_context_p = torch.cat([a_context_p, a_context_nospeech], dim=1)

        
        # noisy video and audio
        noisy_video_latent = torch.randn(
            video_clean_latent.shape,
            dtype=self.dtype,
            generator=self.torch_rng,
        ).to(device=self.device)
        noisy_audio_latent = torch.randn(
            audio_clean_latent.shape,
            dtype=self.dtype,
            generator=self.torch_rng,
        ).to(device=self.device)
        
        # denoise_mask
        video_denoise_mask = torch.ones(
            video_clean_latent.shape[:2]+(1,),
            device=self.device,
            dtype=torch.float32,
        )
        video_denoise_mask[:, :token_per_frame] = 0  # only the first frame is clean
        audio_denoise_mask = torch.ones(
            audio_clean_latent.shape[:2]+(1,),
            device=self.device,
            dtype=torch.float32,
        )
        
        # video_position
        components = PipelineComponents(dtype=self.dtype, device=self.device)
        video_latent_shape = VideoLatentShape.from_pixel_shape(
            shape=video_pixel_shape,
            latent_channels=components.video_latent_channels,
            scale_factors=components.video_scale_factors,
        )
        video_patchifier = components.video_patchifier
        video_latent_coords = video_patchifier.get_patch_grid_bounds(
            output_shape=video_latent_shape,
            device=self.device,
        )
        video_positions = get_pixel_coords(
            latent_coords=video_latent_coords,
            scale_factors=components.video_scale_factors,
            causal_fix=True,
        ).float()
        video_positions[:, 0, ...] = video_positions[:, 0, ...] / frame_rate

        # RoPE time coordinate manipulation (TI2V):
        # set reference image (frame 0) time RoPE to 0, shift remaining latents forward by 10s.
        # video_positions shape: (B, 3, T, 2) — dim 1 idx 0 = time axis, last dim = [start, end)
        n_patches_spatial = token_per_frame
        if n_patches_spatial < video_positions.shape[2]:
            new_time = video_positions[:, :1, :, :].clone()  # (B, 1, T, 2)
            new_time[:, :, :n_patches_spatial, :] = 0.0      # frame 0 → time = 0
            new_time[:, :, n_patches_spatial:, :] += 10.0    # frame 1+ → time += 10s
            video_positions = torch.cat(
                [new_time, video_positions[:, 1:, :, :]], dim=1
            )

        # audio_position
        audio_patchifier = components.audio_patchifier
        audio_latent_coords = audio_patchifier.get_patch_grid_bounds(
            output_shape=audio_latent_shape,
            device=self.device,
        )
        audio_positions = audio_latent_coords

        # RoPE time coordinate manipulation (TI2V, audio):
        # keep first audio token's time at 0 (aligned with reference image), shift rest by 10s.
        # audio_positions shape: (B, 1, T, 2)
        if audio_positions.shape[2] > 1:
            new_audio_time = audio_positions.clone()
            new_audio_time[:, :, 0:1, :] = 0.0   # first token → time = 0
            new_audio_time[:, :, 1:, :] += 10.0   # remaining → time += 10s
            audio_positions = new_audio_time
        
        # build video and audio state
        with torch.no_grad():
            initial_video_latent_state = LatentState(
                latent=noisy_video_latent.detach(),
                denoise_mask=video_denoise_mask.detach(),
                positions=video_positions.detach(),
                clean_latent=video_clean_latent.detach(),
                attention_mask=None,
            )
            initial_audio_latent_state = LatentState(
                latent=noisy_audio_latent.detach(),
                denoise_mask=audio_denoise_mask.detach(),
                positions=audio_positions.detach(),
                clean_latent=audio_clean_latent.detach(),
                attention_mask=None,
            )
            
            # build conditional dict
            conditional_dict = {
                "v_context": v_context_p.detach().to(self.device),
                "a_context": a_context_p.detach().to(self.device),
            }
            unconditional_dict = {
                "v_context": v_context_n.detach().to(self.device),
                "a_context": a_context_n.detach().to(self.device),
            }
        
        new_batch = {
            "conditional_dict": conditional_dict,
            "unconditional_dict": unconditional_dict,
            "initial_video_latent_state": initial_video_latent_state,
            "initial_audio_latent_state": initial_audio_latent_state,
            "video_latent_num_frames": latent_num_frames,
            "video_num_frames": num_frames,
            "width": width,
            "height": height,
        }
        # self.text_encoder.to('cpu')
        torch.cuda.empty_cache()

        return new_batch

    def write_video(self, video_pred, audio_pred, prompt, video_num_frames=153, width=768, height=512, frame_rate=24, output_dir: str = None):
        torch.cuda.empty_cache()
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        gen_output_path = os.path.join(output_dir, f"output_sample_{prompt[:20]}_{timestamp}.mp4")
        with torch.no_grad():
            decoded_video = self.video_vae.decode_to_visualize(video_pred, \
                                            num_frames=video_num_frames, width=width, height=height, frame_rate=frame_rate,
                                            use_tiling=True)
            decoded_audio = self.audio_vae.decode_to_visualize(audio_pred, \
                                            num_frames=video_num_frames, width=width, height=height, frame_rate=frame_rate)           
            save_video(video=decoded_video, audio=decoded_audio, fps=frame_rate, output_path=gen_output_path, video_chunks_number=1)
    
    def generate(
        self,
        image_path: str,
        prompt: str,
        video_num_frames: int,
        resolution: str,
        save_video: bool = False,
        output_dir: str = None,
        frame_rate=24,
    ):  
        TRAIN_NUM_FRAMES = 169
        with torch.no_grad():
            print(f"start processing image={image_path}, prompt={prompt}")
            # process data in ~ 169 frames
            info_dict = self.adapt_batch(
                image_path, 
                prompt, 
                num_inference_frames=TRAIN_NUM_FRAMES, 
                resolution=resolution,
                frame_rate=frame_rate)
            # infer data in actual latent num frames
            info_dict['video_latent_num_frames_output'] = (video_num_frames - 1) // 8 + 1
            video_pred, audio_pred = \
                self.inference_with_trajectory_inference(info_dict['initial_video_latent_state'],
                                            info_dict['initial_audio_latent_state'],
                                            video_latent_num_frames_output=info_dict['video_latent_num_frames_output'],
                                            video_latent_num_frames=info_dict['video_latent_num_frames'],
                                            text_context_dict=info_dict['conditional_dict'],)
            self.write_video(
                video_pred, 
                audio_pred, 
                prompt=prompt,
                video_num_frames=video_num_frames, 
                width=info_dict['width'], 
                height=info_dict['height'], 
                frame_rate=frame_rate,
                output_dir=output_dir,)

            del info_dict, video_pred, audio_pred
            torch.cuda.empty_cache()
