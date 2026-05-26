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

import torchaudio
import torch.nn.functional as F
from ltx_core.types import Audio
import datetime

class LTX23CausalAI2VInferencePipeline:
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
        )

        return kv_cache_snapshot


    ######## below are inference pipeline code ########

    @torch.no_grad()
    def inference_with_trajectory_inference(
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

        for block in tqdm(blocks, desc="Inference blocks"):
            block_idx = block.block_idx

            kv_cache_list = self.generator.kv_cache_list

            kv_cache_snapshot = self._set_kv_cache(
                blocks,
                block_idx,
                video_seqlen_frame
            )

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
                # print(sigma)
                exit_flag = (sigma_idx == num_denoising_steps - 1)
                video_latent_model_input = video_noisy_input
                audio_latent_model_input = audio_noisy_input

                if block_idx == 0:
                    video_timesteps = sigma * first_block_denoise_mask
                    video_latent_model_input = video_latent_model_input * first_block_denoise_mask + first_block_clean_latent * (1 - first_block_denoise_mask)
                else:
                    video_timesteps = sigma * torch.ones_like(video_latent_state.denoise_mask[:, self.generator.kv_cache.current_video_kv_cache_start:self.generator.kv_cache.current_video_kv_cache_end])
                audio_timesteps = sigma * torch.ones_like(audio_latent_state.denoise_mask[:, self.generator.kv_cache.current_audio_kv_cache_start:self.generator.kv_cache.current_audio_kv_cache_end])

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
                    # pred_video, pred_audio = self.generator(
                    #     video_latent_model_input=video_latent_model_input,
                    #     audio_latent_model_input=audio_latent_model_input,
                    #     sigma=sigma,
                    #     video_timesteps=video_timesteps,
                    #     audio_timesteps=audio_timesteps,
                    #     video_positions=video_latent_state.positions,
                    #     audio_positions=audio_latent_state.positions,
                    #     video_context=text_context_dict['v_context'],
                    #     audio_context=text_context_dict['a_context'],
                    #     perturbations=None,
                    #     kv_cache_list=kv_cache_list,
                    #     kv_cache_snapshot=kv_cache_snapshot,
                    # )

                    cur_block_clean_audio = audio_latent_state.clean_latent[:, block.audio_start:block.audio_end]
                    pred_audio = cur_block_clean_audio

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
                    # pred_video, pred_audio = self.generator(
                    #     video_latent_model_input=video_latent_model_input,
                    #     audio_latent_model_input=audio_latent_model_input,
                    #     sigma=sigma,
                    #     video_timesteps=video_timesteps,
                    #     audio_timesteps=audio_timesteps,
                    #     video_positions=video_latent_state.positions,
                    #     audio_positions=audio_latent_state.positions,
                    #     video_context=text_context_dict['v_context'],
                    #     audio_context=text_context_dict['a_context'],
                    #     perturbations=None,
                    #     kv_cache_list=kv_cache_list,
                    #     kv_cache_snapshot=kv_cache_snapshot,
                    # )
                    # only for the first frame ti2av
                    if block_idx == 0:
                        pred_video = pred_video * first_block_denoise_mask + first_block_clean_latent * (1 - first_block_denoise_mask)

                    cur_block_clean_audio = audio_latent_state.clean_latent[:, block.audio_start:block.audio_end]
                    pred_audio = cur_block_clean_audio

                    break

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

    def get_audio_from_wav(self, audio_path, target_sr=16000, mono=False):
        """
        直接从wav文件读取整个音频，保持与原函数一致的输出格式
        """
        # torchaudio.load 默认返回 shape 为 (C, T) 的 float32 tensor
        audio_chunk, sample_rate = torchaudio.load(audio_path)
        
        # 1. 如果采样率不匹配，进行重采样
        if sample_rate != target_sr:
            audio_chunk = torchaudio.functional.resample(audio_chunk, orig_freq=sample_rate, new_freq=target_sr)
            sample_rate = target_sr

        # 2. 声道处理逻辑 (完全对齐原函数的 numpy 逻辑，改为原生 Tensor 操作)
        if mono:
            if audio_chunk.shape[0] > 1:
                audio_chunk = audio_chunk.mean(dim=0, keepdim=True)  # 转为单声道 (1, T)
        else:
            C, T = audio_chunk.shape
            if C == 1:
                audio_chunk = audio_chunk.repeat(2, 1)  # 复制为双声道 (2, T)
            elif C == 2:
                pass  # 已经是双声道，保持不变
            else:
                mono_audio = audio_chunk.mean(dim=0, keepdim=True)   # 先转为单声道 (1, T)
                audio_chunk = mono_audio.repeat(2, 1)  # 再复制为双声道 (2, T)
                
        # 3. 音频峰值归一化 (audio normalization)
        abs_max = audio_chunk.abs().max()
        audio_chunk = audio_chunk / (abs_max + 1e-9) * 0.95
        
        # 4. Padding (保持原逻辑：在序列尾部即 T 维度上补充 44100 个 0)
        audio_chunk = F.pad(audio_chunk, (0, 44100))
        
        # 由于不需要截断，去掉了 start_idx 和 clip_length 的切片逻辑
        return audio_chunk

    def adapt_batch(
        self, 
        image_path,
        audio_path, 
        prompt, 
        num_inference_frames=169, 
        resolution="480p", 
        frame_rate=24, 
        use_nospeech_prompt=False):

        audio_chunk = self.get_audio_from_wav(audio_path)
        audio_data = Audio(audio_chunk[None].to(self.device, self.dtype), sampling_rate=16000)
        with torch.no_grad():
            audio_clean_latent = self.audio_vae.encode(audio_data.to(device=self.device))
        audio_clean_latent = rearrange(audio_clean_latent, "b c t f -> b t (c f)")

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
        # audio_clean_latent = torch.randn([video_clean_latent.shape[0], audio_latent_shape.frames, video_clean_latent.shape[2]], dtype=self.dtype, device=self.device)
        
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
        
        # audio_position
        audio_patchifier = components.audio_patchifier
        audio_latent_coords = audio_patchifier.get_patch_grid_bounds(
            output_shape=audio_latent_shape,
            device=self.device,
        )
        audio_positions = audio_latent_coords
        
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
                                            num_frames=video_num_frames, width=width, height=height, frame_rate=frame_rate)
            decoded_audio = self.audio_vae.decode_to_visualize(audio_pred, \
                                            num_frames=video_num_frames, width=width, height=height, frame_rate=frame_rate)           
            save_video(video=decoded_video, audio=decoded_audio, fps=frame_rate, output_path=gen_output_path, video_chunks_number=1)
    
    def generate(
        self,
        image_path: str,
        audio_path: str,
        prompt: str,
        video_num_frames: int,
        resolution: str,
        save_video: bool = False,
        output_dir: str = None,
        frame_rate=24,
    ):  
        with torch.no_grad():

            info_dict = self.adapt_batch(
                image_path,
                audio_path, 
                prompt, 
                num_inference_frames=video_num_frames, 
                resolution=resolution,
                frame_rate=frame_rate)
            video_pred, audio_pred = \
                self.inference_with_trajectory_inference(info_dict['initial_video_latent_state'],
                                            info_dict['initial_audio_latent_state'],
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
