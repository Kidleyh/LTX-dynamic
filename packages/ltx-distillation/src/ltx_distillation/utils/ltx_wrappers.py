import types
from typing import List, Optional
import torch
import gc


from utils.scheduler import SchedulerInterface
from ltx_pipelines.utils import ModelLedger
from ltx_core.utils import to_denoised, to_velocity
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.transformer.modality import Modality, KVCache
from ltx_core.guidance.perturbations import BatchedPerturbationConfig
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
from ltx_core.types import Audio
from ltx_pipelines.utils.media_io import encode_video as write_video
from einops import rearrange
from typing import Iterator

from ltx_core.tools import AudioLatentTools, LatentTools, VideoLatentTools
from ltx_pipelines.utils.types import PipelineComponents
from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_core.types import Audio
from ltx_core.model.audio_vae.ops import AudioProcessor


class LTXTextEncoder(torch.nn.Module):
    def __init__(self,
                 checkpoint_path="/gemini/platform/public/aigc/human_guozz2/model/LTX-2.3/ltx-2.3-22b-dev.safetensors",
                 gemma_root_path="/gemini/platform/public/aigc/human_guozz2/model/gemma-3-12b-it-qat-q4_0-unquantized/",
                 dtype=torch.bfloat16,
                 device=torch.device("cuda"),
                 ) -> None:
        super().__init__()

        self.model_ledger = ModelLedger(
            dtype=dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root_path,
            loras=(),
        )
        
        self.text_encoder = self.model_ledger.text_encoder().eval().requires_grad_(False)
        self.embeddings_processor = self.model_ledger.gemma_embeddings_processor()
        
    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str], device=None) -> dict:
        if device is None:
            device = self.device
        raw_outputs = [self.text_encoder.encode(p) for p in text_prompts]
        embeddings_processor = self.embeddings_processor
        results: list[EmbeddingsProcessorOutput] = [
            embeddings_processor.process_hidden_states(hs, mask) for hs, mask in raw_outputs
        ]
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        return results



class LTXVAEWrapper(torch.nn.Module):
    def __init__(self,
                 checkpoint_path="/gemini/platform/public/aigc/human_guozz2/model/LTX-2.3/ltx-2.3-22b-dev.safetensors",
                 dtype=torch.bfloat16,
                 device=torch.device("cuda"),
                 use_vae_encoder=True,
                 use_vae_decoder=True,
                 ):
        super().__init__()
        
        self.model_ledger = ModelLedger(
            dtype=dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            loras=(),
        )
        # init model
        if use_vae_encoder:
            self.video_vae_encoder = self.model_ledger.video_encoder().eval().requires_grad_(False)
            self.audio_vae_encoder = self.model_ledger.audio_encoder().eval().requires_grad_(False)
        if use_vae_decoder:
            self.video_vae_decoder = self.model_ledger.video_decoder().eval().requires_grad_(False)
            self.audio_vae_decoder = self.model_ledger.audio_decoder().eval().requires_grad_(False)
            self.audio_vocoder = self.model_ledger.vocoder().eval().requires_grad_(False)
        self.dtype = dtype
        self.device = device
        

    def encode_video(self, pixel: torch.Tensor) -> torch.Tensor:
        video_latent = self.video_vae_encoder(pixel)
        return video_latent
    
    def encode_audio(self, audio: Audio) -> torch.Tensor:
        dtype = next(self.audio_vae_encoder.parameters()).dtype
        device = next(self.audio_vae_encoder.parameters()).device
        audio_processor = AudioProcessor(
            target_sample_rate=self.audio_vae_encoder.sample_rate,
            mel_bins=self.audio_vae_encoder.mel_bins,
            mel_hop_length=self.audio_vae_encoder.mel_hop_length,
            n_fft=self.audio_vae_encoder.n_fft,
        ).to(device=device)
        mel_spectrogram = audio_processor.waveform_to_mel(audio.to(device=device))
        latent = self.audio_vae_encoder(mel_spectrogram.to(dtype=dtype))
        return latent

    def decode_video(self, latent_video: torch.Tensor, tiling_config=None) -> Iterator[torch.Tensor]:
        def convert_to_uint8(frames: torch.Tensor) -> torch.Tensor:
            frames = (((frames + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
            frames = rearrange(frames[0], "c f h w -> f h w c")
            return frames

        if tiling_config is not None:
            for frames in self.video_vae_decoder.tiled_decode(latent_video, tiling_config):
                yield convert_to_uint8(frames)
        else:
            decoded_video = self.video_vae_decoder(latent_video)
            yield convert_to_uint8(decoded_video)

    def decode_audio(self, latent_audio: torch.Tensor) -> Audio:
        decoded_audio = self.audio_vae_decoder(latent_audio)
        output_audio = self.audio_vocoder(decoded_audio).squeeze(0).float()
        return Audio(waveform=output_audio, sampling_rate=self.audio_vocoder.output_sampling_rate)
    
    def write_video(self, video, audio, fps=24, output_path="output.mp4", video_chunks_number=1):
        return write_video(video=video, audio=audio, fps=fps, output_path=output_path, video_chunks_number=video_chunks_number)
    
    def forward(self, video_latent, audio_latent, output_path="output.mp4", num_frames=137, width=768, height=512, frame_rate=24):
        components = PipelineComponents(dtype=self.dtype, device=self.device)
        video_pixel_shape = VideoPixelShape(batch=video_latent.shape[0], frames=num_frames, width=width, height=height, fps=frame_rate)
        video_latent_shape = VideoLatentShape.from_pixel_shape(
            shape=video_pixel_shape,
            latent_channels=components.video_latent_channels,
            scale_factors=components.video_scale_factors,
        )
        video_tools = VideoLatentTools(components.video_patchifier, video_latent_shape, frame_rate)
        video_latent = video_tools.patchifier.unpatchify(video_latent, video_latent_shape)
        decoded_video = self.decode_video(video_latent)
        
        audio_latent_shape = AudioLatentShape.from_video_pixel_shape(video_pixel_shape)
        audio_tools = AudioLatentTools(components.audio_patchifier, audio_latent_shape)
        audio_latent = audio_tools.patchifier.unpatchify(audio_latent, audio_latent_shape)
        decoded_audio = self.decode_audio(audio_latent)
        return self.write_video(decoded_video, decoded_audio, output_path=output_path, fps=frame_rate)
        


class DiffusionLTXWrapper(torch.nn.Module):
    def __init__(
            self,
            is_causal=False,
            max_token_size=-1,
            checkpoint_path="/gemini/platform/public/aigc/human_guozz2/model/LTX-2.3/ltx-2.3-22b-dev.safetensors",
            is_train=True,
            dtype=torch.bfloat16,
            device=torch.device("cuda"),
            is_sf=False,
    ):
        super().__init__()
        
        self.model_ledger = ModelLedger(
            dtype=dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            loras=(),
        )
        
        if is_sf:
            self.model = self.model_ledger.transformer().velocity_model
        elif is_causal:
            self.model = self.model_ledger.causal_transformer().velocity_model
        else:
            self.model = self.model_ledger.transformer().velocity_model

        self.model.eval()

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal

        self.scheduler =  LTX2Scheduler()
        self.sigmas = self.scheduler.execute(steps=30).to(dtype=torch.float32)

        self.seq_len = max_token_size # [NOTE]: for 480p training resolution # 32760  # [1, 21, 16, 60, 104]
        self.post_init()

        self.num_transformer_blocks = len(self.model.transformer_blocks)  # 48
        self.num_attention_heads = self.model.num_attention_heads
        self.num_frame_per_block = 2  # maybe change to 3
        self.kv_cache = None

    def enable_gradient_checkpointing(self) -> None:
        self.model.set_gradient_checkpointing(enable=True)

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        AssertionError("not support cls branch for LTX wrapper.")

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor, sigma: float|torch.Tensor) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        sigma: the noise level with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        """
        
        x0_pred = to_denoised(sample=xt, velocity=flow_pred, sigma=sigma, calc_dtype=flow_pred.dtype)
        return x0_pred

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, sigma: float|torch.Tensor) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        sigma: the noise level with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        
        flow_pred = to_velocity(sample=xt, denoised_sample=x0_pred, sigma=sigma, calc_dtype=x0_pred.dtype)
        return flow_pred


    def forward(
        self,
        video: Modality | None,
        audio: Modality | None,
        perturbations: BatchedPerturbationConfig = None,
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

        # X0 prediction
        flow_pred_v, flow_pred_a = self.model(
            video,
            audio,
            perturbations,
            kv_cache=self.kv_cache,
        )

        # flow_pred video and audio are both velocity prediction, we need to convert them to x0 prediction for loss calculation and scheduler step.
        pred_x0_v = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred_v,
            xt=video.latent,
            sigma=video.timesteps
        )
        pred_x0_a = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred_a,
            xt=audio.latent,
            sigma=audio.timesteps
        )

        return flow_pred_v, pred_x0_v, flow_pred_a, pred_x0_a

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
        video_self_attn_kv_cache = []
        video_cross_attn_kv_cache = []
        audio_self_attn_kv_cache = []
        audio_cross_attn_kv_cache = []
        a2v_cross_attn_kv_cache = []
        v2a_cross_attn_kv_cache = []
        # Use the default KV cache size
        # kv_cache_size = self.num_frame_per_block * (32*32)
        for _ in range(self.num_transformer_blocks):
            video_self_attn_kv_cache.append({
                    "k": torch.zeros((batch_size, kv_cache_size['video_self_attn_kv_cache_size'], 4096), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, kv_cache_size['video_self_attn_kv_cache_size'], 4096), dtype=dtype, device=device),
                   })
            video_cross_attn_kv_cache.append({
                    "k": torch.zeros((batch_size, kv_cache_size['video_cross_attn_kv_cache_size'], 4096), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, kv_cache_size['video_cross_attn_kv_cache_size'], 4096), dtype=dtype, device=device),
                   })
            audio_self_attn_kv_cache.append({
                    "k": torch.zeros((batch_size, kv_cache_size['audio_self_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, kv_cache_size['audio_self_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
                   })
            audio_cross_attn_kv_cache.append({
                    "k": torch.zeros((batch_size, kv_cache_size['audio_cross_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, kv_cache_size['audio_cross_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
                   })
            a2v_cross_attn_kv_cache.append({
                    "k": torch.zeros((batch_size, kv_cache_size['a2v_cross_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, kv_cache_size['a2v_cross_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
                   })
            v2a_cross_attn_kv_cache.append({
                    "k": torch.zeros((batch_size, kv_cache_size['v2a_cross_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
                    "v": torch.zeros((batch_size, kv_cache_size['v2a_cross_attn_kv_cache_size'], 2048), dtype=dtype, device=device),
                   })
        
        # always store the clean cache
        self.kv_cache.video_self_attn_kv_cache = video_self_attn_kv_cache
        self.kv_cache.video_cross_attn_kv_cache = video_cross_attn_kv_cache 
        self.kv_cache.audio_self_attn_kv_cache = audio_self_attn_kv_cache 
        self.kv_cache.audio_cross_attn_kv_cache = audio_cross_attn_kv_cache 
        self.kv_cache.a2v_cross_attn_kv_cache = a2v_cross_attn_kv_cache 
        self.kv_cache.v2a_cross_attn_kv_cache = v2a_cross_attn_kv_cache
        self.kv_cache.current_video_kv_cache_start = 0
        self.kv_cache.current_audio_kv_cache_start = 0
        self.kv_cache.current_transformer_block_index = None

