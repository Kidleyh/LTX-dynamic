"""
VAE Wrappers for visualization and validation during DMD distillation.
"""

from typing import Optional
import torch
import torch.nn as nn
from ltx_trainer.model_loader import (load_video_vae_decoder, 
                                      load_audio_vae_decoder, 
                                      load_vocoder, 
                                      load_video_vae_encoder,
                                      load_audio_vae_encoder)   

from ltx_core.types import Audio
from ltx_core.model.audio_vae.ops import AudioProcessor



from ltx_core.tools import AudioLatentTools, LatentTools, VideoLatentTools
from ltx_pipelines.utils.types import PipelineComponents
from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_core.types import Audio
from ltx_core.model.audio_vae.ops import AudioProcessor
from ltx_core.model.video_vae.tiling import TilingConfig

from einops import rearrange



class VideoVAEWrapper(nn.Module):
    """
    Wrapper for Video VAE encoder and decoder.

    Used for:
    - Encoding videos to latent space (for visualization)
    - Decoding latents to pixel space (for validation)
    """

    def __init__(
        self,
        encoder=None,
        decoder=None,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            encoder: VideoEncoder instance (optional)
            decoder: VideoDecoder instance
            device: Target device
            dtype: Model dtype
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode(self, video: torch.Tensor) -> torch.Tensor:
        """
        Encode video to latent space.

        Args:
            video: Pixel video [B, C, F, H, W] in range [-1, 1]

        Returns:
            Latent [B, F', C_latent, H', W']
        """
        if self.encoder is None:
            raise ValueError("Encoder not initialized")

        return self.encoder(video)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor, tiling_config=None) -> torch.Tensor:
        """
        Decode latent to pixel space.

        Args:
            latent: Latent [B, F, C, H, W]

        Returns:
            Video [B, C, F_out, H_out, W_out] in range [-1, 1]
        """
        if self.decoder is None:
            raise ValueError("Decoder not initialized")

        # Decoder expects [B, C, F, H, W].
        # Our DMD code stores video as [B, F, C, H, W] where C=128.
        # Detect this by checking if dim 2 (not dim 1) equals 128.
        if latent.dim() == 5 and latent.shape[2] == 128:
            # Input is [B, F, C, H, W], need to permute to [B, C, F, H, W]
            latent = latent.permute(0, 2, 1, 3, 4)

        if tiling_config is not None:
            frames_all = []
            for frames in self.decoder.tiled_decode(latent, tiling_config):
                frames_all.append(frames.cpu())
            return torch.cat(frames_all, dim=2)
        else:
            return self.decoder(latent)

        # return self.decoder(latent)

    def convert_to_uint8(self, frames: torch.Tensor) -> torch.Tensor:
        frames = (((frames + 1.0) / 2.0).clamp(0.0, 1.0) * 255.0).to(torch.uint8)
        frames = rearrange(frames[0], "c f h w -> f h w c")
        return frames

    @torch.no_grad()
    def decode_to_visualize(self, latent, num_frames=137, width=768, height=512, frame_rate=24, use_tiling=False, tiling_config=None):
        components = PipelineComponents(dtype=self.dtype, device=self.device)
        video_pixel_shape = VideoPixelShape(batch=latent.shape[0], frames=num_frames, width=width, height=height, fps=frame_rate)
        video_latent_shape = VideoLatentShape.from_pixel_shape(
            shape=video_pixel_shape,
            latent_channels=components.video_latent_channels,
            scale_factors=components.video_scale_factors,
        )
        video_tools = VideoLatentTools(components.video_patchifier, video_latent_shape, frame_rate)
        video_latent = video_tools.patchifier.unpatchify(latent, video_latent_shape)
        if use_tiling:
            if tiling_config is None:
                tiling_config = TilingConfig.default()
        else:
            tiling_config = None
        decoded_video = self.decode(video_latent, tiling_config=tiling_config)
        decoded_video = self.convert_to_uint8(decoded_video)
        return decoded_video


    @torch.no_grad()
    def decode_to_pixel(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent to pixel video for visualization.

        Args:
            latent: Latent [B, F, C, H, W]

        Returns:
            Video frames suitable for logging (normalized to [0, 1])
        """
        video = self.decode(latent)
        # Normalize from [-1, 1] to [0, 1]
        video = (video + 1) / 2
        video = video.clamp(0, 1)
        return video


class AudioVAEWrapper(nn.Module):
    """
    Wrapper for Audio VAE decoder and vocoder.

    Used for:
    - Decoding audio latents to mel spectrogram
    - Converting mel to waveform via vocoder
    """

    def __init__(
        self,
        encoder=None,
        decoder=None,
        vocoder=None,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            decoder: AudioDecoder instance
            vocoder: Vocoder instance
            device: Target device
            dtype: Model dtype
        """
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocoder = vocoder
        self.device = device
        self.dtype = dtype

    @torch.no_grad()
    def encode(self, audio: Audio) -> torch.Tensor:
        dtype = next(self.encoder.parameters()).dtype
        device = next(self.encoder.parameters()).device
        audio_processor = AudioProcessor(
            target_sample_rate=self.encoder.sample_rate,
            mel_bins=self.encoder.mel_bins,
            mel_hop_length=self.encoder.mel_hop_length,
            n_fft=self.encoder.n_fft,
        ).to(device=device)
        mel_spectrogram = audio_processor.waveform_to_mel(audio.to(device=device))
        latent = self.encoder(mel_spectrogram.to(dtype=dtype))
        return latent

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode audio latent to mel spectrogram.

        The DMD pipeline produces audio latents in the transformer's sequence
        format ``[B, T, C*F]`` (3D), but the ``AudioDecoder`` expects the VAE
        spatial format ``[B, C, T, F]`` (4D).  This method handles the
        conversion automatically using the decoder's ``z_channels`` and
        ``mel_bins`` attributes (set during checkpoint loading).

        Args:
            latent: Audio latent, either ``[B, T, C*F]`` (transformer) or
                    ``[B, C, T, F]`` (VAE).

        Returns:
            Mel spectrogram ``[B, out_ch, time, freq]``.
        """
        if self.decoder is None:
            raise ValueError("Decoder not initialized")

        # Reshape 3D transformer latent → 4D VAE latent when necessary.
        # The transformer stores audio as [B, T, C*F] where C=z_channels and
        # F=latent_mel_bins.  The AudioDecoder expects [B, C, T, F].
        # Note: decoder.mel_bins is the *output* spectrogram size (e.g. 64),
        # NOT the latent mel dimension.  The latent mel dim = CF // z_channels.
        if latent.dim() == 3:
            B, T, CF = latent.shape
            z_channels = getattr(self.decoder, "z_channels", None)

            if z_channels is not None:
                latent_mel = CF // z_channels  # e.g. 128 // 8 = 16
                # "b t (c f) -> b c t f"
                latent = latent.reshape(B, T, z_channels, latent_mel).permute(0, 2, 1, 3)
            else:
                raise ValueError(
                    f"Cannot reshape 3D audio latent {latent.shape} to 4D: "
                    "decoder is missing z_channels attribute."
                )

        return self.decoder(latent)

    @torch.no_grad()
    def decode_audio(self, latent_audio: torch.Tensor) -> Audio:
        decoded_audio = self.decoder(latent_audio)
        output_audio = self.vocoder(decoded_audio).squeeze(0).float()
        return Audio(waveform=output_audio, sampling_rate=self.vocoder.output_sampling_rate)
    

    @torch.no_grad()
    def decode_to_visualize(self, latent: torch.Tensor, num_frames: int, width: int, height: int, frame_rate: int):

        # audio_latent_shape = AudioLatentShape.from_video_pixel_shape(video_pixel_shape)
        # audio_tools = AudioLatentTools(components.audio_patchifier, audio_latent_shape)
        # audio_latent = audio_tools.patchifier.unpatchify(audio_latent, audio_latent_shape)

        # output_audio = self.decode_to_waveform(latent)
        # return Audio(waveform=output_audio, sampling_rate=self.vocoder.output_sampling_rate)

        components = PipelineComponents(dtype=self.dtype, device=self.device)
        video_pixel_shape = VideoPixelShape(batch=latent.shape[0], frames=num_frames, width=width, height=height, fps=frame_rate)
        video_latent_shape = VideoLatentShape.from_pixel_shape(
            shape=video_pixel_shape,
            latent_channels=components.video_latent_channels,
            scale_factors=components.video_scale_factors,
        )
        audio_latent_shape = AudioLatentShape.from_video_pixel_shape(video_pixel_shape)
        audio_tools = AudioLatentTools(components.audio_patchifier, audio_latent_shape)
        latent = audio_tools.patchifier.unpatchify(latent, audio_latent_shape)
        decoded_audio = self.decode_audio(latent)
        return decoded_audio

    # def write_video(self, video, audio, fps=24, output_path="output.mp4", video_chunks_number=1):
    #     return write_video(video=video, audio=audio, fps=fps, output_path=output_path, video_chunks_number=video_chunks_number)
    
    # def visualize(self, video_latent, audio_latent, output_path="output.mp4", num_frames=137, width=768, height=512, frame_rate=24, use_tiling=True, tiling_config=None):

    #     if use_tiling:
    #         if tiling_config is None:
    #             tiling_config = TilingConfig.default()
    #     else:
    #         tiling_config = None
    #     decoded_video = self.decode_video(video_latent, tiling_config=tiling_config)
        
    #     audio_latent_shape = AudioLatentShape.from_video_pixel_shape(video_pixel_shape)
    #     audio_tools = AudioLatentTools(components.audio_patchifier, audio_latent_shape)
    #     audio_latent = audio_tools.patchifier.unpatchify(audio_latent, audio_latent_shape)
    #     decoded_audio = self.decode_audio(audio_latent)
    #     return self.write_video(decoded_video, decoded_audio, output_path=output_path, fps=frame_rate)

    @torch.no_grad()
    def decode_to_waveform(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode audio latent to waveform.

        Args:
            latent: Audio latent [B, F, C]

        Returns:
            Waveform [B, 1, samples]
        """
        mel = self.decode(latent)

        if self.vocoder is None:
            raise ValueError("Vocoder not initialized")

        # Cast to float32 after vocoder to match the original LTX-2 pipeline's
        # decode_audio() behavior (audio_vae.py:479). The vocoder's 240x upsampling
        # chain amplifies bfloat16 quantization errors into audible high-frequency
        # noise; float32 output prevents this.
        return self.vocoder(mel).float()

def create_vae_wrappers(
    checkpoint_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    registry = None,
) -> tuple[VideoVAEWrapper, AudioVAEWrapper]:
    """
    Factory function to create VAE wrappers from checkpoint.

    Args:
        checkpoint_path: Path to LTX-2 checkpoint
        device: Target device
        dtype: Model dtype

    Returns:
        Tuple of (VideoVAEWrapper, AudioVAEWrapper)
    """
    # from ltx_pipelines.utils.model_ledger import ModelLedger

    # Load to CPU first to avoid safetensors device issues
    # ledger = ModelLedger(
    #     dtype=dtype,
    #     device=torch.device("cpu"),
    #     checkpoint_path=checkpoint_path,
    # )

    video_encoder = load_video_vae_encoder(checkpoint_path, device, dtype)
    audio_encoder = load_audio_vae_encoder(checkpoint_path, device, dtype)
    video_decoder = load_video_vae_decoder(checkpoint_path, device, dtype)
    audio_decoder = load_audio_vae_decoder(checkpoint_path, device, dtype)
    vocoder = load_vocoder(checkpoint_path, device, dtype)

    # video_decoder = ledger.video_decoder()
    # audio_decoder = ledger.audio_decoder()
    # vocoder = ledger.vocoder()

    # Move to target device
    video_encoder = video_encoder.to(device=device, dtype=dtype)
    audio_encoder = audio_encoder.to(device=device, dtype=dtype)
    video_decoder = video_decoder.to(device=device, dtype=dtype)
    audio_decoder = audio_decoder.to(device=device, dtype=dtype)
    vocoder = vocoder.to(device=device, dtype=dtype)

    video_vae = VideoVAEWrapper(
        encoder=video_encoder,
        decoder=video_decoder,
        device=device,
        dtype=dtype,
    )

    audio_vae = AudioVAEWrapper(
        encoder=audio_encoder,
        decoder=audio_decoder,
        vocoder=vocoder,
        device=device,
        dtype=dtype,
    )

    return video_vae, audio_vae
