"""
Gemma Text Encoder Wrapper for DMD distillation.

Provides a simple interface for text encoding without prompt enhancement.
Just pure text -> context embedding conversion.
"""

from typing import List, Dict, Any, Optional
import torch
import torch.nn as nn
from ltx_trainer.model_loader import load_text_encoder, load_embeddings_processor
from ltx_core.text_encoders.gemma.embeddings_processor import EmbeddingsProcessorOutput
import gc

class GemmaTextEncoderWrapper(nn.Module):
    """
    Wrapper for Gemma text encoder to provide DMD-compatible interface.

    This wrapper:
    - Takes raw text prompts (no enhancement needed)
    - Returns conditional_dict with video_context and audio_context
    - Handles batched encoding
    """

    def __init__(
        self,
        text_encoder,
        embeddings_processor,
        device: torch.device = None,
        dtype: torch.dtype = torch.bfloat16,
    ):
        """
        Args:
            text_encoder: AVGemmaTextEncoderModel instance
            device: Target device
            dtype: Model dtype
        """
        super().__init__()
        self.text_encoder = text_encoder
        self.embeddings_processor = embeddings_processor
        self.device = device
        self.dtype = dtype


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

    # @torch.no_grad()
    # def forward(
    #     self,
    #     text_prompts: List[str],
    #     padding_side: str = "left",
    # ) -> Dict[str, torch.Tensor]:
    #     """
    #     Encode text prompts to conditioning embeddings.

    #     Args:
    #         text_prompts: List of text prompts (already processed, no enhancement)
    #         padding_side: Padding side for tokenizer

    #     Returns:
    #         Dictionary containing:
    #             - video_context: [B, seq_len, dim] video conditioning
    #             - audio_context: [B, seq_len, dim] audio conditioning
    #             - attention_mask: [B, seq_len] attention mask
    #     """
    #     batch_size = len(text_prompts)

    #     # Encode each prompt
    #     video_contexts = []
    #     audio_contexts = []
    #     attention_masks = []

    #     for prompt in text_prompts:
    #         # Forward through text encoder
    #         output = self.text_encoder(text=prompt, padding_side=padding_side)

    #         video_contexts.append(output.video_encoding)
    #         audio_contexts.append(output.audio_encoding)
    #         attention_masks.append(output.attention_mask)

    #     # Stack batch
    #     video_context = torch.cat(video_contexts, dim=0)
    #     audio_context = torch.cat(audio_contexts, dim=0)
    #     attention_mask = torch.cat(attention_masks, dim=0)

    #     return {
    #         "video_context": video_context,
    #         "audio_context": audio_context,
    #         "attention_mask": attention_mask,
    #     }

    def encode_batch(
        self,
        text_prompts: List[str],
    ) -> Dict[str, torch.Tensor]:
        """Alias for forward() with default padding."""
        return self.forward(text_prompts)


def create_text_encoder_wrapper(
    checkpoint_path: str,
    gemma_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
    load_in_8bit: bool = False,
    registry = None,
) -> GemmaTextEncoderWrapper:
    """
    Factory function to create GemmaTextEncoderWrapper from checkpoint.

    Args:
        checkpoint_path: Path to LTX-2 checkpoint
        gemma_path: Path to Gemma text encoder
        device: Target device
        dtype: Model dtype

    Returns:
        Configured GemmaTextEncoderWrapper
    """
    # from ltx_pipelines.utils.model_ledger import ModelLedger

    # Load to CPU first to avoid safetensors device issues
    # ledger = ModelLedger(
    #     dtype=dtype,
    #     device=torch.device("cpu"),
    #     checkpoint_path=checkpoint_path,
    #     gemma_root_path=gemma_path,
    # )

    # text_encoder = ledger.text_encoder()

    text_encoder = load_text_encoder(
        gemma_path,
        device,
        dtype,
        load_in_8bit=load_in_8bit,
    )
    embeddings_processor = load_embeddings_processor(
        checkpoint_path,
        device,
        dtype,
    )

    text_encoder = text_encoder.to(device=device, dtype=dtype)

    wrapper = GemmaTextEncoderWrapper(
        text_encoder=text_encoder,
        embeddings_processor=embeddings_processor,
        device=device,
        dtype=dtype,
    )

    return wrapper
