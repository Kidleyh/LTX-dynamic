from ..utils_data.prompt import clean_prompt
import random
import logging
from teleai_data_tool.schema.clip import Clip
from .. import utils_model as gm_utils
from .text_encoder import PromptEncoder
from .clip_transform import CLIPTextTransform
import torch

logger = logging.getLogger(__name__)


class PromptGenerator:
    def __init__(
        self,
        short_prompt_prob=0.5,
        default_prompt="",
        default_prompt_prob=0.2,
        clean_prompt=False,
    ) -> None:
        self.short_prompt_prob = short_prompt_prob
        self.default_prompt = default_prompt
        self.default_prompt_prob = default_prompt_prob
        self.clean_prompt = clean_prompt

    def __call__(self, data_dict):
        if data_dict["text_guidance_drop"]: # random.random() < self.default_prompt_prob:
            prompt = self.default_prompt
        else:
            clip: Clip = data_dict["clip_info"]
            if random.random() < self.short_prompt_prob:
                prompt = clip.caption.short_caption
            else:
                prompt = clip.caption.dense_caption
        if self.clean_prompt:
            prompt = clean_prompt(prompt)
            prompt = clean_prompt(prompt)
        data_dict["prompt"] = prompt
        return data_dict


class PromptToClipEmbedding:
    def __init__(self, model_path, dtype=None) -> None:
        self.clip_transform = CLIPTextTransform(
            gm_utils.get_model_path(model_path), dtype=dtype
        )

    def __call__(self, data_dict):
        prompt = data_dict["prompt"]
        clip_text_embed = self.clip_transform(
            prompt, mode="after_pool", to_numpy=False
        )[0]

        data_dict["clip_text_embed"] = clip_text_embed
        return data_dict


class PromptToTransformerEmbedding:
    """
    extract text embedding from prompts
    """

    def __init__(
        self,
        model_name,
        model_path,
        max_length=None,
        with_attention_mask=False,
        image_condition_type="token_replace",
    ):
        self.prompt_encoder = PromptEncoder(
            model_name, gm_utils.get_model_path(model_path)
        )
        self.model_name = model_name
        self.max_length = max_length
        self.image_condition_type=image_condition_type
        self.with_attention_mask = with_attention_mask

    def __call__(self, data_dict):
        prompt = data_dict["prompt"]
        if self.model_name == "llava":
            prompt_template = {
                "template": (
                    "<|start_header_id|>system<|end_header_id|>\n\n<image>\nDescribe the video by detailing the following aspects according to the reference image: "
                    "1. The main content and theme of the video."
                    "2. The color, shape, size, texture, quantity, text, and spatial relationships of the objects."
                    "3. Actions, events, behaviors temporal relationships, physical movement changes of the objects."
                    "4. background environment, light, style and atmosphere."
                    "5. camera angles, movements, and transitions used in the video:<|eot_id|>\n\n"
                    "<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>"
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                ),
                "crop_start": 103,
                "image_emb_start": 5,
                "image_emb_end": 581,
                "image_emb_len": 576,
                "double_return_token_id": 271,
            }
            image = data_dict["first_ref_image"]
            prompt = [prompt] if isinstance(prompt, str) else prompt
            prompt = [prompt_template["template"].format(p) for p in prompt]
            prompt_embeds, prompt_masks = self.prompt_encoder(
                prompt,
                image,
                max_length=(self.max_length + prompt_template.get("crop_start", 0)),
                prompt_template=prompt_template,
                with_attention_mask=self.with_attention_mask,
                image_condition_type=self.image_condition_type,
            )
        else:
            prompt_embeds, prompt_masks = self.prompt_encoder(
                prompt,
                max_length=self.max_length,
                with_attention_mask=self.with_attention_mask,
            )
        data_dict["prompt_embeds"] = prompt_embeds[0]
        data_dict["prompt_masks"] = prompt_masks[0]
        return data_dict
