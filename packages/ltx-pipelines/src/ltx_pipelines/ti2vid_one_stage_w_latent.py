import logging
from collections.abc import Iterator

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, LatentState, VideoPixelShape
from ltx_pipelines.utils import (
    ModelLedger,
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    encode_prompts,
    euler_denoising_loop,
    get_device,
    multi_modal_guider_factory_denoising_func,
)
from ltx_pipelines.utils.args import ImageConditioningInput, default_1_stage_arg_parser, detect_checkpoint_path
from ltx_pipelines.utils.constants import detect_params
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

from ltx_core.conditioning import (
    ConditioningItem,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
)

device = get_device()


class TI2VidOneStagePipeline:
    """
    Single-stage text/image-to-video generation pipeline.
    Generates video at the target resolution in a single diffusion pass with
    classifier-free guidance (CFG). Supports optional image conditioning via
    the images parameter.
    Assumes full non distilled model is provided in the checkpoint_path.
    """

    def __init__(
        self,
        checkpoint_path: str,
        gemma_root: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device = device,
        quantization: QuantizationPolicy | None = None,
    ):
        self.dtype = torch.bfloat16
        self.device = device
        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=checkpoint_path,
            gemma_root_path=gemma_root,
            loras=loras,
            quantization=quantization,
        )
        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )

    def __call__(  # noqa: PLR0913
        self,
        prompt: str,
        negative_prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        num_inference_steps: int,
        video_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        audio_guider_params: MultiModalGuiderParams | MultiModalGuiderFactory,
        images: list[ImageConditioningInput],
        enhance_prompt: bool = False,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=False)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        ctx_p, ctx_n = encode_prompts(
            [prompt, negative_prompt],
            self.model_ledger,
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
            enhance_prompt_seed=seed,
        )
        v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
        v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding

        # Encode image conditionings with the VAE encoder, then free it
        # before loading the transformer to reduce peak VRAM.
        stage_1_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        video_encoder = self.model_ledger.video_encoder()
        stage_1_conditionings = combined_image_conditionings(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )
        torch.cuda.synchronize()
        # del video_encoder
        cleanup_memory()

        transformer = self.model_ledger.transformer()
        sigmas = LTX2Scheduler().execute(steps=num_inference_steps).to(dtype=torch.float32, device=self.device)

        video_guider_factory = create_multimodal_guider_factory(
            params=video_guider_params,
            negative_context=v_context_n,
        )
        audio_guider_factory = create_multimodal_guider_factory(
            params=audio_guider_params,
            negative_context=a_context_n,
        )

        def first_stage_denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol, return_every_latent: bool = False,
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=multi_modal_guider_factory_denoising_func(
                    video_guider_factory=video_guider_factory,
                    audio_guider_factory=audio_guider_factory,
                    v_context=v_context_p,
                    a_context=a_context_p,
                    transformer=transformer,  # noqa: F821
                ),
                return_every_latent=return_every_latent,
            )

        video_state, audio_state, (video_every_latents, audio_every_latents) = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=sigmas,
            stepper=stepper,
            denoising_loop_fn=first_stage_denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            return_every_latent=True,
        )

        idxs = [0,1,2,3,4,5,12,21,26,-1]
        stored_data = {
            "video_noisy_inputs": [video_every_latents[i] for i in idxs],
            "audio_noisy_inputs": [audio_every_latents[i] for i in idxs],
            "conditional_dict": {"v_context_p": v_context_p, "a_context_p": a_context_p},
            "unconditional_dict": {"v_context_n": v_context_n, "a_context_n": a_context_n},
            "clip_context": stage_1_conditionings,}
        
        def to_cpu_detach(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu()
            elif isinstance(x, dict):
                return {k: to_cpu_detach(v) for k, v in x.items()}
            elif isinstance(x, list):
                return [to_cpu_detach(v) for v in x]
            elif isinstance(x, LatentState):
                return LatentState(
                    latent=x.latent.detach().cpu(),
                    denoise_mask=x.denoise_mask.detach().cpu(),
                    positions=x.positions.detach().cpu(),
                    clean_latent=x.clean_latent.detach().cpu(),
                    attention_mask=x.attention_mask.detach().cpu() if x.attention_mask is not None else None
                )
            elif isinstance(x, VideoConditionByLatentIndex):
                return VideoConditionByLatentIndex(
                    latent=x.latent.detach().cpu(),
                    strength=x.strength,
                    latent_idx=x.latent_idx,
                )
            else:
                return x

        stored_data_cpu = to_cpu_detach(stored_data)
        # torch.save(stored_data_cpu, "debug_data.pt")

        torch.cuda.synchronize()
        del transformer
        cleanup_memory()

        decoded_video = vae_decode_video(video_state.latent, self.model_ledger.video_decoder(), generator=generator)
        decoded_audio = vae_decode_audio(
            audio_state.latent, self.model_ledger.audio_decoder(), self.model_ledger.vocoder()
        )
        return decoded_video, decoded_audio, stored_data_cpu


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    checkpoint_path = detect_checkpoint_path()
    params = detect_params(checkpoint_path)
    parser = default_1_stage_arg_parser(params=params)
    args = parser.parse_args()
    
    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
    )
    
    import os
    import json
    import re
    import langid
    import cv2
    
    caption_paths = os.listdir("/gemini/platform/public/aigc/teleaudio/csh/javg/datasets/datasets/07_caption/part_001/")[4000:5000]
    text_with_speech_list = []
    for video_caption_path in caption_paths:
        video_caption_path = os.path.join("/gemini/platform/public/aigc/teleaudio/csh/javg/datasets/datasets/07_caption/part_001/", video_caption_path)
        with open(video_caption_path, "r", encoding="utf-8") as f:
            video_caption = json.load(f)
        text_with_speech = video_caption['audiovisual_caption']
        match_rule = r'\[.*?\]\[.*?\]:\s*"?([^"]+)"?'
        speech_text = []
        lang_list = []
        for k,v in video_caption['audio_content'].items():
            if "speech content" in k:
                match = re.search(match_rule, v)
                if match:
                    speech_content = match.group(1).strip()
                    lang, score = langid.classify(speech_content)
                    if len(speech_content) < 4:
                        lang_list.append("zh")
                    else:
                        lang_list.append(lang)
                    speech_text.append(speech_content)
                    text_with_speech = text_with_speech.replace(k,f"“{speech_content}”")
        if len(lang_list)==0 or "zh" in lang_list or 'en' in lang_list:
            text_with_speech_list.append(text_with_speech)
        else:
            continue
        
        video_path = video_caption_path.replace("07_caption", "08_fps25").replace(".json", ".mp4")
        # if not os.path.exists(os.path.join("/gemini/platform/public/aigc/human_guozz2/code/songqy/opensource_code/LTX-2/packages/images", os.path.basename(video_path).replace(".mp4", ".png"))):
        #     cap = cv2.VideoCapture(video_path)
        #     ret, frame = cap.read()   # 读取第一帧
        #     if ret:
        #         cv2.imwrite(os.path.join("/gemini/platform/public/aigc/human_guozz2/code/songqy/opensource_code/LTX-2/packages/images", os.path.basename(video_path).replace(".mp4", ".png")), frame)  # 将第一帧保存为图片
        #     cap.release()

        args.output_path = os.path.join("/gemini/platform/public/aigc/human_guozz2/code/songqy/opensource_code/LTX-2/packages/saved_video_3_true", os.path.basename(video_path))
        args.prompt = text_with_speech
        args.images = [ImageConditioningInput(os.path.join("/gemini/platform/public/aigc/human_guozz2/code/songqy/opensource_code/LTX-2/packages/images", os.path.basename(video_path).replace(".mp4", ".png")), 0, 1.0)]
        
        print(args.prompt)
        
        if os.path.exists(args.output_path):
            continue
        
        video, audio, stored_data_cpu = pipeline(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            frame_rate=args.frame_rate,
            num_inference_steps=args.num_inference_steps,
            video_guider_params=MultiModalGuiderParams(
                cfg_scale=args.video_cfg_guidance_scale,
                stg_scale=args.video_stg_guidance_scale,
                rescale_scale=args.video_rescale_scale,
                modality_scale=args.a2v_guidance_scale,
                skip_step=args.video_skip_step,
                stg_blocks=args.video_stg_blocks,
            ),
            audio_guider_params=MultiModalGuiderParams(
                cfg_scale=args.audio_cfg_guidance_scale,
                stg_scale=args.audio_stg_guidance_scale,
                rescale_scale=args.audio_rescale_scale,
                modality_scale=args.v2a_guidance_scale,
                skip_step=args.audio_skip_step,
                stg_blocks=args.audio_stg_blocks,
            ),
            images=args.images,
        )

        encode_video(
            video=video,
            fps=args.frame_rate,
            audio=audio,
            output_path=args.output_path,
            video_chunks_number=1,
        )
        torch.save(stored_data_cpu, os.path.join("/gemini/platform/public/aigc/human_guozz2/code/songqy/opensource_code/LTX-2/packages/ODE_sample_latents_3_true", os.path.basename(video_path).replace(".mp4", ".pt")))


if __name__ == "__main__":
    main()
