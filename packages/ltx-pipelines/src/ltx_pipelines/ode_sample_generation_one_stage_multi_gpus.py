import logging
import os
from collections.abc import Iterator

import torch
import torch.distributed as dist  # [新增] 用于多卡并行
import json
import re
import langid
import cv2

from ltx_core.components.guiders import (
    MultiModalGuiderFactory,
    MultiModalGuiderParams,
    create_multimodal_guider_factory,
)
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import Registry
from ltx_core.model.video_vae.tiling import TilingConfig
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio
from ltx_pipelines.utils import (
    assert_resolution,
    combined_image_conditionings,
    get_device,
)
from ltx_core.types import Audio, LatentState, VideoPixelShape
from ltx_pipelines.utils.args import ImageConditioningInput, default_1_stage_arg_parser, detect_checkpoint_path
from ltx_pipelines.utils.blocks import (
    AudioDecoder,
    DiffusionStage,
    ImageConditioner,
    PromptEncoder,
    VideoDecoder,
)
from ltx_pipelines.utils.constants import detect_params
from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import ModalitySpec

from ltx_core.conditioning import (
    ConditioningItem,
    VideoConditionByKeyframeIndex,
    VideoConditionByLatentIndex,
)

from ltx_distillation.models.text_encoder_wrapper import GemmaTextEncoderWrapper, create_text_encoder_wrapper

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
        device: torch.device | None = None,
        text_encoder_device: torch.device | None = None, # <--- 确保参数在这里
        quantization: QuantizationPolicy | None = None,
        registry: Registry | None = None,
        torch_compile: bool = False,
        trained_ckpt_path: str | None = None,
        sigmas: list[float] | None = None,
    ):
        self.dtype = torch.bfloat16
        self.device = device or get_device()
        self.text_encoder_device = text_encoder_device or self.device # <--- 保存 text_encoder 的专属设备
        
        self.prompt_encoder = create_text_encoder_wrapper(
            checkpoint_path=checkpoint_path,
            gemma_path=gemma_root,
            device=self.text_encoder_device, # <--- 部署在指定的文本编码卡上
            dtype=self.dtype,
            load_in_8bit=False,
            registry=None,
        )

        self._scheduler = LTX2Scheduler()

        self.image_conditioner = ImageConditioner(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
        )
        self.stage = DiffusionStage(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            loras=tuple(loras),
            quantization=quantization,
            registry=registry,
            torch_compile=torch_compile,
        )
        self.video_decoder = VideoDecoder(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
        )
        self.audio_decoder = AudioDecoder(
            checkpoint_path=checkpoint_path,
            dtype=self.dtype,
            device=self.device,
            registry=registry,
        )
        self.trained_ckpt_path = trained_ckpt_path
        self.sigmas = sigmas # torch.tensor(sigmas, dtype=self.dtype, device=self.device)

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
        streaming_prefetch_count: int | None = None,
        tiling_config: TilingConfig | None = None,
        max_batch_size: int = 1,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        assert_resolution(height=height, width=width, is_two_stage=False)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        dtype = torch.bfloat16

        with torch.no_grad():
            ctx_p, ctx_n = self.prompt_encoder(
                [prompt, negative_prompt],
                device=self.text_encoder_device,
            )

        # ================= 修改：跨卡 Tensor 转移 =================
        # 文本编码器出来的特征在 text_encoder_device 上，需要 .to(self.device) 才能给 self.stage 使用
        # 使用安全保护机制，防止某些情况下 audio_encoding 为 None 导致报错
        v_context_p = ctx_p.video_encoding.to(self.device) if ctx_p.video_encoding is not None else None
        a_context_p = ctx_p.audio_encoding.to(self.device) if ctx_p.audio_encoding is not None else None
        v_context_n = ctx_n.video_encoding.to(self.device) if ctx_n.video_encoding is not None else None
        a_context_n = ctx_n.audio_encoding.to(self.device) if ctx_n.audio_encoding is not None else None
        # =========================================================

        stage_1_conditionings = self.image_conditioner(
            lambda enc: combined_image_conditionings(
                images=images,
                height=height,
                width=width,
                video_encoder=enc,
                dtype=dtype,
                device=self.device,
            )
        )

        sigmas = (torch.tensor(self.sigmas, dtype=self.dtype, device=self.device) if len(self.sigmas) > 0 else self._scheduler.execute(steps=num_inference_steps)).to(
            dtype=torch.float32, device=self.device
        )
        video_guider_factory = create_multimodal_guider_factory(
            params=video_guider_params,
            negative_context=v_context_n,
        )
        audio_guider_factory = create_multimodal_guider_factory(
            params=audio_guider_params,
            negative_context=a_context_n,
        )

        video_state, audio_state, (video_every_latents, audio_every_latents) = self.stage(
            denoiser=FactoryGuidedDenoiser(
                v_context=v_context_p,
                a_context=a_context_p,
                video_guider_factory=video_guider_factory,
                audio_guider_factory=audio_guider_factory,
            ),
            sigmas=sigmas,
            noiser=noiser,
            width=width,
            height=height,
            frames=num_frames,
            fps=frame_rate,
            video=ModalitySpec(
                context=v_context_p,
                conditionings=stage_1_conditionings,
            ),
            audio=ModalitySpec(
                context=a_context_p,
            ),
            streaming_prefetch_count=streaming_prefetch_count,
            max_batch_size=max_batch_size,
            gen_ode=True,
            trained_ckpt_path=self.trained_ckpt_path,
        )

        # idxs = [0,1,2,3]
        # num_inference_steps==30: 
        # 0 -> 1  
        # 12 -> 0.9098  
        # 17 -> 0.8356
        # 20 -> 0.7664
        # 21 -> 0.7364
        # 22 -> 0.7017
        # 25 -> 0.5531
        # 26 -> 0.4802
        # 27 -> 0.3875
        # 28 -> 0.2661
        # 30 -> 0.0000
        idxs = [0,12,20,22,25,26,30] # 0~1 2~0.7664 5~0.4802 6~0  # 0 1 4 7 -> 1000 910 736 480
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

        decoded_video = self.video_decoder(video_state.latent, tiling_config, generator=generator)
        decoded_audio = self.audio_decoder(audio_state.latent)
        # 注意: 如果这里返回的是三个值 (包括 stored_data_cpu), 请在此处增加返回, 保持和你原来脚本的解包一致

        return decoded_video, decoded_audio, stored_data_cpu # 这里做了一个兼容原脚本的假设，返回潜变量以备保存


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    
    # ================= 修改：多卡分布式环境初始化 =================
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    
    # 【核心逻辑】：每个进程占用 2 张卡
    # 假设你的一台机器有 8 张卡，你需要使用 torchrun --nproc_per_node=4 启动
    # local_rank 0 -> 使用 cuda:0 (stage) 和 cuda:1 (text_encoder)
    # local_rank 1 -> 使用 cuda:2 (stage) 和 cuda:3 (text_encoder)
    device = torch.device(f"cuda:{local_rank * 2}") 
    text_encoder_device = torch.device(f"cuda:{local_rank * 2 + 1}")
    
    # 设置主卡为 stage 所在的卡，方便后续默认张量分配
    torch.cuda.set_device(device)
    # ==============================================================

    checkpoint_path = detect_checkpoint_path()
    params = detect_params(checkpoint_path)
    parser = default_1_stage_arg_parser(params=params)

    # ================= 新增：将硬编码路径暴露为命令行参数 =================
    parser.add_argument("--trained_ckpt_path", type=str, default=None) # "/gemini/platform/public/aigc/human_guozz2/code/hys/LTX-2/ltx_experiments/0415_ltx_node_bs21_bidmd_smalllr/checkpoint_model_005000/model_gen.pt"
    parser.add_argument("--images_path", type=str, default="/gemini/platform/public/aigc/human_guozz2/code/songqy/opensource_code/LTX-2/packages/images")
    parser.add_argument("--caption_path", type=str, default="/gemini/platform/public/aigc/teleaudio/csh/javg/datasets/datasets/07_caption/part_001/")
    parser.add_argument("--video_save_path", type=str, default="/gemini/platform/public/aigc/human_guozz2/code/hys/LTX-2/ode_data/0507_multisteps_ode/videos")
    parser.add_argument("--ode_save_path", type=str, default="/gemini/platform/public/aigc/human_guozz2/code/hys/LTX-2/ode_data/0507_multisteps_ode/ode_samples")
    parser.add_argument("--start_idx", type=int, default=0, help="处理文件的起始索引")
    parser.add_argument("--end_idx", type=int, default=-1, help="处理文件的结束索引")
    parser.add_argument(
            '--sigmas', 
            type=float, 
            nargs='+', 
            help='输入一个或多个浮点数，用空格隔开',
            default=[] # 建议加上默认值
        )
    # ======================================================================

    args = parser.parse_args()
    
    os.makedirs(args.video_save_path, exist_ok=True)
    os.makedirs(args.ode_save_path, exist_ok=True)

    # 实例化 pipeline 时传入两张不同的卡
    pipeline = TI2VidOneStagePipeline(
        checkpoint_path=args.checkpoint_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
        torch_compile=args.compile,
        device=device,                           # <--- 传给 stage 和 decoder
        text_encoder_device=text_encoder_device, # <--- 传给 prompt_encoder
        trained_ckpt_path=args.trained_ckpt_path,
        sigmas=args.sigmas,
    )

    # 获取所有排序好的文件列表，保证所有卡看到的列表一致
    all_caption_files = sorted(os.listdir(args.caption_path))
    print(f"Total files: {len(all_caption_files)}, using {args.start_idx} to {args.end_idx}")
    all_caption_files = all_caption_files[args.start_idx:args.end_idx]
    
    # ================= 新增：将任务切片分配给不同的显卡 =================
    my_caption_files = all_caption_files[rank::world_size]
    logging.info(f"[Rank {rank}/{world_size}] Assigned {len(my_caption_files)} tasks.")
    # ====================================================================

    text_with_speech_list = []

    # 遍历只属于当前进程的文件
    for filename in my_caption_files:
        video_caption_path = os.path.join(args.caption_path, filename)
        
        with open(video_caption_path, "r", encoding="utf-8") as f:
            video_caption = json.load(f)
            
        text_with_speech = video_caption['audiovisual_caption']
        match_rule = r'\[.*?\]\[.*?\]:\s*"?([^"]+)"?'
        speech_text = []
        lang_list = []
        
        for k, v in video_caption['audio_content'].items():
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
                    text_with_speech = text_with_speech.replace(k, f"“{speech_content}”")
                    
        if len(lang_list) == 0 or "zh" in lang_list or 'en' in lang_list:
            text_with_speech_list.append(text_with_speech)
        else:
            continue
        
        # 统一处理路径替换 (替换 07_caption 等逻辑由于已经是参数化，提取纯文件名替换比较安全)
        # 假设原文件名为 xxx.json，对应视频为 xxx.mp4
        video_filename = filename.replace(".json", ".mp4")
        image_filename = filename.replace(".json", ".png")
        
        args.output_path = os.path.join(args.video_save_path, video_filename)
        args.prompt = text_with_speech
        args.images = [ImageConditioningInput(os.path.join(args.images_path, image_filename), 0, 1.0)]
        
        # print(f"[Rank {rank}] Processing: {args.prompt}")
        
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
            streaming_prefetch_count=args.streaming_prefetch_count,
            max_batch_size=args.max_batch_size,
        )

        encode_video(
            video=video,
            fps=args.frame_rate,
            audio=audio,
            output_path=args.output_path,
            video_chunks_number=1,
        )

        torch.save(stored_data_cpu, os.path.join(args.ode_save_path, filename.replace(".json", ".pt")))

if __name__ == "__main__":
    main()