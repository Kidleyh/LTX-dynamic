import argparse
import yaml
import torch
import time
from ltx_distillation.models.ltx_wrapper import LTX2DiffusionWrapper, create_ltx2_wrapper, create_causal_ltx2_wrapper
from ltx_distillation.models.text_encoder_wrapper import GemmaTextEncoderWrapper, create_text_encoder_wrapper
from ltx_distillation.models.vae_wrapper import VideoVAEWrapper, AudioVAEWrapper, create_vae_wrappers
from ltx_core.components.schedulers import LTX2Scheduler
from ltx_core.model.video_vae.tiling import TilingConfig, SpatialTilingConfig, TemporalTilingConfig
from ltx_core.model.transformer.compiling import compile_transformer 

# pipelines
from ltx_distillation.inference.causal_pipeline_ltx23 import LTX23CausalAVInferencePipeline
from ltx_distillation.inference.causal_pipeline_ltx23_ai2v import LTX23CausalAI2VInferencePipeline
from ltx_distillation.inference.causal_pipeline_ltx23_stream import LTX23CausalAVStreamInferencePipeline
from ltx_distillation.inference.causal_pipeline_ltx23_stream_switch import LTX23CausalAVStreamSwitchInferencePipeline

######################## auxiliary functions start ###############################
def _remap_state_dict_keys_generator(state_dict: dict) -> dict:
    sample_keys = list(state_dict.keys())[:20]
    has_model = any(k.startswith("model.") for k in sample_keys)
    if not has_model:
        has_model = any(k.startswith("model.") for k in state_dict)
    has_velocity_model = any(k.startswith("model.velocity_model.") for k in sample_keys)
    if not has_velocity_model:
        has_velocity_model = any(k.startswith("model.velocity_model.") for k in state_dict)
    
    if has_velocity_model:
        remapped = {}
        for k, v in state_dict.items():
            if not k.startswith("model.velocity_model."):
                continue
            new_key = k[len("model.velocity_model."):]
            remapped[new_key] = v
        return remapped

    elif has_model:
        remapped = {}
        for k, v in state_dict.items():
            if not k.startswith("model."):
                continue
            new_key = k[len("model."):]
            remapped[new_key] = v
        return remapped
    
    return state_dict

def _build_wrapper(args, dtype, device, use_causal: bool):
    if use_causal:
        return create_causal_ltx2_wrapper(
            checkpoint_path=args.checkpoint_path,
            gemma_path=args.gemma_path,
            device=device,
            dtype=dtype,
            use_flex_attention=args.use_flex_attention,
            registry=None,
        )
    return create_ltx2_wrapper(
        checkpoint_path=args.checkpoint_path,
        gemma_path=args.gemma_path,
        device=device,
        dtype=dtype,
        registry=None,
    )

def prepare_models(args, dtype, device, text_encoder_device, generator_use_causal: bool, debug_mode=False):
    # setup models
    generator = _build_wrapper(args, dtype=dtype, device="cpu", use_causal=generator_use_causal)
    print("generator original ckpt loaded.")
    if not debug_mode:
        print(f"Loading pretrained generator from {args.generator_ckpt_path}")
        ckpt = torch.load(args.generator_ckpt_path, map_location="cpu")
        gen_sd = ckpt.get("generator", ckpt)
        gen_sd = _remap_state_dict_keys_generator(gen_sd)
        missing_g, unexpected_g = generator.model.velocity_model.load_state_dict(gen_sd, strict=False)
        real_missing_g = [k for k in missing_g if "mask_builder" not in k]
        if real_missing_g:
            print(f"  [generator] missing keys ({len(real_missing_g)}): {real_missing_g[:10]}...")
        if unexpected_g:
            print(f"  [generator] unexpected keys ({len(unexpected_g)}): {unexpected_g[:10]}...")
        generator.to(device)
        print("generator loaded to device.")
    else:
        print("debug mode, skip loading pretrained generator")

    text_encoder = create_text_encoder_wrapper(
        checkpoint_path=args.checkpoint_path,
        gemma_path=args.gemma_path,
        device=text_encoder_device,
        dtype=dtype,
        load_in_8bit=False,
        registry=None,
    )
    video_vae, audio_vae = create_vae_wrappers(
        checkpoint_path=args.checkpoint_path,
        device=device,
        dtype=dtype,
        registry=None,
    )
    return generator, text_encoder, video_vae, audio_vae

# setup add noise func
def add_noise(
    original: torch.Tensor,
    noise: torch.Tensor,
    sigma: torch.Tensor,
) -> torch.Tensor:
    # Reshape sigma for broadcasting
    if sigma.dim() == 1:
        # [B] -> [B, 1, 1, 1, ...] for proper broadcasting
        sigma = sigma.reshape(-1, *[1] * (original.dim() - 1))
    elif sigma.dim() == 2:
        # [B, T] -> [B, T, 1, 1, ...] for video/audio
        sigma = sigma.reshape(*sigma.shape, *[1] * (original.dim() - 2))
    sigma = sigma.to(dtype=original.dtype)
    return ((1 - sigma) * original + sigma * noise).to(dtype=original.dtype)

def setup_denoising_sigmas(args, device, dtype):
    _denoising_sigmas = []
    denoising_step_list = args.denoising_step_list # [1000, 757, 522, 0]
    _full_sigmas = LTX2Scheduler().execute(steps=40)
    for t in denoising_step_list:
        target_sigma = t / 1000.0
        idx = (_full_sigmas - target_sigma).abs().argmin().item()
        _denoising_sigmas.append(_full_sigmas[idx])
    denoising_sigmas = torch.stack(_denoising_sigmas).to(device)
    return denoising_sigmas

def setup_device(args):
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    device = torch.device(f"cuda:{args.dit_device_idx}")
    text_encoder_device = torch.device(f"cuda:{args.text_encoder_device_idx}")
    print(f"device={device} text_encoder_device={text_encoder_device}")
    return dtype, device, text_encoder_device

######################## auxiliary functions end ###############################

######################## infer examples start ###############################

def infer_with_causal_selfforcing_pipeline(args):
    # load test data
    with open(args.text_data_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    print(f"len(data)={len(data)} data[0]={data[0]}")
    # setup dtype and device
    dtype, device, text_encoder_device = setup_device(args)
    # prepare models
    generator, text_encoder, video_vae, audio_vae = prepare_models(args, 
                                                                   dtype, 
                                                                   device, 
                                                                   text_encoder_device, 
                                                                   generator_use_causal=True, 
                                                                   debug_mode=args.debug_mode)
    # setup denoising sigmas
    denoising_sigmas = setup_denoising_sigmas(args, device, dtype)
    # setup inference pipeline
    inference_pipeline = LTX23CausalAVInferencePipeline(
        generator=generator,
        text_encoder=text_encoder,
        video_vae=video_vae,
        audio_vae=audio_vae,
        device=device,
        dtype=dtype,
        use_kv_cache=True,
        clear_cuda_cache_per_round=True,
        # other params
        add_noise_fn=add_noise,
        denoising_sigmas=denoising_sigmas,
        num_frame_per_block=3,
        num_audio_token_per_block=25,
        text_encoder_device=text_encoder_device,
    )
    # run inference for all data
    for item in data:
        output_path = inference_pipeline.generate(
            image_path=item["image"],
            prompt=item["prompt"],
            video_num_frames=args.video_num_frame,
            resolution=args.resolution,
            save_video=True,
            output_dir=args.output_dir,
        )
        print(f"video saved to {output_path}")


def infer_with_causal_selfforcing_streaming_pipeline(args):
    # load test data
    with open(args.text_data_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    print(f"len(data)={len(data)} data[0]={data[0]}")
    # setup dtype and device
    dtype, device, text_encoder_device = setup_device(args)
    # prepare models
    generator, text_encoder, video_vae, audio_vae = prepare_models(args, 
                                                                   dtype, 
                                                                   device, 
                                                                   text_encoder_device, 
                                                                   generator_use_causal=True, 
                                                                   debug_mode=args.debug_mode)

    # print("Start compiling the model")
    # t1 = time.time()
    # generator.model.velocity_model = compile_transformer(generator.model.velocity_model)
    # print(f"Model compiled. Compilation time: {time.time() - t1}")

    
    # setup denoising sigmas
    denoising_sigmas = setup_denoising_sigmas(args, device, dtype)

    # tiling config
    custom_spatial = SpatialTilingConfig(
        tile_size_in_pixels=512, 
        tile_overlap_in_pixels=64,
    )
    custom_temporal = TemporalTilingConfig(
        tile_size_in_frames=24, 
        tile_overlap_in_frames=8,
    )
    config = TilingConfig(
        spatial_config=custom_spatial,
        temporal_config=custom_temporal
    )

    # setup inference pipeline
    inference_pipeline = LTX23CausalAVStreamInferencePipeline(
        generator=generator,
        text_encoder=text_encoder,
        video_vae=video_vae,
        audio_vae=audio_vae,
        device=device,
        dtype=dtype,
        use_kv_cache=True,
        clear_cuda_cache_per_round=True,
        # other params
        add_noise_fn=add_noise,
        denoising_sigmas=denoising_sigmas,
        num_frame_per_block=3,
        num_audio_token_per_block=25,
        text_encoder_device=text_encoder_device,
        tiling_config=config,
    )
    # run inference for all data
    for item in data:
        output_path = inference_pipeline.generate(
            image_path=item["image"],
            prompt=item["prompt"],
            video_num_frames=args.video_num_frame,
            resolution=args.resolution,
            save_video=True,
            output_dir=args.output_dir,
        )
        print(f"video saved to {output_path}")


def infer_with_causal_selfforcing_streaming_switch_pipeline(args):
    # load test data
    with open(args.text_data_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    print(f"len(data)={len(data)} data[0]={data[0]}")
    # setup dtype and device
    dtype, device, text_encoder_device = setup_device(args)
    # prepare models
    generator, text_encoder, video_vae, audio_vae = prepare_models(args, 
                                                                   dtype, 
                                                                   device, 
                                                                   text_encoder_device, 
                                                                   generator_use_causal=True, 
                                                                   debug_mode=args.debug_mode)

    # print("Start compiling the model")
    # t1 = time.time()
    # generator.model.velocity_model = compile_transformer(generator.model.velocity_model)
    # print(f"Model compiled. Compilation time: {time.time() - t1}")

    
    # setup denoising sigmas
    denoising_sigmas = setup_denoising_sigmas(args, device, dtype)

    # tiling config
    custom_spatial = SpatialTilingConfig(
        tile_size_in_pixels=512, 
        tile_overlap_in_pixels=64,
    )
    custom_temporal = TemporalTilingConfig(
        tile_size_in_frames=24, 
        tile_overlap_in_frames=8,
    )
    config = TilingConfig(
        spatial_config=custom_spatial,
        temporal_config=custom_temporal
    )

    # setup inference pipeline
    inference_pipeline = LTX23CausalAVStreamSwitchInferencePipeline(
        generator=generator,
        text_encoder=text_encoder,
        video_vae=video_vae,
        audio_vae=audio_vae,
        device=device,
        dtype=dtype,
        use_kv_cache=True,
        clear_cuda_cache_per_round=True,
        # other params
        add_noise_fn=add_noise,
        denoising_sigmas=denoising_sigmas,
        num_frame_per_block=3,
        num_audio_token_per_block=25,
        text_encoder_device=text_encoder_device,
        tiling_config=config,
    )
    # run inference for all data
    for item in data:
        output_path = inference_pipeline.generate(
            image_path=item["image"],
            prompt_list=item["prompt"],
            video_num_frames=args.video_num_frame,
            resolution=args.resolution,
            save_video=True,
            output_dir=args.output_dir,
        )
        print(f"video saved to {output_path}")


def infer_with_causal_selfforcing_ai2v_pipeline(args):
    # load test data
    with open(args.text_data_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    print(f"len(data)={len(data)} data[0]={data[0]}")
    # setup dtype and device
    dtype, device, text_encoder_device = setup_device(args)
    # prepare models
    generator, text_encoder, video_vae, audio_vae = prepare_models(args, 
                                                                   dtype, 
                                                                   device, 
                                                                   text_encoder_device, 
                                                                   generator_use_causal=True, 
                                                                   debug_mode=args.debug_mode)
    # print("Start compiling the model")
    # t1 = time.time()
    # generator = torch.compile(generator)
    # print(f"Model compiled. Compilation time: {time.time() - t1}")

    # setup denoising sigmas
    denoising_sigmas = setup_denoising_sigmas(args, device, dtype)
    # setup inference pipeline
    inference_pipeline = LTX23CausalAI2VInferencePipeline(
        generator=generator,
        text_encoder=text_encoder,
        video_vae=video_vae,
        audio_vae=audio_vae,
        device=device,
        dtype=dtype,
        use_kv_cache=True,
        clear_cuda_cache_per_round=True,
        # other params
        add_noise_fn=add_noise,
        denoising_sigmas=denoising_sigmas,
        num_frame_per_block=3,
        num_audio_token_per_block=25,
        text_encoder_device=text_encoder_device,
    )
    # run inference for all data
    for item in data:
        output_path = inference_pipeline.generate(
            image_path=item["image"],
            audio_path=item["audio"],
            prompt=item["prompt"],
            video_num_frames=args.video_num_frame,
            resolution=args.resolution,
            save_video=True,
            output_dir=args.output_dir,
        )
        print(f"video saved to {output_path}")

if __name__ == "__main__":

    """
    example run cmd:

        standard i2av:
            python scripts/test/infer.py \
            --infer_mode causal_self_forcing \
            --checkpoint_path /gemini/platform/public/aigc/human_guozz2/model/LTX-2.3/ltx-2.3-22b-dev.safetensors \
            --generator_ckpt_path /gemini/platform/public/aigc/human_guozz2/code/hys/LTX-2/ltx_experiments/0425_ltx_node_bs48_causaldmd_lr2e-5_cfgv5a9_astart0/checkpoint_model_002000/model_gen.pt \
            --output_dir /gemini/platform/public/aigc/human_guozz2/code/hys/LTX-2/ltx_experiments/test_outputs/0428/2000 \
            --gemma_path /gemini/platform/public/aigc/human_guozz2/model/gemma-3-12b-it-qat-q4_0-unquantized \
            --text_data_path /gemini/platform/public/aigc/human_guozz2/code/hys/LTX-2/assets/testset/testdata.yaml
        ai2v:
            python scripts/test/infer.py \
            --infer_mode causal_self_forcing_ai2v \
            --checkpoint_path /gemini/platform/public/aigc/human_guozz2/model/LTX-2.3/ltx-2.3-22b-dev.safetensors \
            --generator_ckpt_path /gemini/platform/public/aigc/human_guozz2/code/hys/LTX-2/ltx_experiments/0425_ltx_node_bs48_causaldmd_lr2e-5_cfgv5a9_astart0/checkpoint_model_002000/model_gen.pt \
            --output_dir /gemini/platform/public/aigc/human_guozz2/code/hys/LTX-2/ltx_experiments/test_outputs/0428/2000-ai2v \
            --gemma_path /gemini/platform/public/aigc/human_guozz2/model/gemma-3-12b-it-qat-q4_0-unquantized \
            --text_data_path /gemini/platform/public/aigc/human_guozz2/code/hys/LTX-2/assets/testset/testdata_ai2v.yaml

    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--infer_mode", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--generator_ckpt_path", type=str, required=True)
    parser.add_argument("--gemma_path", type=str, required=True)
    parser.add_argument("--text_data_path", type=str, default="/gemini/platform/public/aigc/human_guozz2/code/hys/LTX-2/assets/testset/testdata.yaml")
    parser.add_argument("--use_flex_attention", type=bool, default=False)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--resolution", type=str, default="480p")
    parser.add_argument("--video_num_frame", type=int, default=169)
    parser.add_argument("--denoising_step_list", type=list, default=[1000, 757, 522, 0])
    parser.add_argument("--output_dir", type=str, default="ltx_experiments/test_output_dir")
    parser.add_argument("--dit_device_idx", type=int, default=0)
    parser.add_argument("--text_encoder_device_idx", type=int, default=1)
    parser.add_argument("--debug_mode", type=bool, default=False)
    args = parser.parse_args()

    # infer with causal self-forcing pipeline
    if args.infer_mode == "causal_self_forcing":
        infer_with_causal_selfforcing_pipeline(args)
    elif args.infer_mode == "causal_self_forcing_ai2v": # ai2v推理
        infer_with_causal_selfforcing_ai2v_pipeline(args)
    elif args.infer_mode == "causal_self_forcing_streaming": # vae支持长视频推理, 其他和causal_self_forcing相同
        infer_with_causal_selfforcing_streaming_pipeline(args)
    elif args.infer_mode == "causal_self_forcing_streaming_switch": # 支持多个prompt切换推理
        infer_with_causal_selfforcing_streaming_switch_pipeline(args)