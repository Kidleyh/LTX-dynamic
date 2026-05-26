import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image
from transformers import AutoTokenizer

current_file_path = os.path.abspath(__file__)
project_roots = [os.path.dirname(current_file_path), os.path.dirname(os.path.dirname(current_file_path)), os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))]
for project_root in project_roots:
    sys.path.insert(0, project_root) if project_root not in sys.path else None

import random
import multiprocessing
import torch
import torch.multiprocessing as mp

from transformers import WhisperModel, AutoFeatureExtractor
import librosa
import json
from transformers import Wav2Vec2FeatureExtractor
from wan.modules.audio_wav2vec.wav2vec2 import Wav2Vec2Model

import pyloudnorm as pyln
from einops import rearrange
import soundfile as sf
import subprocess
import argparse 
from datetime import datetime

from pipeline import OdeGenerationPipeline
from examples.infer_sample_config.ode_samples_16fps_gd_shuping_1_1 import TEST_SAMPLES
"""
ode_samples_16fps_gd_hengping_1
ode_samples_16fps_gd_shuping_1_0
ode_samples_16fps_gd_shuping_1_1
ode_samples_16fps_openhuman_vid_003_005
ode_samples_16fps_sora2pro1_0
ode_samples_16fps_sora2pro1_3
"""
import torch.distributed as dist

from utils.misc import set_seed
from torchvision.io import write_video

"""
cd /gemini/platform/public/aigc/human_guozz2/code/hys/longtalker_0319
tmux new -s infer
conda deactivate
python wan/data/generate_ode_pair_ai2v_16fps_reward.py --config_path configs/ode_data_generation_ai2v_16fps_gz_757_reward.yaml --output_folder ode_samples_16fps_reward_1209/ode_samples_16fps_openhuman_vid_001
# python wan/data/generate_ode_pair_ai2v_16fps.py --config_path configs/ode_data_generation_ai2v_16fps_gz.yaml --output_folder ode_samples_16fps/celebv_text_01
"""

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a image or video from a text prompt or image using Wan"
    )
    parser.add_argument("--config_path", type=str, help="Path to the config file")
    parser.add_argument("--checkpoint_path", type=str, help="Path to the checkpoint folder")
    parser.add_argument("--data_path", type=str, help="Path to the dataset")
    parser.add_argument("--extended_prompt_path", type=str, help="Path to the extended prompt")
    parser.add_argument("--output_folder", type=str, help="Output folder")
    parser.add_argument("--num_output_frames", type=int, default=21,
                        help="Number of overlap frames between sliding windows")
    parser.add_argument("--i2v", action="store_true", help="Whether to perform I2V (or T2V by default)")
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA parameters")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--num_samples", type=int, default=1, help="Number of samples to generate per prompt")
    parser.add_argument("--save_with_index", action="store_true",
                        help="Whether to save the video using the index or prompt as the filename")
    args = parser.parse_args()
    return args

def custom_init(device, wav2vec):    
    audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec, local_files_only=True).to(device)
    audio_encoder.feature_extractor._freeze_parameters()
    wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec, local_files_only=True)
    return wav2vec_feature_extractor, audio_encoder

def loudness_norm(audio_array, sr=16000, lufs=-23):
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio_array)
    if abs(loudness) > 100:
        return audio_array
    normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
    return normalized_audio

def audio_prepare_multi(left_path, right_path, audio_type, sample_rate=16000):

    if not (left_path=='None' or right_path=='None'):
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = audio_prepare_single(right_path)
    elif left_path=='None':
        human_speech_array2 = audio_prepare_single(right_path)
        human_speech_array1 = np.zeros(human_speech_array2.shape[0])
    elif right_path=='None':
        human_speech_array1 = audio_prepare_single(left_path)
        human_speech_array2 = np.zeros(human_speech_array1.shape[0])

    if audio_type=='para':
        new_human_speech1 = human_speech_array1
        new_human_speech2 = human_speech_array2
    elif audio_type=='add':
        new_human_speech1 = np.concatenate([human_speech_array1[: human_speech_array1.shape[0]], np.zeros(human_speech_array2.shape[0])]) 
        new_human_speech2 = np.concatenate([np.zeros(human_speech_array1.shape[0]), human_speech_array2[:human_speech_array2.shape[0]]])
    sum_human_speechs = new_human_speech1 + new_human_speech2
    return new_human_speech1, new_human_speech2, sum_human_speechs

def get_embedding(speech_array, wav2vec_feature_extractor, audio_encoder, sr=16000, device='cpu'):
    audio_duration = len(speech_array) / sr
    video_length = audio_duration * 16 # Assume the video fps is 25

    # wav2vec_feature_extractor
    audio_feature = np.squeeze(
        wav2vec_feature_extractor(speech_array, sampling_rate=sr).input_values
    )
    audio_feature = torch.from_numpy(audio_feature).float().to(device=device)
    audio_feature = audio_feature.unsqueeze(0)

    # audio encoder
    with torch.no_grad():
        embeddings = audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

    if len(embeddings) == 0:
        print("Fail to extract audio embedding")
        return None

    audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
    audio_emb = rearrange(audio_emb, "b s d -> s b d")

    audio_emb = audio_emb.cpu().detach()
    return audio_emb

def extract_audio_from_video(filename, sample_rate):
    raw_audio_path = filename.split('/')[-1].split('.')[0]+'.wav'
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(filename),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "2",
        str(raw_audio_path),
    ]
    subprocess.run(ffmpeg_command, check=True)
    human_speech_array, sr = librosa.load(raw_audio_path, sr=sample_rate)
    human_speech_array = loudness_norm(human_speech_array, sr)
    os.remove(raw_audio_path)

    return human_speech_array

def audio_prepare_single(audio_path, sample_rate=16000):
    ext = os.path.splitext(audio_path)[1].lower()
    if ext in ['.mp4', '.mov', '.avi', '.mkv']:
        human_speech_array = extract_audio_from_video(audio_path, sample_rate)
        return human_speech_array
    else:
        human_speech_array, sr = librosa.load(audio_path, sr=sample_rate)
        human_speech_array = loudness_norm(human_speech_array, sr)
        return human_speech_array

def run_test(sample_id, test_data, device, rank):

    device = torch.device(f"cuda:{rank}")
    # load pipeline
    args = _parse_args()
    # pipeline init
    config = OmegaConf.load(args.config_path)
    default_config = OmegaConf.load("configs/default_config.yaml")
    config = OmegaConf.merge(default_config, config)
    sampling_steps = 50
    lora_dir = None # ["wan_models/lightx2v_I2V_14B_480p_cfg_step_distill_rank256_bf16.safetensors"] # ["models/Wan2.1_I2V_14B_FusionX_LoRA.safetensors"] # models/Wan21_T2V_14B_lightx2v_cfg_step_distill_lora_rank32.safetensors
    # auto sample steps switch
    lora_path_vxf = "/gemini/platform/public/aigc/human_guozz2/code/hys/longtalker_0319/wan_models/Wan2.1-Fun-14B-InP-MPS.safetensors"
    if lora_dir is not None:
        if "lightx2v" in lora_dir[0]:
            sampling_steps = 4
            config.guidance_scale = 1.0
        elif "Wan2.1_I2V_14B_FusionX_LoRA.safetensors" in lora_dir[0]:
            sampling_steps = 8
            config.guidance_scale = 1.0
    lora_scales = [1.0]
    # AI2VWRAPPER里面默认读取了transformer/的权重
    pipeline = OdeGenerationPipeline(config, device=device, lora_dir=lora_dir, lora_scales=lora_scales, sampling_steps=sampling_steps, lora_path_2=lora_path_vxf)
    pipeline = pipeline.to(device=device, dtype=torch.bfloat16)
    pipeline.independent_first_frame = False
    # load audio model
    wav2vec_dir = "wan_models/chinese-wav2vec2-base"
    wav2vec_feature_extractor, audio_encoder = custom_init('cpu', wav2vec_dir)

    # if local_rank == 0:
    os.makedirs(args.output_folder, exist_ok=True)
    # if dist.is_initialized():
    #     dist.barrier()
    set_seed(args.seed)
    save_audio ="samples/save_audio"
    ode_latent_dir = os.path.join(args.output_folder, "ode_latent")
    video_output_dir = os.path.join(args.output_folder, "video")
    os.makedirs(ode_latent_dir, exist_ok=True)
    os.makedirs(video_output_dir, exist_ok=True)

    pre_existed_video_names = [i for i in os.listdir(video_output_dir) if i.endswith(".mp4")]

    # data in
    # input_data = test_data
    input_data_list = test_data
    for input_data in input_data_list:
        audio_name = os.path.basename(input_data['cond_audio']['person1'])[:-4]

        # a continue strategy
        for n in pre_existed_video_names:
            if audio_name in n:
                print(f"video already exists: {n}")
                continue

        audio_save_dir = os.path.join(save_audio, input_data['cond_image'].split('/')[-1].split('.')[0])
        os.makedirs(audio_save_dir,exist_ok=True)

        if len(input_data['cond_audio'])==2:
            new_human_speech1, new_human_speech2, sum_human_speechs = audio_prepare_multi(input_data['cond_audio']['person1'], input_data['cond_audio']['person2'], input_data['audio_type'])
            audio_embedding_1 = get_embedding(new_human_speech1, wav2vec_feature_extractor, audio_encoder)
            audio_embedding_2 = get_embedding(new_human_speech2, wav2vec_feature_extractor, audio_encoder)
            emb1_path = os.path.join(audio_save_dir, '1.pt')
            emb2_path = os.path.join(audio_save_dir, '2.pt')
            sum_audio = os.path.join(audio_save_dir, 'sum.wav')
            sf.write(sum_audio, sum_human_speechs, 16000)
            torch.save(audio_embedding_1, emb1_path)
            torch.save(audio_embedding_2, emb2_path)
            input_data['cond_audio']['person1'] = emb1_path
            input_data['cond_audio']['person2'] = emb2_path
            input_data['video_audio'] = sum_audio
        elif len(input_data['cond_audio'])==1:
            human_speech = audio_prepare_single(input_data['cond_audio']['person1'])
            audio_embedding = get_embedding(human_speech, wav2vec_feature_extractor, audio_encoder)
            emb_path = os.path.join(audio_save_dir, '1.pt')
            sum_audio = os.path.join(audio_save_dir, 'sum.wav')
            sf.write(sum_audio, human_speech, 16000)
            torch.save(audio_embedding, emb_path)
            input_data['cond_audio']['person1'] = emb_path
            input_data['video_audio'] = sum_audio


        return_video = True
        stored_data, video = pipeline.generate(
            input_data=input_data,
            # return_latents=True,
            guidance_scale=config.guidance_scale,
            frame_num=config.video_sample_n_frames,
            return_video=return_video,
            size_buckget='384p'
        )

        formatted_time = datetime.now().strftime("%m%d%H%M%S")
        ode_data_name = f"{input_data['prompt_index']:05d}_{formatted_time}_rank{rank}##{audio_name}"

        torch.save(
            {input_data['prompt']: stored_data},
            os.path.join(ode_latent_dir, f"{ode_data_name}.pt")
        )

        if return_video:
            temp_video_path = os.path.join(video_output_dir, f"{ode_data_name}.mp4")
            write_video(temp_video_path, video[0], fps=16)
            save_path = os.path.join(video_output_dir, f"{ode_data_name}_audio.mp4")
            final_command = [
                "ffmpeg",
                "-y",
                "-i",
                temp_video_path,
                "-i",
                input_data['video_audio'],
                "-c:v",
                "libx264",
                "-c:a",
                "aac",
                "-shortest",
                save_path,
            ]
            subprocess.run(final_command, check=True)
            os.system(f"rm {temp_video_path}")

# 每个进程分配一个显卡并执行任务
def run_on_gpu(rank, world_size, sample_id, test_data):
    device = torch.device(f'cuda:{rank}')  # 设置当前进程使用的GPU
    result = run_test(sample_id, test_data, device, rank)
    # print(result)

# 初始化多进程环境
def init_process(rank, world_size, TEST_SAMPLES, fn, backend='nccl'):
    # 初始化分布式训练环境
    for i, (sample_id, test_data) in enumerate(TEST_SAMPLES.items()):
        if i == rank:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
            fn(rank, world_size, sample_id, test_data)

def main():
    num_gpus = 8  # 假设你有4张显卡
    world_size = num_gpus

    # 使用 torch.multiprocessing.spawn 来启动多进程
    mp.spawn(init_process,
                args=(world_size, TEST_SAMPLES, run_on_gpu),
                nprocs=num_gpus,  # 每个进程使用一个GPU
                join=True)

if __name__ == '__main__':
    main()


"""
python wan/data/generate_ode_pair_ai2v_16fps_reward.py --config_path configs/ode_data_generation_ai2v_16fps_gz_757_reward.yaml --output_folder samples
"""