#!/bin/bash

# 如果需要指定 GPU，可以打开
# export CUDA_VISIBLE_DEVICES=0,1,2,3

torchrun \
  --nproc_per_node=8 \
  /gemini/platform/public/aigc/human_guozz2/code/hys/LTX-2/packages/ltx-pipelines/src/ltx_pipelines/ode_sample_generation_one_stage_multi_gpus.py \
  --images_path /gemini/platform/public/aigc/human_guozz2/code/songqy/opensource_code/LTX-2/packages/images \
  --caption_path /gemini/platform/public/aigc/teleaudio/csh/javg/datasets/datasets/07_caption/part_001/ \
  --video_save_path /gemini/platform/public/aigc/human_guozz2/code/hys/LTX-2/ode_data/0417/videos \
  --ode_save_path /gemini/platform/public/aigc/human_guozz2/code/hys/LTX-2/ode_data/0417/ode_samples \
  --checkpoint-path /gemini/platform/public/aigc/human_guozz2/model/LTX-2.3/ltx-2.3-22b-dev.safetensors \
  --trained_ckpt_path /gemini/platform/public/aigc/human_guozz2/code/hys/LTX-2/ltx_experiments/0415_ltx_node_bs21_bidmd_smalllr/checkpoint_model_005000/model_gen.pt \
  --gemma-root /gemini/platform/public/aigc/human_guozz2/model/gemma-3-12b-it-qat-q4_0-unquantized \
  --prompt None \
  --output-path None \
  --start_idx 0000 \
  --end_idx 2000 \
  --sigmas 1.0 0.757 0.522 0.0 \
  --video-cfg-guidance-scale 1.0 \
  --video-stg-guidance-scale 0.0 \
  --video-rescale-scale 0.0 \
  --audio-cfg-guidance-scale 1.0 \
  --audio-stg-guidance-scale 0.0 \
  --audio-rescale-scale 0.0 \
  --a2v-guidance-scale 1.0 \
  --v2a-guidance-scale 1.0 \
  --num-frames 169