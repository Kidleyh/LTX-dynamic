# ti2v
python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path /gemini/platform/public/aigc/human_guozz2/model/LTX-2.3/ltx-2.3-22b-dev.safetensors \
    --distilled-lora /gemini/platform/public/aigc/human_guozz2/model/LTX-2.3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
    --spatial-upsampler-path /gemini/platform/public/aigc/human_guozz2/model/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    --gemma-root /gemini/platform/public/aigc/human_guozz2/model/gemma-3-12b-it-qat-q4_0-unquantized \
    --prompt "A beautiful sunset over the ocean" \
    --output-path output.mp4

# a2v
python -m ltx_pipelines.a2vid_one_stage \
    --checkpoint-path /gemini/platform/public/aigc/human_guozz2/model/LTX-2.3/ltx-2.3-22b-dev.safetensors \
    --distilled-lora /gemini/platform/public/aigc/human_guozz2/model/LTX-2.3/ltx-2.3-22b-distilled-lora-384.safetensors 0.8 \
    --spatial-upsampler-path /gemini/platform/public/aigc/human_guozz2/model/LTX-2.3/ltx-2.3-spatial-upscaler-x2-1.0.safetensors \
    --gemma-root /gemini/platform/public/aigc/human_guozz2/model/gemma-3-12b-it-qat-q4_0-unquantized \
    --prompt "a man is talking with smile" \
    --audio-path "/gemini/platform/public/aigc/human_guozz2/code/hys/LTX-2/assets/talk_male_law_10s.wav" \
    --image "/gemini/platform/public/aigc/human_guozz2/code/hys/LTX-2/assets/banshen_test.png" 0 0 \ 
    --output-path output.mp4

# pip install -e packages/ltx-trainer/ -i https://mirrors.aliyun.com/pypi/simple/