import argparse
import importlib.util
import os
from pathlib import Path

repo = Path('/gemini/platform/public/aigc/human_guozz2/code/lyh/job/OmniStream-LTX-dynamic')
spec = importlib.util.spec_from_file_location('infer_entry', repo / 'scripts/test/infer.py')
infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer)

ckpt = os.environ.get('GENERATOR_CKPT')
out = os.environ.get('OUTPUT_DIR')
mode = os.environ.get('INFER_MODE', 'causal_self_forcing_streaming')
if not ckpt or not out:
    raise SystemExit('Need GENERATOR_CKPT and OUTPUT_DIR env vars')

args = argparse.Namespace(
    infer_mode=mode,
    checkpoint_path='/gemini/platform/public/aigc/human_guozz2/model/LTX-2.3/ltx-2.3-22b-dev.safetensors',
    generator_ckpt_path=ckpt,
    gemma_path='/gemini/platform/public/aigc/human_guozz2/model/gemma-3-12b-it-qat-q4_0-unquantized',
    text_data_path=str(repo / 'assets/testset/val10_exclude20000/val10_infer.yaml'),
    use_flex_attention=False,
    dtype='bfloat16',
    resolution='480p',
    video_num_frame=121,
    denoising_step_list=[1000, 994, 988, 981, 975, 909, 725, 422],
    output_dir=out,
    dit_device_idx=0,
    text_encoder_device_idx=1,
    debug_mode=False,
)

if mode == 'causal_self_forcing_streaming':
    infer.infer_with_causal_selfforcing_streaming_pipeline(args)
elif mode == 'causal_self_forcing':
    infer.infer_with_causal_selfforcing_pipeline(args)
else:
    raise SystemExit(f'Unsupported INFER_MODE={mode}')
