import argparse
import importlib.util
from pathlib import Path

repo = Path('/gemini/platform/public/aigc/human_guozz2/code/lyh/job/OmniStream-LTX-dynamic')
spec = importlib.util.spec_from_file_location('infer_entry', repo / 'scripts/test/infer.py')
infer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(infer)

args = argparse.Namespace(
    infer_mode='causal_self_forcing_streaming',
    checkpoint_path='/gemini/platform/public/aigc/human_guozz2/model/LTX-2.3/ltx-2.3-22b-dev.safetensors',
    generator_ckpt_path=str(repo / 'ltx_experiments/ltx23_causal_dmd_4nodes_512x768_121f_normalopt_seq2_bs1_cpuoffload_8step_log500/checkpoint_model_000500/model_gen.pt'),
    gemma_path='/gemini/platform/public/aigc/human_guozz2/model/gemma-3-12b-it-qat-q4_0-unquantized',
    text_data_path=str(repo / 'assets/testset/codex_ckpt500_smoke.yaml'),
    use_flex_attention=False,
    dtype='bfloat16',
    resolution='480p',
    video_num_frame=121,
    denoising_step_list=[1000, 994, 988, 981, 975, 909, 725, 422],
    output_dir=str(repo / 'ltx_experiments/test_outputs/ckpt500_512x768_121f_8step_smoke'),
    dit_device_idx=0,
    text_encoder_device_idx=1,
    debug_mode=False,
)

infer.infer_with_causal_selfforcing_streaming_pipeline(args)
