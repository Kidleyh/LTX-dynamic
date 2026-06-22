#!/usr/bin/env python3
import argparse
from pathlib import Path

from omegaconf import OmegaConf

from ltx_distillation.data.ltx_unified_dataset import build_ltx_unified_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="packages/ltx-distillation/configs/causal_dmd/ltx23_causal_dmd_lyh.yaml")
    parser.add_argument("--max-items", type=int, default=64)
    parser.add_argument("--indices", default="0,1,2")
    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.dataset_type = "unified"
    config.unified_dataset_max_items = args.max_items
    dataset = build_ltx_unified_dataset(config)
    print(f"dataset_len={len(dataset)}")
    print(f"sampler_metadata_len={len(dataset.dataset)}")

    for raw_idx in args.indices.split(","):
        idx = int(raw_idx.strip())
        sample = dataset[idx]
        print(f"\n[idx={idx}]")
        print("keys=", sorted(sample.keys()))
        print("pixel_values_shape=", getattr(sample["pixel_values"], "shape", None), sample["pixel_values"].dtype)
        print("audio_data_shape=", tuple(sample["audio_data"].shape), sample["audio_data"].dtype)
        print("video_sample_n_frames=", sample["video_sample_n_frames"])
        print("file_path=", sample["file_path"])
        print("text=", sample["text"][:500].replace("\n", " "))


if __name__ == "__main__":
    main()
