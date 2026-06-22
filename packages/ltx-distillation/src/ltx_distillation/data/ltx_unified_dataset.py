import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .unified_dataset import UnifiedCutDataset
from .unified_operators import LoadMagiPromptFile


class LTXUnifiedDatasetAdapter(Dataset):
    """Adapter from UnifiedCutDataset samples to the legacy LTX trainer batch format."""

    def __init__(self, unified_dataset, video_sample_n_frames, frame_rate=24, audio_sample_rate=16000, max_refetch=20):
        self.unified_dataset = unified_dataset
        self.video_sample_n_frames = int(video_sample_n_frames)
        self.frame_rate = int(frame_rate)
        self.audio_sample_rate = int(audio_sample_rate)
        self.max_refetch = int(max_refetch)
        self.dataset = [self._sampler_record(record) for record in unified_dataset.data]
        self._logged_sample_errors = 0

    def _sampler_record(self, record):
        video_path = record.get("video") or record.get("file_path") or record.get("input_audio")
        return {
            "file_path": video_path,
            "video_train_time": record.get("video_train_time", 6.5),
            "width": record.get("width"),
            "height": record.get("height"),
        }

    def __len__(self):
        return len(self.unified_dataset)

    def _video_to_numpy(self, video):
        if video is None or len(video) == 0:
            raise ValueError("empty video")
        frames = []
        for frame in video:
            arr = np.asarray(frame.convert("RGB") if hasattr(frame, "convert") else frame)
            if arr.ndim != 3 or arr.shape[-1] != 3:
                raise ValueError(f"unexpected frame shape {arr.shape}")
            frames.append(arr)
        return np.stack(frames, axis=0)

    def _load_audio(self, video_path, num_frames, av_offset=0):
        from moviepy import VideoFileClip

        clip = VideoFileClip(video_path)
        try:
            if clip.audio is None:
                raise ValueError("video has no audio track")
            sample_rate = self.audio_sample_rate
            audio = clip.audio.to_soundarray(fps=sample_rate)
        finally:
            clip.close()

        if audio.ndim == 1:
            audio = audio[None, :]
        else:
            audio = audio.T
        channels, _ = audio.shape
        if channels == 1:
            audio = np.repeat(audio, 2, axis=0)
        elif channels > 2:
            audio = np.repeat(audio.mean(axis=0, keepdims=True), 2, axis=0)
        elif channels == 2:
            pass
        else:
            raise ValueError("empty audio")

        audio_tensor = torch.tensor(audio, dtype=torch.float32)
        abs_max = audio_tensor.abs().max()
        audio_tensor = audio_tensor / (abs_max + 1e-9) * 0.95

        # Match legacy semantics: offset is in 25fps frame units. Positive values
        # shift audio earlier; negative values skip audio head frames.
        av_offset = int(av_offset or 0)
        start_sample = max(0, int(round((-av_offset) / 25.0 * sample_rate)))
        target_samples = int(math.ceil(num_frames / float(self.frame_rate) * sample_rate))
        end_sample = start_sample + target_samples
        if audio_tensor.shape[-1] < end_sample:
            audio_tensor = torch.nn.functional.pad(audio_tensor, (0, end_sample - audio_tensor.shape[-1]))
        return audio_tensor[:, start_sample:end_sample]

    def __getitem__(self, idx):
        for attempt in range(self.max_refetch + 1):
            item_idx = (idx + attempt) % len(self.unified_dataset)
            try:
                item = self.unified_dataset[item_idx]
                if item is None:
                    raise ValueError("unified dataset returned None")
                pixel_values = self._video_to_numpy(item.get("video"))
                if pixel_values.shape[0] <= 0:
                    raise ValueError("empty pixel_values")
                audio_path = item.get("input_audio") or self.dataset[item_idx].get("file_path")
                if audio_path is None:
                    raise ValueError("missing audio/video path")
                sample = {
                    "pixel_values": pixel_values,
                    "audio_data": self._load_audio(audio_path, pixel_values.shape[0], item.get("av_offset", 0)),
                    "text": item.get("prompt") or "The person is talking.",
                    "idx": item_idx,
                    "file_path": audio_path,
                    "video_sample_n_frames": pixel_values.shape[0],
                }
                return sample
            except Exception as exc:
                if self._logged_sample_errors < 5:
                    path = self.dataset[item_idx].get("file_path", "<unknown>")
                    print(f"[unified dataset warning] refetch idx={item_idx}: {type(exc).__name__}: {exc}; path={path}")
                    self._logged_sample_errors += 1
                idx = random.randint(0, max(0, len(self.unified_dataset) - 1))
        raise RuntimeError(f"Failed to fetch a valid unified dataset sample after {self.max_refetch + 1} attempts")


def build_ltx_unified_dataset(config):
    video_operator = UnifiedCutDataset.default_video_operator(
        base_path=getattr(config, "dataset_base_path", "") or "",
        height=getattr(config, "video_height", None),
        width=getattr(config, "video_width", None),
        num_frames=config.video_sample_n_frames,
        time_division_factor=8,
        time_division_remainder=1,
        frame_rate=getattr(config, "frame_rate", 24),
        fix_frame_rate=True,
    )
    unified_dataset = UnifiedCutDataset(
        base_path=getattr(config, "dataset_base_path", "") or "",
        metadata_path=config.train_data_meta,
        repeat=getattr(config, "train_dataset_repeat", 1),
        data_file_keys=("video_caption_path",),
        main_data_operator=video_operator,
        special_operator_map={"video_caption_path": LoadMagiPromptFile()},
        max_data_items=getattr(config, "unified_dataset_max_items", None),
        min_frames=getattr(config, "video_sample_n_frames", 1),
        enable_label_bbox_crop=getattr(config, "unified_enable_label_bbox_crop", False),
    )
    return LTXUnifiedDatasetAdapter(
        unified_dataset,
        video_sample_n_frames=config.video_sample_n_frames,
        frame_rate=getattr(config, "frame_rate", 24),
        audio_sample_rate=getattr(config, "dataset_audio_sample_rate", 16000),
        max_refetch=getattr(config, "dataset_max_refetch", 20),
    )
