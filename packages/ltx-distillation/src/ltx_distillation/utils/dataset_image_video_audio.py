import gc
import json
import os
import random
from contextlib import contextmanager
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from decord import VideoReader
from einops import rearrange
from func_timeout import FunctionTimedOut, func_timeout
from torch.utils.data.dataset import Dataset

import torchaudio
import re
from moviepy import VideoFileClip

VIDEO_READER_TIMEOUT = 20

def get_random_mask(shape, image_start_only=False):
    f, c, h, w = shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

    if not image_start_only:
        if f != 1:
            mask_index = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.05, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05]) 
        else:
            mask_index = np.random.choice([0, 1], p = [0.2, 0.8])
        if mask_index == 0:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)
            mask[:, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 1:
            mask[:, :, :, :] = 1
        elif mask_index == 2:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:, :, :, :] = 1
        elif mask_index == 3:
            mask_frame_index = np.random.randint(1, 5)
            mask[mask_frame_index:-mask_frame_index, :, :, :] = 1
        elif mask_index == 4:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            block_size_x = torch.randint(w // 4, w // 4 * 3, (1,)).item()  # 方块的宽度范围
            block_size_y = torch.randint(h // 4, h // 4 * 3, (1,)).item()  # 方块的高度范围

            start_x = max(center_x - block_size_x // 2, 0)
            end_x = min(center_x + block_size_x // 2, w)
            start_y = max(center_y - block_size_y // 2, 0)
            end_y = min(center_y + block_size_y // 2, h)

            mask_frame_before = np.random.randint(0, f // 2)
            mask_frame_after = np.random.randint(f // 2, f)
            mask[mask_frame_before:mask_frame_after, :, start_y:end_y, start_x:end_x] = 1
        elif mask_index == 5:
            mask = torch.randint(0, 2, (f, 1, h, w), dtype=torch.uint8)
        elif mask_index == 6:
            num_frames_to_mask = random.randint(1, max(f // 2, 1))
            frames_to_mask = random.sample(range(f), num_frames_to_mask)

            for i in frames_to_mask:
                block_height = random.randint(1, h // 4)
                block_width = random.randint(1, w // 4)
                top_left_y = random.randint(0, h - block_height)
                top_left_x = random.randint(0, w - block_width)
                mask[i, 0, top_left_y:top_left_y + block_height, top_left_x:top_left_x + block_width] = 1
        elif mask_index == 7:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            a = torch.randint(min(w, h) // 8, min(w, h) // 4, (1,)).item()  # 长半轴
            b = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()  # 短半轴

            for i in range(h):
                for j in range(w):
                    if ((i - center_y) ** 2) / (b ** 2) + ((j - center_x) ** 2) / (a ** 2) < 1:
                        mask[:, :, i, j] = 1
        elif mask_index == 8:
            center_x = torch.randint(0, w, (1,)).item()
            center_y = torch.randint(0, h, (1,)).item()
            radius = torch.randint(min(h, w) // 8, min(h, w) // 4, (1,)).item()
            for i in range(h):
                for j in range(w):
                    if (i - center_y) ** 2 + (j - center_x) ** 2 < radius ** 2:
                        mask[:, :, i, j] = 1
        elif mask_index == 9:
            for idx in range(f):
                if np.random.rand() > 0.5:
                    mask[idx, :, :, :] = 1
        else:
            raise ValueError(f"The mask_index {mask_index} is not define")
    else:
        if f != 1:
            mask[1:, :, :, :] = 1
        else:
            mask[:, :, :, :] = 1
    return mask

@contextmanager
def VideoReader_contextmanager(*args, **kwargs):
    vr = VideoReader(*args, **kwargs)
    try:
        yield vr
    finally:
        del vr
        gc.collect()

def get_video_reader_batch(video_reader, batch_index):
    frames = video_reader.get_batch(batch_index).asnumpy()
    return frames

def resize_frame(frame, target_short_side):
    h, w, _ = frame.shape
    if h < w:
        if target_short_side > h:
            return frame
        new_h = target_short_side
        new_w = int(target_short_side * w / h)
    else:
        if target_short_side > w:
            return frame
        new_w = target_short_side
        new_h = int(target_short_side * h / w)
    
    resized_frame = cv2.resize(frame, (new_w, new_h))
    return resized_frame

class ImageVideoAudioDataset(Dataset):
    def __init__(
        self,
        ann_path,
        video_sample_size=512, 
        video_sample_stride=1, 
        video_sample_n_frames=121,
        text_drop_ratio=0.,
        enable_bucket=True,
        video_length_drop_start=0.0, 
        video_length_drop_end=1.0,
        enable_inpaint=True,
    ):
        dataset = json.load(open(ann_path))
        self.dataset = dataset

        self.length = len(self.dataset)
        print(f"data scale: {self.length}")
        
        # enable bucket training
        self.enable_bucket = enable_bucket
        self.text_drop_ratio = text_drop_ratio
        self.enable_inpaint = enable_inpaint

        self.video_length_drop_start = video_length_drop_start
        self.video_length_drop_end = video_length_drop_end

        # Video params
        self.video_sample_stride    = video_sample_stride
        self.video_sample_n_frames  = video_sample_n_frames
        self.video_sample_size = tuple(video_sample_size) if not isinstance(video_sample_size, int) else (video_sample_size, video_sample_size)
        self.video_transforms = transforms.Compose(
            [
                transforms.Resize(min(self.video_sample_size)),
                transforms.CenterCrop(self.video_sample_size),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

        self.larger_side_of_image_and_video = min(self.video_sample_size)

    def get_batch(self, idx):
        data_info = self.dataset[idx % len(self.dataset)]
        video_path, video_caption_path = data_info['file_path'], data_info['video_caption_path']
        
        # caption
        with open(video_caption_path, "r", encoding="utf-8") as f:
            video_caption = json.load(f)
        text = video_caption['audiovisual_caption']
        match_rule = r'\[.*?\]\[.*?\]:\s*"?([^"]+)"?'
        for k,v in video_caption['audio_content'].items():
            if "speech content" in k:
                match = re.search(match_rule, v)
                if match:
                    speech_content = match.group(1).strip()
                    text = text.replace(k,f"“{speech_content}”")
    
        with VideoReader_contextmanager(video_path, num_threads=2) as video_reader:
            min_sample_n_frames = min(
                self.video_sample_n_frames, 
                int(len(video_reader) * (self.video_length_drop_end - self.video_length_drop_start) // self.video_sample_stride)
            )
            if min_sample_n_frames == 0:
                raise ValueError(f"No Frames in video.")

            video_length = int(self.video_length_drop_end * len(video_reader))
            clip_length = min(video_length, min_sample_n_frames * self.video_sample_stride)
            max_start_end_time = video_caption['max_start_end_time']
            min_start_idx = int(max_start_end_time[0]*25)
            max_start_idx = min_start_idx + 1
            assert max_start_idx < video_length, "max_start_idx should be smaller than video_length"
            # max_start_idx = max(0, int(max_start_end_time[1]*25) - clip_length)
            # assert clip_length >= self.video_sample_n_frames * self.video_sample_stride, "clip_length is too short"
            start_idx = random.randint(min_start_idx, max_start_idx) if video_length != clip_length else 0
            batch_index = np.linspace(start_idx, start_idx + clip_length - 1, min_sample_n_frames, dtype=int)

            try:
                sample_args = (video_reader, batch_index)
                pixel_values = func_timeout(
                    VIDEO_READER_TIMEOUT, get_video_reader_batch, args=sample_args
                )
                resized_frames = []
                for i in range(len(pixel_values)):
                    frame = pixel_values[i]
                    resized_frame = resize_frame(frame, self.larger_side_of_image_and_video)
                    resized_frames.append(resized_frame)
                pixel_values = np.array(resized_frames)
            except FunctionTimedOut:
                raise ValueError(f"Read {idx} timeout.")
            except Exception as e:
                raise ValueError(f"Failed to extract frames from video. Error is {e}.")

            # Random use no text generation
            if random.random() < self.text_drop_ratio:
                text = ''
            audio_video_offset = int(video_caption.get('audio_video_offset_25fps', 0))
            start_idx = max(0, start_idx - audio_video_offset)
        return pixel_values, text, video_path, start_idx, clip_length


    def get_audio_batch(self, idx, start_idx, clip_length):
        
        def load_audio_from_video(video_path, sr=16000, mono=False):
            clip = VideoFileClip(video_path)
            audio = clip.audio.to_soundarray(fps=sr) 
            if mono and audio.ndim == 2:
                audio = audio.mean(axis=1)  
            elif not mono :
                if audio.ndim == 1:
                    audio = audio[None, :]  # (1, T)
                audio = audio.T
                C, T = audio.shape
                if C == 1:
                    audio = np.repeat(audio, 2, axis=0)
                elif C == 2:
                    pass
                else:
                    mono_audio = audio.mean(axis=0, keepdims=True)  # (1, T)
                    audio = np.repeat(mono_audio, 2, axis=0)
            audio_tensor = torch.tensor(audio, dtype=torch.float32)
            return audio_tensor, sr
        
        data_info = self.dataset[idx % len(self.dataset)]
        video_path = data_info['file_path']
        audio_chunk, sample_rate = load_audio_from_video(video_path)
        abs_max = audio_chunk.abs().max()
        audio_chunk = audio_chunk / (abs_max + 1e-9) * 0.95  # audio normalization

        audio_chunk = F.pad(audio_chunk, (0, 44100))
        audio_chunk = audio_chunk[:, int(start_idx/25*sample_rate):int((start_idx+clip_length)/25*sample_rate)]
        assert audio_chunk.shape[1] > 1/25 * self.video_sample_n_frames * self.video_sample_stride * sample_rate - 2, "audio length is too short"
        return audio_chunk

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        while True:
            sample = {}
            try:
                data_info_local = self.dataset[idx % len(self.dataset)]
                if data_info_local.get('video_train_time', 6.5) == 6.5:
                    self.video_sample_n_frames = 161
                elif data_info_local.get('video_train_time', 6.5) == 5.2:
                    self.video_sample_n_frames = 129
                elif data_info_local.get('video_train_time', 6.5) == 4.0:
                    self.video_sample_n_frames = 97
                else:
                    raise ValueError(f"not supported video_train_time {data_info_local.get('video_train_time', 6.5)}, should be 6.5")
                    
                sample['video_sample_n_frames'] = self.video_sample_n_frames
                pixel_values, text, file_path, start_idx, clip_length = self.get_batch(idx)
                sample["pixel_values"] = pixel_values
                sample["text"] = text
                sample["idx"] = idx
                sample["file_path"] = file_path

                audio_chunk = self.get_audio_batch(idx, start_idx, clip_length)
                if pixel_values.shape[0] != self.video_sample_n_frames:
                    raise ValueError("input video is too short")
                sample["audio_data"] = audio_chunk
                
                if len(sample) > 0:
                    break
            except Exception as e:
                print(e, self.dataset[idx % len(self.dataset)])
                idx = random.randint(0, self.length-1)

        return sample
