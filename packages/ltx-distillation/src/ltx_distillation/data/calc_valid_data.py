import sys
sys.path.insert(0, "/gemini-1/space/human_guozz2/code/hys/videoxfun_0408/videox_fun/data")

import csv
import io
import json
import math
import os
import random
from threading import Thread

import albumentations
import cv2
import gc
import numpy as np
import numpy.typing as npt
import torch
import torchvision.transforms as transforms

from func_timeout import func_timeout, FunctionTimedOut
from decord import VideoReader
from PIL import Image
from torch.utils.data import BatchSampler, Sampler
from torch.utils.data.dataset import Dataset
from contextlib import contextmanager

from tqdm import tqdm
from cattrs import structure
import pickle as pkl
import copy

from collections import defaultdict
from dataclasses import dataclass, field
from teleai_data_tool.file.lmdb_client import LmdbClient
from teleai_data_tool.file.file_client import FileClient
from teleai_data_tool.schema.annotation import Caption, FilterState, CameraMeta
from teleai_data_tool.schema.frame import Frame
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union, Dict


# from transforms import build_transform, image_utils

# from bucket_sampler import ASPECT_RATIO_512

class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict, callable], optional): Sequence of transform
            object or config dict to be composed.
    """

    def __init__(self, transforms: Optional[Sequence[Union[dict, Callable]]]):
        self.transforms: List[Callable] = []

        if transforms is None:
            transforms = []

        for transform in transforms:
            # `Compose` can be built with config dict with type and
            # corresponding arguments.
            if isinstance(transform, dict):
                transform = build_transform(transform)
                if not callable(transform):
                    raise TypeError(
                        f"transform should be a callable object, "
                        f"but got {type(transform)}"
                    )
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError(
                    f"transform must be a callable object or dict, "
                    f"but got {type(transform)}"
                )

    def __call__(self, data: dict) -> Optional[dict]:
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            # The transform will return None when it failed to load images or
            # cannot find suitable augmentation parameters to augment the data.
            # Here we simply return None if the transform returns None and the
            # dataset will handle it by randomly selecting another data sample.
            if data is None:
                return None
        return data

    def __repr__(self):
        """Print ``self.transforms`` in sequence.

        Returns:
            str: Formatted string.
        """
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string

@dataclass
class LipSync:
    LSE_C: Optional[float] = None  
    LSE_D: Optional[float] = None  
    audio_video_offset: Optional[int] = field(default_factory=list)
    audio_valid_range: List[int] = field(default_factory=list)
    audio_bbox: List[List] = field(default_factory=list)

@dataclass
class AudioFeaturePath:
    human: Optional[str] = None  
    origin: Optional[str] = None  
    background: Optional[str] = None  

@dataclass
class AudioRawPath:
    human: Optional[str] = None  
    origin: Optional[str] = None  
    background: Optional[str] = None  

# @dataclass
# class FaceInfo:
#     variance_expr: List[float] = None
#     variance_head: List[float] = None
#     lip_keypoints: List[Any] = field(default_factory=list) # [NOTE] 类型需要注意
#     lip_kp_conf: List[Any] = field(default_factory=list) # [NOTE] 类型需要注意

@dataclass
class Clip:
    id: str
    file_path: str  # clip的存储位置
    height: float
    width: float
    length: float
    fps: int
    valid_range: List[int] = field(default_factory=list)
    # audio_valid_range: List[int] = field(default_factory=list)
    tags: List[str] = field(
        default_factory=list
    )  # 数据主题 people, animals, plants, landscaped, vechicles, object,
    # buildings, animation
    raw_video_path: str = ""  # 原始完整视频存储位置
    start_frame_id: int = 0  # 在原始数据中的开始帧号
    end_frame_id: int = 0  # 在原始视频中的结束帧号
    track_id: int = 0 # 在原始数据中的轨迹id
    data_path: str = ""  # 数据集路径
    train_task: str = "ai2v" # clip用于的训练任务
    frames: List[Frame] = field(default_factory=list)
    caption: Optional[Caption] = None  # 文本标签
    lip_sync: Optional[LipSync] = None  # 同步分数
    audio_feature_path: Optional[AudioFeaturePath] = None  # 音频特征路径
    # audio_raw_path: Optional[AudioFeaturePath] = None  # 音频特征路径
    # face_info: Optional[FaceInfo] = None  # 面部信息
    filter_state: Optional[FilterState] = None  # 过滤器给的状态
    camera_meta: Optional[CameraMeta] = None  # 相机参数
    camera_movement: Optional[str] = (
        None  # 相机运动, 包括 zoom in, zoom out, pand down, pan left, pan right, tilt up,
    )
    # tilt down, tilt left, tilt right, around left , around right, static shot, handheld shot
    meta: Dict[str, str] = field(default_factory=dict)  # other meta info

    @property
    def aspect_ratio(self):
        return self.height / self.width

    @property
    def num_frames(self):
        return self.end_frame_id - self.start_frame_id + 1

VIDEO_READER_TIMEOUT = 20

def get_random_mask(shape, image_start_only=False):
    f, c, h, w = shape
    mask = torch.zeros((f, 1, h, w), dtype=torch.uint8)

    if not image_start_only:
        if f != 1:
            # mask_index = np.random.choice([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], p=[0.05, 0.2, 0.2, 0.2, 0.05, 0.05, 0.05, 0.1, 0.05, 0.05]) 
            mask_index = np.random.choice([-1, -2], p=[0.5, 0.5]) 
        else:
            # mask_index = np.random.choice([0, 1], p = [0.2, 0.8])
            mask_index = np.random.choice([-1, -2], p = [0.5, 0.5])
        if mask_index == -1:
            mask_frame_index = 1
            mask[mask_frame_index:, :, :, :] = 1
        elif mask_index == -2:
            mask_frame_index = np.random.randint(1, 15)
            mask[mask_frame_index:, :, :, :] = 1
        elif mask_index == 0:
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

# class ImageVideoSampler(BatchSampler):
#     """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

#     Args:
#         sampler (Sampler): Base sampler.
#         dataset (Dataset): Dataset providing data information.
#         batch_size (int): Size of mini-batch.
#         drop_last (bool): If ``True``, the sampler will drop the last batch if
#             its size would be less than ``batch_size``.
#         aspect_ratios (dict): The predefined aspect ratios.
#     """

#     def __init__(self,
#                  sampler: Sampler,
#                  dataset: Dataset,
#                  batch_size: int,
#                  drop_last: bool = False
#                 ) -> None:
#         if not isinstance(sampler, Sampler):
#             raise TypeError('sampler should be an instance of ``Sampler``, '
#                             f'but got {sampler}')
#         if not isinstance(batch_size, int) or batch_size <= 0:
#             raise ValueError('batch_size should be a positive integer value, '
#                              f'but got batch_size={batch_size}')
#         self.sampler = sampler
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.drop_last = drop_last

#         # buckets for each aspect ratio
#         self.bucket = {'image':[], 'video':[]}

#     def __iter__(self):
#         for idx in self.sampler:
#             content_type = self.dataset.dataset[idx].get('type', 'image')
#             self.bucket[content_type].append(idx)

#             # yield a batch of indices in the same aspect ratio group
#             if len(self.bucket['video']) == self.batch_size:
#                 bucket = self.bucket['video']
#                 yield bucket[:]
#                 del bucket[:]
#             elif len(self.bucket['image']) == self.batch_size:
#                 bucket = self.bucket['image']
#                 yield bucket[:]
#                 del bucket[:]

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

class AudioVisualDataset(Dataset):
    def __init__(
            self,
            # ann_path, data_root=None, # [NOTE]: data_root should be deleted.
            video_sample_size=512, video_sample_stride=4, video_sample_n_frames=16,
            image_sample_size=512,
            video_repeat=0,
            text_drop_ratio=0.1,
            enable_bucket=False,
            video_length_drop_start=0.0, 
            video_length_drop_end=1.0,
            enable_inpaint=False,
            dataset_paths_txt=[],
        ):

        self.enable_bucket_index = True

        # target_fps = 12.5
        data_fps = 25
        target_fps = data_fps / video_sample_stride
        # video_sample_stride = int(data_fps / target_fps)

        dst_size = (512, 512)
        # num_frames = video_sample_n_frames
        # image_condition_type = "token_replace"
        # self.data_root = data_root

        with open(dataset_paths_txt, "r") as f:
            dataset_paths = [line.strip() for line in f.readlines()]
        data_path_list = dataset_paths 

        self.filter_cfg = dict(
                    dst_size=dst_size,
                    min_num_frames=video_sample_n_frames * video_sample_stride,
                    multiple=16,
                    min_area=480*320,
                    optical_flow_th=2,
                    aesthetic_th=4,
                    ocr_th=0.01,
                    bucket_size_th=4,
                )

        raw_clip_dataset = self.load_data_list(data_path_list)
        # self.dataset = self.load_data_list_debug(data_path_list) # TODO: debug, to delete
        # self.dataset = self.load_data_list_debug_lmk(data_path_list) # TODO: debug, to delete
        valid_data_list_ai2v, valid_data_list_i2v = self.filter_data(raw_clip_dataset)


    def load_data_list(self, data_path_list):
        data_list = []
        for data_path in tqdm(data_path_list):
            
            # [NOTE] adapt variances and face kps
            # landmarks_dir = os.path.join(os.path.dirname(data_path), "v0.0.1", "face", "landmarks")
            # variances_dir = os.path.join(os.path.dirname(data_path), "v0.0.1", "face", "variances")
            # if not os.path.exists(landmarks_dir):
            #     print(f"no landmarks for {data_path}")
            #     continue
            # if not os.path.exists(variances_dir):
            #     print(f"no variances for {data_path}")
            #     continue

            with open(data_path) as f:
                dataset = json.load(f)
            for clip in tqdm(dataset["clips"]):
                temp_clip = copy.deepcopy(clip)

                # [NOTE] load lmk & variance
                # landmark_info_path = os.path.join(landmarks_dir, str(clip['id'])+".pkl")
                # with open(landmark_info_path, "rb") as f:
                #     landmark_info = pkl.load(f)
                # variance_info_path = os.path.join(variances_dir, str(clip['id'])+".pkl")
                # with open(variance_info_path, "rb") as f:
                #     variance_info = pkl.load(f)

                for k in range(len(clip['lip_sync']['LSE_C'])): # track-wise data
                    
                    # [NOTE] add lmk info to clip
                    # temp_clip["face_info"] = {}
                    # temp_clip['face_info']['variance_expr'] = variance_info[k]['expr_variances']
                    # temp_clip['face_info']['variance_head'] = variance_info[k]['head_variances']
                    # temp_clip['face_info']['lip_keypoints'] = [i[self.outer_lip_index] for i in landmark_info[k]['kps']]
                    # temp_clip['face_info']['lip_kp_conf'] = [i[self.outer_lip_index] for i in landmark_info[k]['kps_score']]

                    temp_clip['track_id'] = k # 降低内存调用, 在transforms里读取kp之类的参数
                    temp_clip["data_path"] = os.path.dirname(data_path)

                    temp_clip['lip_sync']['LSE_C'] = clip['lip_sync']['LSE_C'][k]
                    temp_clip['lip_sync']['LSE_D'] = clip['lip_sync']['LSE_D'][k]
                    temp_clip['lip_sync']['audio_video_offset'] = clip['lip_sync']['audio_video_offset'][k]
                    temp_clip['lip_sync']['audio_valid_range'] = clip['lip_sync']['audio_valid_range'][k]
                    temp_clip['lip_sync']['audio_bbox'] = clip['lip_sync']['audio_bbox'][k]
                    clip_struc = structure(temp_clip, Clip)
                    clip_struc.file_path = f"{dataset['clip_data_root']}:{clip_struc.file_path}"
                    clip_struc.meta["data_format"] = dataset["clip_data_type"]
                    data_list.append(clip_struc)
        return data_list


    def longest_above_threshold(self, lst, threshold):
        max_length = 0
        max_start, max_end = -1, -1
        start = 0
        
        for end in range(len(lst)):
            if lst[end] <= threshold:
                start = end + 1  # 重新开始新子数组
            else:
                length = end - start + 1
                if length > max_length:
                    max_length = length
                    max_start, max_end = start, end
        
        return (max_start, max_end) if max_length > 0 else None

    # overwrite filter
    def filter_data(self, raw_clip_dataset):
        dst_size = self.filter_cfg.get("dst_size", (768, 432))
        min_num_frames = self.filter_cfg.get("min_num_frames", 90)
        multiple = self.filter_cfg.get("multiple", 16)
        min_area = self.filter_cfg.get("min_area", dst_size[0] * dst_size[1])
        optical_flow_th = self.filter_cfg.get("optical_flow_th", 2)
        aesthetic_th = self.filter_cfg.get("aesthetic_th", 4)
        bucket_size_th = self.filter_cfg.get("bucket_size_th", 4)
        motion_th = self.filter_cfg.get("motion_th", 0) 
        clearity_th = self.filter_cfg.get("clearity_th", 0.95) 
        ocr_score_th = self.filter_cfg.get("ocr_th", 0) 
        training_suitability_th = self.filter_cfg.get("training_suitability_th", 3.7) 
        area_th = self.filter_cfg.get("area_th", 1280 * 720) 
        new_data_ai2v_list = []
        new_data_i2v_list = []
        shape_list = []
        shape_num_map = defaultdict(int)
        audio_filter_cnt = 0
        face_area_filter_cnt = 0
        score_filter_cnt = 0
        video_len_filter_cnt = 0
        ar_ratio_filter_cnt = 0
        for clip in raw_clip_dataset:

            # audio: 使用这个条件也滤掉了audio_valid_range为-1的情况
            # if abs(clip.lip_sync.audio_video_offset) > 6 or \
            #     clip.lip_sync.LSE_C <= 4 or \
            #         clip.lip_sync.LSE_D >= 13:
            if abs(clip.lip_sync.audio_video_offset) > 5 or \
                clip.lip_sync.LSE_C <= 3 or \
                    clip.lip_sync.LSE_D >= 10:
                audio_filter_cnt += 1
                clip.train_task = "i2v"
                # continue

            # face area
            face_bbox = clip.lip_sync.audio_bbox
            areas = []
            for i in range(len(face_bbox)):
                x1, y1, x2, y2 = face_bbox[i]
                areas.append(int((y2 - y1) * (x2 - x1)))

            # 寻找最长满足人脸大小大于某个阈值的子序列
            # area_flag = True
            # for a in areas:
            #     if a < 128*128:
            #         area_flag = False
            # if not area_flag:
            #     continue
            start_end_indx = self.longest_above_threshold(areas, 128*128) # 128x128 + 90 frame: 1046/15015, 128x128: 1177, 32x32: 1212/15015
            if start_end_indx is None:
                face_area_filter_cnt += 1
                continue
            else:
                start, end = start_end_indx
            # 更新audio_valid_range
            clip.lip_sync.audio_valid_range[0], clip.lip_sync.audio_valid_range[1] = start + clip.lip_sync.audio_valid_range[0], end + clip.lip_sync.audio_valid_range[0]

            
            if clip.height * clip.width < min_area:
                score_filter_cnt += 1
                continue

            # aesthetic
            if (
                clip.filter_state.aesthetic is None
                or clip.filter_state.aesthetic < aesthetic_th
            ):
                score_filter_cnt += 1
                continue

            # optical_flow
            if clip.filter_state.optical_flow != -1.0:
                if (
                    clip.filter_state.optical_flow is None
                    or clip.filter_state.optical_flow < optical_flow_th
                ):
                    score_filter_cnt += 1
                    continue

            # ocr
            if clip.filter_state.ocr_score != -1.0:
                if (
                    clip.filter_state.ocr_score is None
                    or clip.filter_state.ocr_score > ocr_score_th
                ):
                    score_filter_cnt += 1
                    continue

            try:
            # size
                # [NOTE]: 上面已经有area了，这个是什么?
                # if clip.filter_state.area < area_th:
                #     continue

                # length
                if clip.length < min_num_frames:
                    video_len_filter_cnt += 1
                    continue

                # if clip.lip_sync.audio_valid_range[1] - clip.lip_sync.audio_valid_range[0] < min_num_frames:
                #     continue
                # video valid range和audio valid range求交集, 作为最终可用的clip长度
                actual_valid_range_s = max(clip.valid_range[0], clip.lip_sync.audio_valid_range[0])
                actual_valid_range_e = min(clip.valid_range[1], clip.lip_sync.audio_valid_range[1])
                if actual_valid_range_e - actual_valid_range_s < min_num_frames + 10: # [NOTE] 前后留一个冗余
                    video_len_filter_cnt += 1
                    continue

                if (abs(float(clip.width / clip.height) - 0.57) > 0.1) and (abs(float(clip.width / clip.height) - 1) > 0.1) and (abs(float(clip.width / clip.height) - 1.75) > 0.1) :
                    # [NOTE]: 只保留接近16:9 1:1 9:16的视频
                    ar_ratio_filter_cnt += 1
                    continue

                # clearity [NOTE]: 这个目前没有
                # if (
                #     clip.filter_state.clearity is not None
                #     and clip.filter_state.clearity < clearity_th
                # ):
                #     continue

                # motion [NOTE]: 这个目前没有
                # if (
                #     clip.filter_state.motion is not None
                #     and clip.filter_state.motion < motion_th
                # ):
                #     continue

                # training_suitability [NOTE]: 这个目前没有
                # if (
                #     clip.filter_state.video_training_suitability is not None
                #     and clip.filter_state.video_training_suitability < training_suitability_th
                # ):
                #     continue

            except:
                import warnings
                warnings.warn("filter_data:line123 do not have some state")

            if True:
                # dst_width, dst_height = image_utils.get_image_size( # multiple: 最小因子, 按面积比例缩放
                #     (clip.width, clip.height),
                #     (clip.width, clip.height), # [NOTE]: Keep
                #     mode="fixed",
                #     multiple=multiple,
                # )
                dst_width, dst_height = 512, 512
                video_info = f"{dst_width}__{dst_height}"
                if video_info not in shape_list:
                    shape_list.append(video_info)
                setattr(clip, "bucket_index", shape_list.index(video_info))
                shape_num_map[video_info] += 1
                setattr(clip, "video_info", (dst_width, dst_height))
            
            if clip.train_task == "i2v":
                new_data_i2v_list.append(clip)
            elif clip.train_task == "ai2v":
                new_data_ai2v_list.append(clip)
            else:
                raise NotImplementedError
            
        if True: # True之前为enable_bucket_index（用videoxfun内的机制代替）
            invalid_bucket_id_list = []
            for k, v in shape_num_map.items():
                if v < bucket_size_th:
                    invalid_bucket_id_list.append(k) # 每个bucket不少于bucket_size_th
            valid_data_list_i2v = [
                clip
                for clip in new_data_i2v_list
                if clip.bucket_index not in invalid_bucket_id_list
            ]

            valid_data_list_ai2v = [
                clip
                for clip in new_data_ai2v_list
                if clip.bucket_index not in invalid_bucket_id_list
            ]

            bucket_index_list_i2v = defaultdict(list)
            for i, clip in enumerate(valid_data_list_i2v):
                bucket_index_list_i2v[clip.bucket_index].append(i)
            self.bucket_index_list_i2v = [
                np.array(item) for item in bucket_index_list_i2v.values()
            ]

            bucket_index_list_ai2v = defaultdict(list)
            for i, clip in enumerate(valid_data_list_ai2v):
                bucket_index_list_ai2v[clip.bucket_index].append(i)
            self.bucket_index_list_ai2v = [
                np.array(item) for item in bucket_index_list_ai2v.values()
            ]

        # valid_data_list = valid_data_list[:1200]
        # else:
        #     valid_data_list = new_data_list
        print(
            f"finish filter dataset, from {len(raw_clip_dataset)} to ai2v {len(valid_data_list_ai2v)} i2v {len(valid_data_list_i2v)}\n",
            f"bad audio quality: {audio_filter_cnt} \n",
            f"bad score: {score_filter_cnt} \n",
            f"too small face: {face_area_filter_cnt} \n",
            f"too short: {video_len_filter_cnt} \n",
            f"aspect ratio not suitable: {ar_ratio_filter_cnt} \n",
        )

        total_frames = 0
        for c in valid_data_list_ai2v:
            total_frames += len(c.lip_sync.audio_bbox)

        print(
            f"finish filter dataset, total frames: {total_frames}, ai2v total hours {total_frames / 25 / 3600}"
        )

        total_frames = 0
        for c in valid_data_list_i2v:
            total_frames += len(c.lip_sync.audio_bbox)
        print(
            f"finish filter dataset, total frames: {total_frames}, i2v total hours {total_frames / 25 / 3600}"
        )

        # self.ai2v_length = len(valid_data_list_ai2v)
        # self.i2v_length = len(valid_data_list_i2v)

        # valid_data_list = valid_data_list_ai2v + valid_data_list_i2v

        return valid_data_list_ai2v, valid_data_list_i2v


if __name__ == "__main__":
    dataset = AudioVisualDataset(video_sample_stride=1, video_sample_n_frames=81, dataset_paths_txt="/gemini-1/space/human_guozz2/code/hys/videoxfun_0711/scripts/wan2.2/dataset_txts/dataset_all_0807.txt")