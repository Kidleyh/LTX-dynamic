# Copyright (c) OpenMMLab. All rights reserved.
import os
from typing import (Generic, Iterable, Iterator, List, Optional, Sequence,
                    Sized, TypeVar, Union)

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import BatchSampler, Dataset, Sampler
import random

# from asyncio import taskgroups

# ASPECT_RATIO_512 = {
#     '0.25': [256.0, 1024.0], '0.26': [256.0, 992.0], '0.27': [256.0, 960.0], '0.28': [256.0, 928.0],
#     '0.32': [288.0, 896.0], '0.33': [288.0, 864.0], '0.35': [288.0, 832.0], '0.4': [320.0, 800.0],
#     '0.42': [320.0, 768.0], '0.48': [352.0, 736.0], '0.5': [352.0, 704.0], '0.52': [352.0, 672.0],
#     '0.57': [384.0, 672.0], '0.6': [384.0, 640.0], '0.68': [416.0, 608.0], '0.72': [416.0, 576.0],
#     '0.78': [448.0, 576.0], '0.82': [448.0, 544.0], '0.88': [480.0, 544.0], '0.94': [480.0, 512.0],
#     '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], '1.13': [544.0, 480.0], '1.21': [544.0, 448.0],
#     '1.29': [576.0, 448.0], '1.38': [576.0, 416.0], '1.46': [608.0, 416.0], '1.67': [640.0, 384.0],
#     '1.75': [672.0, 384.0], '2.0': [704.0, 352.0], '2.09': [736.0, 352.0], '2.4': [768.0, 320.0],
#     '2.5': [800.0, 320.0], '2.89': [832.0, 288.0], '3.0': [864.0, 288.0], '3.11': [896.0, 288.0],
#     '3.62': [928.0, 256.0], '3.75': [960.0, 256.0], '3.88': [992.0, 256.0], '4.0': [1024.0, 256.0]
# }

# ASPECT_RATIO_512 = {
#     '0.57': [384.0, 672.0],
#     '1.0': [512.0, 512.0],
#     '1.75': [672.0, 384.0],
# }
# ASPECT_RATIO_512 = {
#      '0.26': [320, 1216], '0.38': [384, 1024], '0.50': [448, 896], '0.67': [512, 768], 
#      '0.82': [576, 704],  '1.00': [640, 640],  '1.22': [704, 576], '1.50': [768, 512], 
#      '1.86': [832, 448],  '2.00': [896, 448],  '2.50': [960, 384], '2.83': [1088, 384], 
#      '3.60': [1152, 320], '3.80': [1216, 320], '4.00': [1280, 320]}

# ASPECT_RATIO_512 = {
#      '0.26': [320, 1216], '0.38': [384, 1024], '0.50': [448, 896], '0.67': [512, 768], 
#      '0.82': [576, 704],  '1.00': [640, 640],  '1.22': [704, 576], '1.50': [768, 512], 
#      '1.86': [832, 448],  '2.00': [896, 448],  '2.50': [960, 384], '2.83': [1088, 384], 
#      '3.60': [1152, 320], '3.80': [1216, 320], '4.00': [1280, 320]}

# ASPECT_RATIO_512 = {
#     '0.25': [256.0, 1024.0], '0.32': [288.0, 896.0], '0.42': [320.0, 768.0], 
#     '0.57': [384.0, 672.0], '0.78': [448.0, 576.0], '1.0': [512.0, 512.0], 
#     '1.29': [576.0, 448.0], '1.75': [672.0, 384.0], '2.5': [800.0, 320.0], '3.62': [928.0, 256.0],
# }

ASPECT_RATIO_512 = {
    '0.57': [384.0, 672.0], '1.0': [512.0, 512.0], '1.75': [672.0, 384.0]
}

# ASPECT_RATIO_5B = {
#     '1.8': [1152, 640], '0.56': [640, 1152], # 16:9
#     '1.33': [1024, 768], '0.75': [768, 1024], # 4:3
#     '1.0':  [896, 896], # 1:1
# }

ASPECT_RATIO_5B = {
    '1.75': [1120, 640], '0.57': [640, 1120], # 16:9
    '1.33': [1024, 768], '0.75': [768, 1024], # 4:3
    '1.0':  [896, 896], # 1:1
}

ASPECT_RATIO_5B_SMALL = {
    '1.75': [1056, 608], '0.57': [608, 1056], # 16:9
    '1.33': [928, 704], '0.75': [704, 928], # 4:3
    '1.0':  [800, 800], # 1:1
}

ASPECT_RATIO_608 = {
    '0.58': [480, 832], '0.81':  [544, 672], '1.0':  [608, 608], 
    '1.24': [672, 544], '1.73':  [832, 480],
}



ASPECT_RATIO_RANDOM_CROP_512 = {
    '0.42': [320.0, 768.0], '0.5': [352.0, 704.0], 
    '0.57': [384.0, 672.0], '0.68': [416.0, 608.0], '0.78': [448.0, 576.0], '0.88': [480.0, 544.0], 
    '0.94': [480.0, 512.0], '1.0': [512.0, 512.0], '1.07': [512.0, 480.0], 
    '1.13': [544.0, 480.0], '1.29': [576.0, 448.0], '1.46': [608.0, 416.0], '1.75': [672.0, 384.0], 
    '2.0': [704.0, 352.0],  '2.4': [768.0, 320.0]
}
ASPECT_RATIO_RANDOM_CROP_PROB = [
    1, 2,
    4, 4, 4, 4,
    8, 8, 8,
    4, 4, 4, 4,
    2, 1
]
ASPECT_RATIO_RANDOM_CROP_PROB = np.array(ASPECT_RATIO_RANDOM_CROP_PROB) / sum(ASPECT_RATIO_RANDOM_CROP_PROB)

def get_closest_ratio(height: float, width: float, ratios: dict = ASPECT_RATIO_512):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)

def get_image_size_without_loading(path):
    with Image.open(path) as img:
        return img.size  # (width, height)

class RandomSampler(Sampler[int]):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.

    If with replacement, then user can specify :attr:`num_samples` to draw.

    Args:
        data_source (Dataset): dataset to sample from
        replacement (bool): samples are drawn on-demand with replacement if ``True``, default=``False``
        num_samples (int): number of samples to draw, default=`len(dataset)`.
        generator (Generator): Generator used in sampling.
    """

    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, replacement: bool = False,
                 num_samples: Optional[int] = None, generator=None) -> None:
        self.data_source = data_source
        self.replacement = replacement
        self._num_samples = num_samples
        self.generator = generator
        self._pos_start = 0

        if not isinstance(self.replacement, bool):
            raise TypeError(f"replacement should be a boolean value, but got replacement={self.replacement}")

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(f"num_samples should be a positive integer value, but got num_samples={self.num_samples}")

    @property
    def num_samples(self) -> int:
        # dataset size might change at runtime
        if self._num_samples is None:
            return len(self.data_source)
        return self._num_samples

    def __iter__(self) -> Iterator[int]:
        n = len(self.data_source)
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(high=n, size=(32,), dtype=torch.int64, generator=generator).tolist()
            yield from torch.randint(high=n, size=(self.num_samples % 32,), dtype=torch.int64, generator=generator).tolist()
        else:
            for _ in range(self.num_samples // n):
                xx = torch.randperm(n, generator=generator).tolist()
                if self._pos_start >= n:
                    self._pos_start = 0
                # Debug print disabled to keep training logs small.
                for idx in range(self._pos_start, n):
                    yield xx[idx]
                    self._pos_start = (self._pos_start + 1) % n
                self._pos_start = 0
            yield from torch.randperm(n, generator=generator).tolist()[:self.num_samples % n]

    def __len__(self) -> int:
        return self.num_samples

class AspectRatioBatchImageSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """
    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        train_folder: str = None,
        aspect_ratios: dict = ASPECT_RATIO_512,
        drop_last: bool = False,
        config=None,
        **kwargs
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.train_folder = train_folder
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last
        self.config = config
        # buckets for each aspect ratio 
        self._aspect_ratio_buckets = {ratio: [] for ratio in aspect_ratios}
        # [str(k) for k, v in aspect_ratios] 
        self.current_available_bucket_keys = list(aspect_ratios.keys())

    def __iter__(self):
        for idx in self.sampler:
            try:
                image_dict = self.dataset[idx]

                width, height = image_dict.get("width", None), image_dict.get("height", None)
                if width is None or height is None:
                    image_id, name = image_dict['file_path'], image_dict['text']
                    if self.train_folder is None:
                        image_dir = image_id
                    else:
                        image_dir = os.path.join(self.train_folder, image_id)

                    width, height = get_image_size_without_loading(image_dir)

                    ratio = height / width # self.dataset[idx]
                else:
                    height = int(height)
                    width = int(width)
                    ratio = height / width # self.dataset[idx]
            except Exception as e:
                print(e)
                continue
            # find the closest aspect ratio
            closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
            if closest_ratio not in self.current_available_bucket_keys:
                continue
            bucket = self._aspect_ratio_buckets[closest_ratio]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

class AspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """
    def __init__(
        self,
        sampler: Sampler,
        dataset: Dataset,
        batch_size: int,
        video_folder: str = None,
        train_data_format: str = "webvid",
        aspect_ratios: dict = ASPECT_RATIO_512,
        drop_last: bool = False,
        config=None,
        **kwargs
    ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.video_folder = video_folder
        self.train_data_format = train_data_format
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last
        self.config = config
        # buckets for each aspect ratio 
        self._aspect_ratio_buckets = {ratio: [] for ratio in aspect_ratios}
        # [str(k) for k, v in aspect_ratios] 
        self.current_available_bucket_keys = list(aspect_ratios.keys())

    def __iter__(self):
        for idx in self.sampler:
            try:
                video_dict = self.dataset[idx]
                width, more = video_dict.get("width", None), video_dict.get("height", None)

                if width is None or height is None:
                    if self.train_data_format == "normal":
                        video_id, name = video_dict['file_path'], video_dict['text']
                        if self.video_folder is None:
                            video_dir = video_id
                        else:
                            video_dir = os.path.join(self.video_folder, video_id)
                    else:
                        videoid, name, page_dir = video_dict['videoid'], video_dict['name'], video_dict['page_dir']
                        video_dir = os.path.join(self.video_folder, f"{videoid}.mp4")
                    cap = cv2.VideoCapture(video_dir)

                    # 获取视频尺寸
                    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 浮点数转换为整数
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 浮点数转换为整数
                    
                    ratio = height / width # self.dataset[idx]
                else:
                    height = int(height)
                    width = int(width)
                    ratio = height / width # self.dataset[idx]
            except Exception as e:
                print(e)
                continue
            # find the closest aspect ratio
            closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
            if closest_ratio not in self.current_available_bucket_keys:
                continue
            bucket = self._aspect_ratio_buckets[closest_ratio]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

class AspectRatioBatchImageVideoSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler: Sampler,
                 dataset: Dataset,
                 batch_size: int,
                 train_folder: str = None,
                 aspect_ratios: dict = ASPECT_RATIO_512,
                 drop_last: bool = False
                ) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.dataset = dataset
        self.train_folder = train_folder
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.current_available_bucket_keys = list(aspect_ratios.keys())
        self.bucket = {
            'image':{ratio: [] for ratio in aspect_ratios}, 
            'video':{ratio: [] for ratio in aspect_ratios}
        }

    def __iter__(self):
        for idx in self.sampler:
            # content_type = self.dataset[idx].get('type', 'image')
            # if content_type == 'image':
            #     try:
            #         image_dict = self.dataset[idx]

            #         width, height = image_dict.get("width", None), image_dict.get("height", None)
            #         if width is None or height is None:
            #             image_id, name = image_dict['file_path'], image_dict['text']
            #             if self.train_folder is None:
            #                 image_dir = image_id
            #             else:
            #                 image_dir = os.path.join(self.train_folder, image_id)

            #             width, height = get_image_size_without_loading(image_dir)

            #             ratio = height / width # self.dataset[idx]
            #         else:
            #             height = int(height)
            #             width = int(width)
            #             ratio = height / width # self.dataset[idx]
            #     except Exception as e:
            #         print(e)
            #         continue
            #     # find the closest aspect ratio
            #     closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
            #     if closest_ratio not in self.current_available_bucket_keys:
            #         continue
            #     bucket = self.bucket['image'][closest_ratio]
            #     bucket.append(idx)
            #     # yield a batch of indices in the same aspect ratio group
            #     if len(bucket) == self.batch_size:
            #         yield bucket[:]
            #         del bucket[:]
            # else:
            try:
                video_dict = self.dataset[idx]
                width, height = video_dict.width, video_dict.height
                # if width is None or height is None:
                #     video_id, name = video_dict['file_path'], video_dict['text']
                #     if self.train_folder is None:
                #         video_dir = video_id
                #     else:
                #         video_dir = os.path.join(self.train_folder, video_id)
                #     cap = cv2.VideoCapture(video_dir)

                #     # 获取视频尺寸
                #     width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))   # 浮点数转换为整数
                #     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 浮点数转换为整数
                    
                #     ratio = height / width # self.dataset[idx]
                # else:
                height = int(height)
                width = int(width)
                ratio = height / width # self.dataset[idx]
            except Exception as e:
                print(e)
                continue
            # find the closest aspect ratio
            closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
            if closest_ratio not in self.current_available_bucket_keys:
                continue
            bucket = self.bucket['video'][closest_ratio]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]


class AspectRatioBatchSingletaskVideoSampler(BatchSampler):

    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler_ai2v: Sampler,
                 ai2v_dataset: Dataset,
                 batch_size: int,
                 aspect_ratios: dict = ASPECT_RATIO_512,
                 drop_last: bool = False
                ) -> None:

        if not isinstance(sampler_ai2v, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler_ai2v}')

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')

        self.sampler_ai2v = sampler_ai2v
        self.ai2v_dataset = ai2v_dataset
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.current_available_bucket_keys = list(aspect_ratios.keys())
        self.bucket = {
            # 'video_ai2v':{ratio: [] for ratio in aspect_ratios}, 
            # 'video_i2v':{ratio: [] for ratio in aspect_ratios}
            'video_singletask':{ratio: [] for ratio in aspect_ratios}
        }

        self.task_ai2v_prob = 1.0 # 1
        self.task_i2v_prob = 0.0 # 0

        assert self.task_ai2v_prob == 1.0, "task_ai2v_prob should be 1.0"

        self.epoch = 0

    def __iter__(self):
        for idx in self.sampler_ai2v:
            try:
                real_idx =(idx + self.epoch * 10) % len(self.ai2v_dataset)
                video_dict = self.ai2v_dataset[real_idx]
                width, height = video_dict.width, video_dict.height
                height = int(height)
                width = int(width)
                ratio = height / width # self.dataset[idx]
            except Exception as e:
                print(e)
                continue
            # find the closest aspect ratio
            closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
            if closest_ratio not in self.current_available_bucket_keys:
                continue
            bucket = self.bucket['video_singletask'][closest_ratio]
            bucket.append(real_idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

class AspectRatioBatchMultitaskVideoSampler(BatchSampler):

    """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        dataset (Dataset): Dataset providing data information.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
        aspect_ratios (dict): The predefined aspect ratios.
    """

    def __init__(self,
                 sampler_ai2v: Sampler,
                 sampler_i2v: Sampler,
                 ai2v_dataset: Dataset,
                 i2v_dataset: Dataset,
                 batch_size: int,
                 train_folder: str = None,
                 aspect_ratios: dict = ASPECT_RATIO_512,
                 ai2v_ratio: float = 1.0,
                 drop_last: bool = False

                ) -> None:

        if not isinstance(sampler_ai2v, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler_ai2v}')

        if not isinstance(sampler_i2v, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler_i2v}')

        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')

        self.sampler = sampler_ai2v
        self.sampler_i2v = sampler_i2v
        
        self.dataset = ai2v_dataset
        self.i2v_dataset = i2v_dataset

        self.train_folder = train_folder
        self.batch_size = batch_size
        self.aspect_ratios = aspect_ratios
        self.drop_last = drop_last

        # buckets for each aspect ratio
        self.current_available_bucket_keys = list(aspect_ratios.keys())
        self.bucket = {
            'video_multitask':{ratio: [] for ratio in aspect_ratios}
        }

        self.task_ai2v_prob = ai2v_ratio # 1
        self.task_i2v_prob = 1.0 - ai2v_ratio # 0

        print(f"ai2v sampling prob: {self.task_ai2v_prob}, i2v sampling prob: {self.task_i2v_prob}")
        assert self.task_ai2v_prob + self.task_i2v_prob == 1.0, "task_ai2v_prob + task_i2v_prob should be 1.0"

    def __iter__(self):

        iter_ai2v = iter(self.sampler)
        iter_i2v = iter(self.sampler_i2v)

        while True:
            if random.random() < self.task_ai2v_prob:
                task = 'ai2v'
            elif random.random() < self.task_ai2v_prob + self.task_i2v_prob:
                task = 'i2v'

            try:
                if task == 'ai2v':
                    try:
                        idx = next(iter_ai2v)
                    except StopIteration:
                        print("ai2v数据迭代完，开始新的循环...")
                        iter_ai2v = iter(self.sampler)  # 重新创建迭代器
                        idx = next(iter_ai2v)
                    # idx = next(iter_ai2v)
                    video_dict = self.dataset[idx]
                elif task == 'i2v':
                    try:
                        idx = next(iter_i2v)
                    except StopIteration:
                        print("i2v数据迭代完，开始新的循环...")
                        iter_i2v = iter(self.sampler_i2v)  # 重新创建迭代器
                        idx = next(iter_i2v)
                    # idx = next(iter_i2v)
                    video_dict = self.i2v_dataset[idx]

                width, height = video_dict.width, video_dict.height
                height = int(height)
                width = int(width)
                ratio = height / width # self.dataset[idx]
            except Exception as e:
                print(e)
                continue
            # find the closest aspect ratio
            closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
            if closest_ratio not in self.current_available_bucket_keys:
                continue

            bucket = self.bucket['video_multitask'][closest_ratio]
            if task == 'i2v':
                idx = idx + len(self.dataset) # 实际dataset中i2v的索引
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        # iter_ai2v = iter(self.sampler_ai2v)
        # iter_i2v = iter(self.sampler_i2v)
        
        # while True:
        #     if random.random() < self.task_ai2v_prob:
        #         task = 'ai2v'
        #     elif random.random() < self.task_ai2v_prob + self.task_i2v_prob:
        #         task = 'i2v'

        #     try:
        #         if task == 'ai2v':
        #             idx = next(iter_ai2v)
        #             video_dict = self.ai2v_dataset[idx]
        #         elif task == 'i2v':
        #             idx = next(iter_i2v)
        #             video_dict = self.i2v_dataset[idx]

        #         width, height = video_dict.width, video_dict.height
        #         height = int(height)
        #         width = int(width)
        #         ratio = height / width # self.dataset[idx]
        #     except Exception as e:
        #         print(e)
        #         continue
        #     # find the closest aspect ratio
        #     closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
        #     if closest_ratio not in self.current_available_bucket_keys:
        #         continue
        #     # if task == 'ai2v':
        #     #     bucket = self.bucket['video_ai2v'][closest_ratio]
        #     # elif task == 'i2v':
        #     #     bucket = self.bucket['video_i2v'][closest_ratio]
        #     #     idx = idx + len(self.ai2v_dataset) # 实际dataset中i2v的索引
        #     # else:
        #     #     raise NotImplementedError
        #     bucket = self.bucket['video_multitask'][closest_ratio]
        #     if task == 'i2v':
        #         idx = idx + len(self.ai2v_dataset) # 实际dataset中i2v的索引
        #     bucket.append(idx)
        #     # yield a batch of indices in the same aspect ratio group
        #     if len(bucket) == self.batch_size:
        #         yield bucket[:]
        #         del bucket[:]


# class AspectRatioBatchMultitaskVideoSampler(BatchSampler):
#     # """A sampler wrapper for grouping images with similar aspect ratio into a same batch.

#     # Args:
#     #     sampler (Sampler): Base sampler.
#     #     dataset (Dataset): Dataset providing data information.
#     #     batch_size (int): Size of mini-batch.
#     #     drop_last (bool): If ``True``, the sampler will drop the last batch if
#     #         its size would be less than ``batch_size``.
#     #     aspect_ratios (dict): The predefined aspect ratios.
#     # """

#     # def __init__(self,
#     #              sampler: Sampler,
#     #              dataset: Dataset,
#     #              batch_size: int,
#     #              train_folder: str = None,
#     #              aspect_ratios: dict = ASPECT_RATIO_512,
#     #              drop_last: bool = False
#     #             ) -> None:
#     #     if not isinstance(sampler, Sampler):
#     #         raise TypeError('sampler should be an instance of ``Sampler``, '
#     #                         f'but got {sampler}')
#     #     if not isinstance(batch_size, int) or batch_size <= 0:
#     #         raise ValueError('batch_size should be a positive integer value, '
#     #                          f'but got batch_size={batch_size}')
#     #     self.sampler = sampler
#     #     self.dataset = dataset
#     #     self.train_folder = train_folder
#     #     self.batch_size = batch_size
#     #     self.aspect_ratios = aspect_ratios
#     #     self.drop_last = drop_last

#     #     # buckets for each aspect ratio
#     #     self.current_available_bucket_keys = list(aspect_ratios.keys())
#     #     self.bucket = {
#     #         # 'image':{ratio: [] for ratio in aspect_ratios}, 
#     #         'video':{ratio: [] for ratio in aspect_ratios}
#     #     }

#     # def __iter__(self):
#     #     for idx in self.sampler:
#     #         try:
#     #             video_dict = self.dataset[idx]
#     #             width, height = video_dict.width, video_dict.height
#     #             height = int(height)
#     #             width = int(width)
#     #             ratio = height / width # self.dataset[idx]
#     #         except Exception as e:
#     #             print(e)
#     #             continue
#     #         # find the closest aspect ratio
#     #         closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
#     #         if closest_ratio not in self.current_available_bucket_keys:
#     #             continue
#     #         bucket = self.bucket['video'][closest_ratio]
#     #         bucket.append(idx)
#     #         # yield a batch of indices in the same aspect ratio group
#     #         if len(bucket) == self.batch_size:
#     #             yield bucket[:]
#     #             del bucket[:]

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
#                  sampler_ai2v: Sampler,
#                  sampler_i2v: Sampler,
#                  ai2v_dataset: Dataset,
#                  i2v_dataset: Dataset,
#                  batch_size: int,
#                  train_folder: str = None,
#                  aspect_ratios: dict = ASPECT_RATIO_512,
#                  drop_last: bool = False
#                 ) -> None:

#         if not isinstance(sampler_ai2v, Sampler):
#             raise TypeError('sampler should be an instance of ``Sampler``, '
#                             f'but got {sampler_ai2v}')

#         if not isinstance(sampler_i2v, Sampler):
#             raise TypeError('sampler should be an instance of ``Sampler``, '
#                             f'but got {sampler_i2v}')

#         if not isinstance(batch_size, int) or batch_size <= 0:
#             raise ValueError('batch_size should be a positive integer value, '
#                              f'but got batch_size={batch_size}')

#         self.sampler_ai2v = sampler_ai2v
#         self.sampler_i2v = sampler_i2v
        
#         self.ai2v_dataset = ai2v_dataset
#         self.i2v_dataset = i2v_dataset

#         self.train_folder = train_folder
#         self.batch_size = batch_size
#         self.aspect_ratios = aspect_ratios
#         self.drop_last = drop_last

#         # buckets for each aspect ratio
#         self.current_available_bucket_keys = list(aspect_ratios.keys())
#         self.bucket = {
#             # 'video_ai2v':{ratio: [] for ratio in aspect_ratios}, 
#             # 'video_i2v':{ratio: [] for ratio in aspect_ratios}
#             'video_multitask':{ratio: [] for ratio in aspect_ratios}
#         }

#         self.task_ai2v_prob = 1.0 # 1
#         self.task_i2v_prob = 0.0 # 0

#         assert self.task_ai2v_prob + self.task_i2v_prob == 1.0, "task_ai2v_prob + task_i2v_prob should be 1.0"

#     def __iter__(self):

#         iter_ai2v = iter(self.sampler_ai2v)
#         iter_i2v = iter(self.sampler_i2v)
        
#         while True:
#             if random.random() < self.task_ai2v_prob:
#                 task = 'ai2v'
#             elif random.random() < self.task_ai2v_prob + self.task_i2v_prob:
#                 task = 'i2v'

#             try:
#                 if task == 'ai2v':
#                     idx = next(iter_ai2v)
#                     video_dict = self.ai2v_dataset[idx]
#                 elif task == 'i2v':
#                     idx = next(iter_i2v)
#                     video_dict = self.i2v_dataset[idx]

#                 width, height = video_dict.width, video_dict.height
#                 height = int(height)
#                 width = int(width)
#                 ratio = height / width # self.dataset[idx]
#             except Exception as e:
#                 print(e)
#                 continue
#             # find the closest aspect ratio
#             closest_ratio = min(self.aspect_ratios.keys(), key=lambda r: abs(float(r) - ratio))
#             if closest_ratio not in self.current_available_bucket_keys:
#                 continue
#             # if task == 'ai2v':
#             #     bucket = self.bucket['video_ai2v'][closest_ratio]
#             # elif task == 'i2v':
#             #     bucket = self.bucket['video_i2v'][closest_ratio]
#             #     idx = idx + len(self.ai2v_dataset) # 实际dataset中i2v的索引
#             # else:
#             #     raise NotImplementedError
#             bucket = self.bucket['video_multitask'][closest_ratio]
#             if task == 'i2v':
#                 idx = idx + len(self.ai2v_dataset) # 实际dataset中i2v的索引
#             bucket.append(idx)
#             # yield a batch of indices in the same aspect ratio group
#             if len(bucket) == self.batch_size:
#                 yield bucket[:]
#                 del bucket[:]