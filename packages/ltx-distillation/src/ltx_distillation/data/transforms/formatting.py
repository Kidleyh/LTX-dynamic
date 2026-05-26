from typing import Sequence, Union
import numpy as np
import torch
# from vast.pipelines.vision.keypoints.pipeline_dwpose import draw_poses
import random
# from vast.pipelines.vision.mask import get_box_frames, get_segment_frames
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


def is_seq_of(seq, expected_type):
    """
    检查给定的序列是否包含特定类型的元素。

    :param seq: 要检查的序列
    :param expected_type: 期望的元素类型
    :return: 如果所有元素都是指定类型，则返回True，否则返回False
    """
    if not isinstance(seq, (list, tuple)):
        return False
    return all(isinstance(item, expected_type) for item in seq)


class PackInputs:
    def __init__(self, image_keys, embedding_keys, dst_size, mean=0.5, std=0.5) -> None:
        self.dst_size = dst_size
        self.image_keys = image_keys
        self.embedding_keys = embedding_keys
        self.mean = mean
        self.std = std

    def __call__(self, data_dict):
        # ultimately resize and crop to dst
        data_dict = self.resize_and_crop(data_dict)
        input_dict = dict()
        input_dict["prompt"] = data_dict["prompt"]
        for image_key in self.image_keys:
            input_dict[image_key] = (
                (data_dict[image_key] / 255.0) - self.mean
            ) / self.std # -1, 1
        for embed_key in self.embedding_keys:
            input_dict[embed_key] = data_dict[embed_key]
        return input_dict

    def resize_and_crop(self, data_dict):
        new_height, new_width, dst_height, dst_width = self.get_new_height_width(
            data_dict
        )
        x1 = random.randint(0, new_width - dst_width)
        y1 = random.randint(0, new_height - dst_height)
        for image_key in self.image_keys:
            images = data_dict[image_key]
            images = F.resize(
                images, (new_height, new_width), InterpolationMode.BILINEAR
            )
            # 通过这个函数, video最终变成dst wh的大小
            images = F.crop(images, y1, x1, dst_height, dst_width)
            data_dict[image_key] = images
        return data_dict

    def get_new_height_width(self, data_dict):
        height = data_dict["video_height"]
        width = data_dict["video_width"]
        dst_width, dst_height = data_dict["video_info"]
        # new wh长宽比和video原始wh一致，但长或宽其中一边长度和dst wh对齐
        if float(dst_height) / height < float(dst_width) / width:
            new_height = int(round(float(dst_width) / width * height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / height * width))
        return new_height, new_width, dst_height, dst_width
