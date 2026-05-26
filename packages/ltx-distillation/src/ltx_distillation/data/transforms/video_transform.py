import copy
import math
import random
import json
import scipy
import numpy as np
import torch
import os
# from vast.datasets import video_utils
# from vast.pipelines.vision.keypoints.pipeline_dwpose import draw_poses
# from vast.pipelines.vision.mask import get_box_frames, get_segment_frames
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F


class MaskGenerator:
    def __init__(self, mask_ratios):
        valid_mask_names = [
            "image_head",
            "image_tail",
            "image_head_tail",
            "image_random",
            "quarter_head",
            "quarter_tail",
            "quarter_head_tail",
            "quarter_random",
            "interpolation",
            "random",
            "identity",
        ]
        assert all(
            mask_name in valid_mask_names for mask_name in mask_ratios.keys()
        ), f"mask_name should be one of {valid_mask_names}, got {mask_ratios.keys()}"
        assert all(
            mask_ratio >= 0 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be greater than or equal to 0, got {mask_ratios.values()}"
        assert all(
            mask_ratio <= 1 for mask_ratio in mask_ratios.values()
        ), f"mask_ratio should be less than or equal to 1, got {mask_ratios.values()}"
        # sum of mask_ratios should be 1
        if "identity" not in mask_ratios:
            mask_ratios["identity"] = 1.0 - sum(mask_ratios.values())
        assert math.isclose(
            sum(mask_ratios.values()), 1.0, abs_tol=1e-6
        ), f"sum of mask_ratios should be 1, got {sum(mask_ratios.values())}"
        self.mask_ratios = mask_ratios

    def get_mask(self, num_frames):
        mask_type = random.random()
        mask_name = None
        prob_acc = 0.0
        for mask, mask_ratio in self.mask_ratios.items():
            prob_acc += mask_ratio
            if mask_type < prob_acc:
                mask_name = mask
                break
        condition_frames_max = max(1, num_frames // 4)
        mask = torch.ones(num_frames, dtype=torch.bool)
        if num_frames <= 1:
            return mask
        if mask_name == "image_head":
            random_size = 1
            mask[:random_size] = 0
        elif mask_name == "image_tail":
            random_size = 1
            mask[-random_size:] = 0
        elif mask_name == "image_head_tail":
            random_size = 1
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "image_random":
            random_size = 1
            random_pos = random.randint(0, num_frames - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "quarter_head":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
        elif mask_name == "quarter_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[-random_size:] = 0
        elif mask_name == "quarter_head_tail":
            random_size = random.randint(1, condition_frames_max)
            mask[:random_size] = 0
            mask[-random_size:] = 0
        elif mask_name == "quarter_random":
            random_size = random.randint(1, condition_frames_max)
            random_pos = random.randint(0, num_frames - random_size)
            mask[random_pos : random_pos + random_size] = 0
        elif mask_name == "interpolation":
            random_start = random.randint(0, 1)
            mask[random_start::2] = 0
        elif mask_name == "random":
            mask_ratio = random.uniform(0.1, 0.9)
            mask = torch.rand(num_frames) > mask_ratio
            # if mask is all False, set the last frame to True
            if not mask.any():
                mask[-1] = 1
        return mask


class VideoTransform:
    def __init__(
        self,
        size_method=1,
        dst_size=None,
        flip=False,
        num_frames=1,
        stride=1,
        cn_mode=None,
        mask_ratios=dict(),
    ):
        self.size_method = size_method
        self.dst_size = dst_size
        self.flip = flip
        self.num_frames = num_frames
        self.stride = stride
        self.cn_mode = cn_mode
        self.mask_generator = MaskGenerator(mask_ratios)
        self.normalize = transforms.Normalize([0.5], [0.5])

    def __call__(self, data_dict):
        num_frames, new_height, new_width, dst_height, dst_width = (
            self.get_new_height_width(data_dict)
        )
        flip = self.flip and torch.rand(1) < 0.5
        x1 = random.randint(0, new_width - dst_width)
        y1 = random.randint(0, new_height - dst_height)
        if "video" in data_dict:
            video = data_dict["video"]
            video_length = data_dict["video_length"]
            assert video_length <= len(video)
            sample_indexes = self.get_sample_indexes(data_dict, num_frames)
            input_images = video_utils.sample_video(video, sample_indexes, method=2)
        else:
            image = data_dict["image"]
            input_images = np.array(image)[None]
        input_images = torch.from_numpy(input_images).permute(0, 3, 1, 2).contiguous()
        height, width = input_images.shape[2], input_images.shape[3]
        if "video" in data_dict:
            # assert data_dict['video_height'] == height and data_dict['video_width'] == width
            pass
        else:
            assert (
                data_dict["image_height"] == height
                and data_dict["image_width"] == width
            )
        if flip:
            input_images = input_images.flip(-1)
        input_images = F.resize(
            input_images, (new_height, new_width), InterpolationMode.BILINEAR
        )
        input_images = F.crop(input_images, y1, x1, dst_height, dst_width)
        ref_images = copy.deepcopy(input_images)
        mask = self.mask_generator.get_mask(num_frames)[:, None, None, None]
        ref_images = ref_images * (mask < 0.5)
        input_images = input_images / 255.0
        ref_images = ref_images / 255.0
        input_images = self.normalize(input_images)
        ref_images = self.normalize(ref_images)
        data_dict["input_images"] = input_images
        data_dict["input_ref_images"] = ref_images

        # cn images
        # import pdb; pdb.set_trace()
        if self.cn_mode is not None and self.cn_mode in data_dict.keys():
            if self.cn_mode == "mask":
                cn = json.loads(data_dict["mask"])
                cns = get_segment_frames(
                    cn, height, width, data_dict["video_length"]
                )  # (T, H, W, 3)
                input_cn_images = []
                for idx in sample_indexes:
                    if idx > len(cns):
                        idx = -1  # 防止报错
                    input_cn_images.append(cns[idx])

            if self.cn_mode == "poses":
                cn = data_dict["poses"]
                # TODO: 对pose 施加随机扰动
                input_cn_images = [
                    draw_poses(
                        cn[idx],
                        height=height,
                        width=width,
                        draw_body=True,
                        draw_hand=True,
                        draw_face=True,
                    )
                    for idx in sample_indexes
                ]

            input_cn_images = np.stack(input_cn_images, axis=0)
            input_cn_images = (
                torch.from_numpy(input_cn_images).permute(0, 3, 1, 2).contiguous()
            )
            if flip:
                input_cn_images = input_cn_images.flip(-1)
            input_cn_images = F.resize(
                input_cn_images, (new_height, new_width), InterpolationMode.NEAREST
            )
            input_cn_images = F.crop(input_cn_images, y1, x1, dst_height, dst_width)
            input_cn_images = input_cn_images / 255.0
            input_cn_images = self.normalize(input_cn_images)
            data_dict["input_cn_images"] = input_cn_images
        else:
            assert self.cn_mode is None

        return data_dict

    def get_new_height_width(self, data_dict):
        if "video_height" in data_dict:
            height = data_dict["video_height"]
            width = data_dict["video_width"]
            num_frames = self.num_frames
        else:
            height = data_dict["image_height"]
            width = data_dict["image_width"]
            num_frames = 1
        if self.size_method == 1:
            dst_width, dst_height = self.dst_size
        elif self.size_method == 2:
            if len(data_dict["video_info"]) == 2:
                dst_width, dst_height = data_dict["video_info"]
            elif len(data_dict["video_info"]) == 3:
                dst_width, dst_height, num_frames = data_dict["video_info"]
            else:
                assert False
        else:
            assert False
        if float(dst_height) / height < float(dst_width) / width:
            new_height = int(round(float(dst_width) / width * height))
            new_width = dst_width
        else:
            new_height = dst_height
            new_width = int(round(float(dst_height) / height * width))
        assert dst_width <= new_width and dst_height <= new_height
        return num_frames, new_height, new_width, dst_height, dst_width

    def get_sample_indexes(self, data_dict, num_frames):
        if "video_valid_range" in data_dict:
            valid_range = data_dict["video_valid_range"]
            valid_range = [int(idx) for idx in valid_range]
        else:
            valid_range = (0, data_dict["video_length"])
        video_length = valid_range[1] - valid_range[0]
        assert num_frames <= video_length
        sample_length = min(video_length, (num_frames - 1) * self.stride + 1)
        start_idx = valid_range[0] + random.randint(0, video_length - sample_length)
        sample_indexes = np.linspace(
            start_idx, start_idx + sample_length - 1, num_frames, dtype=int
        )
        return sample_indexes
