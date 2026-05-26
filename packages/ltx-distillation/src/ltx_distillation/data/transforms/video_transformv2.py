import copy
import math
import random
import json
import scipy
import numpy as np
import torch
import os
from . import video_utils
import torch.nn.functional as F
import pickle as pkl
import cv2
import librosa

from transformers import  Wav2Vec2Processor

# import pyloudnorm as pyln
# from transformers import Wav2Vec2FeatureExtractor
from einops import rearrange
# from videox_fun.models.audio_wav2vec.wav2vec2 import Wav2Vec2Model

# from memory_profiler import profile

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


class GenerateRefImages:
    def __init__(self, mask_cfg=dict()):
        self.mask_generator = MaskGenerator(mask_cfg)

    def __call__(self, data_dict):
        ref_images = copy.deepcopy(data_dict["images"])
        num_frames = ref_images.shape[0]
        mask = self.mask_generator.get_mask(num_frames)[:, None, None, None]
        ref_images = ref_images * (mask < 0.5)
        data_dict["ref_images"] = ref_images # only one image remained
        return data_dict


class GenerateFirstRefImage:
    def __call__(self, data_dict):
        first_ref_image = copy.deepcopy(data_dict["images"][:1, ...])
        data_dict["first_ref_image"] = first_ref_image
        return data_dict

# class GenerateFirstFewRefImages:
#     def __call__(self, data_dict):
#         ref_num = random.randint(1, 15)
#         ref_images = copy.deepcopy(data_dict["images"][:ref_num, ...])

#         # first_ref_image = copy.deepcopy(data_dict["images"][:1, ...])
#         data_dict["first_ref_image"] = ref_images
#         return data_dict

class GenerateRepeatedFirstImage:
    def __call__(self, data_dict):
        first_ref_image = copy.deepcopy(data_dict["images"][:1, ...])
        data_dict["first_ref_image"] = first_ref_image
        return data_dict


class GeneratePoseControlImages:
    def __init__(self):
        pass


class GenerateWav2vec2FeatureOnline:

    def __init__(self):
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained("/gemini-1/space/human_guozz2/code/hys/fantasy_talking/models/wav2vec2-base-960h")
        # self.wav2vec = Wav2Vec2Model.from_pretrained("/gemini-1/space/human_guozz2/code/hys/fantasy_talking/models/wav2vec2-base-960h", use_safetensors=True)
        self.sr = 16000

    def __call__(self, data_dict):
        
        audio_input, sample_rate = librosa.load(data_dict["audio_raw_human_path"], sr=self.sr)  # 采样率为 16kHz
        start_frame_id, end_frame_id = data_dict["sample_indexes_audio"][0], data_dict["sample_indexes_audio"][-1]

        start_time = start_frame_id / 25.
        # end_time = (0 + (num_frames - 1) * 1) / fps
        end_time = end_frame_id / 25.

        start_sample = int(start_time * self.sr)
        end_sample = int(end_time * self.sr)

        audio_segment = audio_input[start_sample:end_sample]
        # if end_sample > audio_input.shape[0]:
        #     print(f"too short audio. {audio_segment.shape[0]} vs {end_sample}")
        #     audio_segment = np.pad(audio_segment, (0, end_sample - audio_input.shape[0]), 'constant')

        input_values = self.wav2vec_processor(
            audio_segment, sampling_rate=sample_rate, return_tensors="pt"
        ).input_values

        data_dict["wav2vec_input_values"] = input_values

        # with torch.no_grad():
        #     fea = self.wav2vec(input_values).last_hidden_state
        # data_dict["wav2vec_feature"] = fea
        return data_dict

class GenerateChineseWav2vec2FeatureOnline:

    def __init__(self, device="cpu"):
        wav2vec_dir = "models/chinese-wav2vec2-base"
        self.wav2vec_feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(wav2vec_dir, local_files_only=True)
        self.audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_dir, local_files_only=True).to(device)
        self.audio_encoder.feature_extractor._freeze_parameters()
        self.sample_rate = 16000
        self.device = device

    def loudness_norm(self, audio_array, sr=16000, lufs=-23):
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_array)
        if abs(loudness) > 100:
            return audio_array
        normalized_audio = pyln.normalize.loudness(audio_array, loudness, lufs)
        return normalized_audio

    def get_embedding(self, speech_array):
        extracted_fps = 25  # Assume the video fps is 25
        audio_duration = len(speech_array) / self.sample_rate
        video_length = audio_duration * extracted_fps

        # wav2vec_feature_extractor
        audio_feature = np.squeeze(
            self.wav2vec_feature_extractor(speech_array, sampling_rate=self.sample_rate).input_values
        )
        audio_feature = torch.from_numpy(audio_feature).float().to(self.device)

        audio_feature = audio_feature.unsqueeze(0) # [NOTE]: stack in train.py to form batch
        # audio encoder
        with torch.no_grad():
            embeddings = self.audio_encoder(audio_feature, seq_len=int(video_length), output_hidden_states=True)

        if len(embeddings) == 0:
            print("Fail to extract audio embedding")
            return None

        audio_emb = torch.stack(embeddings.hidden_states[1:], dim=1).squeeze(0)
        audio_emb = rearrange(audio_emb, "b s d -> s b d")

        audio_emb = audio_emb.cpu().detach()
        return audio_emb

    def __call__(self, data_dict):
        human_speech_array, sr = librosa.load(data_dict["audio_raw_human_path"], sr=self.sample_rate)
        human_speech_array = self.loudness_norm(human_speech_array, sr)

        start_frame_id, end_frame_id = data_dict["sample_indexes_audio"][0], data_dict["sample_indexes_audio"][-1]
        start_time = start_frame_id / 25.
        end_time = end_frame_id / 25.

        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        human_speech_array = human_speech_array[start_sample:end_sample]

        audio_embed = self.get_embedding(human_speech_array)
        data_dict["wav2vec_feature"] = audio_embed
        return data_dict

# class GenerateChineseWav2vecFeature:
#     def __init__(
#         self,
#         num_frames=1,
#     ):
#         self.num_frames = num_frames
#         # self.stride = stride

#     # @profile
#     def __call__(self, data_dict):
#         # [NOTE] currently only use human audio feature for lipsync
#         human_audio_feature = torch.from_numpy(np.load(data_dict["chinese_wav2vec_feature_path"]))
#         # # Get audio features extractly corresponding to the video frames
#         start_frame_id, end_frame_id = data_dict["sample_indexes_audio"][0], data_dict["sample_indexes_audio"][-1]

#         indices = (torch.arange(2 * 2 + 1) - 2) * 1 
#         center_indices = torch.linspace(
#             start_frame_id,
#             end_frame_id,
#             self.num_frames,
#         ).unsqueeze(
#             1
#         ) + indices.unsqueeze(0)
#         center_indices = torch.clamp(center_indices, min=0, max=human_audio_feature.shape[0]-1).int() # f, 5
#         audio_emb = human_audio_feature[center_indices]
#         # human_audio_feature_valid = human_audio_feature[start_frame_id:end_frame_id+1]
#         data_dict["wav2vec_embedding"] = audio_emb

#         if data_dict["audio_guidance_drop"]:
#             data_dict["wav2vec_embedding"] = torch.zeros_like(audio_emb)
#             # print("audio dropped.")

#         return data_dict

class GenerateChineseWav2vecFeature:
    def __init__(
        self,
        num_frames=1,
    ):
        self.num_frames = num_frames
        # self.stride = stride

    # @profile
    def __call__(self, data_dict):
        # [NOTE] currently only use human audio feature for lipsync
        human_audio_feature = torch.from_numpy(np.load(data_dict["chinese_wav2vec_feature_path"]))
        # # Get audio features extractly corresponding to the video frames
        indices = (torch.arange(2 * 2 + 1) - 2) * 1 
        center_indices = torch.from_numpy(data_dict["sample_indexes_audio"]).unsqueeze(1) + indices.unsqueeze(0)
        center_indices = torch.clamp(center_indices, min=0, max=human_audio_feature.shape[0]-1).int() # f, 5
        audio_emb = human_audio_feature[center_indices]
        # human_audio_feature_valid = human_audio_feature[start_frame_id:end_frame_id+1]
        data_dict["wav2vec_embedding"] = audio_emb

        if data_dict["audio_guidance_drop"]:
            data_dict["wav2vec_embedding"] = torch.zeros_like(audio_emb)
            # print("audio dropped.")

        return data_dict


class GenerateWhisperFeature:
    def __call__(self, data_dict):
        # [NOTE] currently only use human audio feature for lipsync
        human_audio_feature = torch.from_numpy(np.load(data_dict["audio_feature_path"].human))
        # Get audio features extractly corresponding to the video frames
        start_frame_id, end_frame_id = data_dict["sample_indexes_audio"][0], data_dict["sample_indexes_audio"][-1]
        human_audio_feature_valid = human_audio_feature[:, start_frame_id*2:end_frame_id*2+2]
        data_dict["human_audio_feature"] = human_audio_feature_valid.flatten(-2) # [b, f, 5*384]
        return data_dict

class GenerateWav2vecFeature:
    def __call__(self, data_dict):
        # [NOTE] currently only use human audio feature for lipsync
        human_audio_feature = torch.from_numpy(np.load(data_dict["wav2vec_feature_path"]))
        # # Get audio features extractly corresponding to the video frames
        start_frame_id, end_frame_id = data_dict["sample_indexes_audio"][0], data_dict["sample_indexes_audio"][-1]
        human_audio_feature_valid = human_audio_feature[:, start_frame_id*2:end_frame_id*2+2]
        data_dict["wav2vec_feature"] = human_audio_feature_valid
        return data_dict

class GenerateAudioFeatures:

    def linear_interpolation(self, features, input_fps=50, output_fps=60, output_len=None):
        features = features.transpose(1, 2)
        seq_len = features.shape[2] / float(input_fps)
        if output_len is None:
            output_len = int(seq_len * output_fps) 
        output_features = F.interpolate(features,size=output_len, align_corners=True, mode='linear')
        return output_features.transpose(1, 2)

    def __call__(self, data_dict):
        
        # human_audio_feature = np.load(data_dict["audio_feature_path"].human)
        # bg_audio_feature = np.load(data_dict["audio_feature_path"].background)
        # origin_audio_feature = np.load(data_dict["audio_feature_path"].origin)

        # [NOTE] currently only use human audio feature for lipsync
        human_audio_feature = torch.from_numpy(np.load(data_dict["audio_feature_path"].human))
        # human_audio_feature = torch.cat([torch.zeros_like(human_audio_feature[:,:4]), human_audio_feature, torch.zeros_like(human_audio_feature[:,:20])], 1) # 多padding一些, 避免fps精度转换损失

        # human_audio_feature: 1, 50fps*t, 5, 384 [NOTE: 0407后续数据都为25fps视频数据, 不用进行插值了]
        # human_audio_feature = human_audio_feature.reshape(human_audio_feature.shape[0],human_audio_feature.shape[1], -1)
        # human_audio_feature = self.linear_interpolation(human_audio_feature, input_fps=50, output_fps=data_dict['fps']*2) # fps transform: 50->60
        # human_audio_feature = human_audio_feature.reshape(human_audio_feature.shape[0],human_audio_feature.shape[1],5,384).contiguous()

        # Get audio features extractly corresponding to the video frames
        start_frame_id, end_frame_id = data_dict["sample_indexes_audio"][0], data_dict["sample_indexes_audio"][-1]
        human_audio_feature_valid = human_audio_feature[:, start_frame_id*2:end_frame_id*2+2]

        # human_audio_feature: 1, N, 5, 384
        pos_idx_ranges = self.split_audio_sequence(human_audio_feature_valid.shape[1], num_frames=len(data_dict["sample_indexes_audio"]))

        human_audio_feature_valid, audio_context_lens = self.split_tensor_with_padding(
            human_audio_feature_valid, pos_idx_ranges, expand_length=4
        )  # [b,21,9+8,5,384] # 专门计算了audio_context_lens

        # human_audio_feature_valid = human_audio_feature_valid.flatten(1,2).flatten(2) # [b,(21)*(9+8),(5*384)]
        human_audio_feature_valid = human_audio_feature_valid # .flatten(3) # [b,(21)*(9+8),(5*384)]

        # human_audio_feature_valid = []
        # for i in data_dict["sample_indexes_audio"]:
        #     # [1, N, 5, 384]
        #     audio_clip = human_audio_feature[:,i*2*1:i*2*1+10].unsqueeze(0) # 1, 1, 10, 5, 384
        #     human_audio_feature_valid.append(audio_clip)

        # human_audio_feature_valid = torch.cat(human_audio_feature_valid, dim=1) # b, f, w, 5, 384

        # [NOTE]: 由于不是全参微调，所以text相关的guidance都不需要学习了
        # if random.random() < 0.1: # for audio guidance
        #     human_audio_feature_valid[...] = 0.

        data_dict["human_audio_feature"] = human_audio_feature_valid
        # data_dict["human_audio_feature"] = torch.zeros(1, len(valid_range), 10, 5, 384)
        return data_dict


    def split_audio_sequence(self, audio_proj_length, num_frames=81):
        """
        Map the audio feature sequence to corresponding latent frame slices.

        Args:
            audio_proj_length (int): The total length of the audio feature sequence
                                    (e.g., 173 in audio_proj[1, 173, 768]).
            num_frames (int): The number of video frames in the training data (default: 81).

        Returns:
            list: A list of [start_idx, end_idx] pairs. Each pair represents the index range
                (within the audio feature sequence) corresponding to a latent frame.
        """
        # Average number of tokens per original video frame
        # 因为audio fps ≠ video fps， 所以需要计算每个video frame对应多少audio token
        tokens_per_frame = audio_proj_length / num_frames

        # Each latent frame covers 4 video frames, and we want the center
        tokens_per_latent_frame = tokens_per_frame * 4
        half_tokens = int(tokens_per_latent_frame / 2)

        # 对每个video token, 找到对应音频的中心token位置下标
        pos_indices = []
        for i in range(int((num_frames - 1) / 4) + 1):
            if i == 0:
                pos_indices.append(0)
            else:
                start_token = tokens_per_frame * ((i - 1) * 4 + 1)
                end_token = tokens_per_frame * (i * 4 + 1)
                center_token = int((start_token + end_token) / 2) - 1 # [NOTE]: 4个token的中间token, 有个-1?
                pos_indices.append(center_token)

        # Build index ranges centered around each position
        pos_idx_ranges = [[idx - half_tokens, idx + half_tokens] for idx in pos_indices]

        # Adjust the first range to avoid negative start index # [NOTE]：这里index0还是负的?
        pos_idx_ranges[0] = [
            -(half_tokens * 2 - pos_idx_ranges[1][0]),
            pos_idx_ranges[1][0],
        ]

        return pos_idx_ranges


    def split_tensor_with_padding(self, input_tensor, pos_idx_ranges, expand_length=0):
        """
        Split the input tensor into subsequences based on index ranges, and apply right-side zero-padding
        if the range exceeds the input boundaries.

        Args:
            input_tensor (Tensor): Input audio tensor of shape [1, L, 768].
            pos_idx_ranges (list): A list of index ranges, e.g. [[-7, 1], [1, 9], ..., [165, 173]].
            expand_length (int): Number of tokens to expand on both sides of each subsequence.

        Returns:
            sub_sequences (Tensor): A tensor of shape [1, F, L, 768], where L is the length after padding.
                                    Each element is a padded subsequence.
            k_lens (Tensor): A tensor of shape [F], representing the actual (unpadded) length of each subsequence.
                            Useful for ignoring padding tokens in attention masks.
        """
        pos_idx_ranges = [
            [idx[0] - expand_length, idx[1] + expand_length] for idx in pos_idx_ranges
        ] # 扩大了每个token的window
        sub_sequences = []
        seq_len = input_tensor.size(1)  # 173
        max_valid_idx = seq_len - 1  # 172
        k_lens_list = []
        # 对每个video token, 按照window提取有效audio窗口, 不足长度的部分补0
        for start, end in pos_idx_ranges:
            # Calculate the fill amount
            pad_front = max(-start, 0) # 如果前后超过正常范围, pad对应长度
            pad_back = max(end - max_valid_idx, 0)

            # Calculate the start and end indices of the valid part
            valid_start = max(start, 0)
            valid_end = min(end, max_valid_idx)

            # Extract the valid part
            if valid_start <= valid_end:
                valid_part = input_tensor[:, valid_start : valid_end + 1, :]
            else:
                # [NOTE]: 弄了一个shape [1, 0, C]的tensor, 什么情况会遇到这种情况?
                valid_part = input_tensor.new_zeros((1, 0, input_tensor.size(2), input_tensor.size(3)))

            # In the sequence dimension (the 1st dimension) perform padding
            padded_subseq = F.pad(
                valid_part,
                (0, 0, 0, 0, 0, pad_back + pad_front),
                mode="constant",
                value=0,
            )
            k_lens_list.append(padded_subseq.size(1) - pad_back - pad_front)

            sub_sequences.append(padded_subseq)
        return torch.stack(sub_sequences, dim=1), torch.tensor(
            k_lens_list, dtype=torch.long
        )

class GuidanceDropSelector:
    """
    Guidance drop selector
    """
    def __call__(self, data_dict):
        # i2v drop setting
        if data_dict['train_task'] == "i2v":
            data_dict["audio_guidance_drop"] = True
            text_drop_rate = 0.0
            selector = random.random()
            if selector < text_drop_rate:
                data_dict["text_guidance_drop"] = True
            else:
                data_dict["text_guidance_drop"] = False
        # ai2v drop setting
        elif data_dict['train_task'] == "ai2v":
            text_uncond_audio_cond_rate = 0.0 # 0
            text_uncond_audio_uncond_rate = 0.0 # 0.15
            text_cond_audio_uncond_rate = 0.0 # 0.15
            selector = random.random()
            if selector < text_uncond_audio_cond_rate:
                data_dict["text_guidance_drop"] = True
                data_dict["audio_guidance_drop"] = False
            elif selector < text_uncond_audio_cond_rate + text_uncond_audio_uncond_rate: # selector < text_uncond_audio_uncond_rate: 
                data_dict["text_guidance_drop"] = True
                data_dict["audio_guidance_drop"] = True
            elif selector < text_uncond_audio_cond_rate + text_uncond_audio_uncond_rate + text_cond_audio_uncond_rate:
                data_dict["text_guidance_drop"] = False
                data_dict["audio_guidance_drop"] = True
            else:
                data_dict["text_guidance_drop"] = False
                data_dict["audio_guidance_drop"] = False
        # print(f"text_guidance_drop: {data_dict['text_guidance_drop']} , audio_guidance_drop: {data_dict['audio_guidance_drop']}")
        return data_dict

class GenerateMotionIndexes:

    def __init__(
        self
    ):
        self.expr_edges = [0.006429346352849934,0.08551379849533265,0.09762220151941432,0.1050072009199342,0.11038047393477307,
                           0.11506373792996857,0.11926090177688667,0.12310524530352761,0.12650798961847215,0.12945997355100575,
                           0.1325360396962608,0.13545389820886908,0.13844715065145216,0.141004369124266,0.14426303274310742,
                           0.1470411141177052,0.14993755475241316,0.15271459872491377,0.15564254310904443,0.15844412751364853,
                           0.16129776841673008,0.16410843104975464,0.16712279705631125,0.1698633670299097,0.17284930674870297,
                           0.1758437499183848,0.17876769792774716,0.1818298815832017,0.18500640181061573,0.18804715804086627,
                           0.1910764791192397,0.19461191272266273,0.1980082410385049,0.2006744507988433,0.20416112770567785,
                           0.20774083019439818,0.21098239909298389,0.21456900928771513,0.21777949489709872,0.22100118205587185,
                           0.2243870997966398,0.22777320097829057,0.23128412921525812,0.23434352878937517,0.23774685525068392,
                           0.2409499272793058,0.24443997759970157,0.24789637251015822,0.251283217359208,0.2548796944893957,
                           0.2587110015722359,0.2625584003658159,0.26652372774472827,0.2703153599709967,0.2742450867110354,
                           0.27792738169025266,0.28162806339597984,0.2856475613315721,0.2896144350042191,0.29341353357863376,
                           0.2974492693358803,0.3013440968205975,0.30532178207784105,0.3097215154574408,0.3139058294546438,
                           0.3184158112148518,0.32284737097055577,0.3276479818150586,0.3322833425038498,0.33704168400773094,
                           0.34217001933241376,0.34704917950927955,0.35190293139675644,0.3566794477598765,0.3622329179307007,
                           0.3675328027132783,0.37292531420724195,0.3782416206656427,0.3836559970275134,0.38977282135082764,
                           0.3952428785225349,0.4016553150453851,0.40755377381845576,0.41370741722904286,0.4203261931408155,
                           0.42699976220478064,0.4341070597163165,0.4411592854044694,0.4479401379308138,0.45569307448513874,
                           0.4635404678454157,0.47120404929958387,0.48027417886213436,0.48897048884100336,0.4973801475171837,
                           0.5068336790031418,0.5158780815573234,0.5251518384125776,0.5355930301055288,0.5465275237640145,
                           0.5584753507122299,0.5709623881623037,0.5837638232615251,0.5963920995832415,0.6103472762082458,
                           0.6259649076584056,0.6416483976132302,0.6575825193698819,0.6731604314244369,0.6907422351176924,
                           0.7095891518905614,0.7323025674121094,0.7561481658918316,0.7822430276985393,0.8093752245136783,
                           0.8372420798168555,0.8711873925181345,0.9029566263714388,0.9457908162696684,0.9905904018840501,
                           1.041326992138139,1.0996791646439723,1.1704443857358797,1.263213441504366,1.381633050040954,
                           1.5303583625169614,1.7519620340211346,2.1849328443743534,12.034975940196864]

        self.head_edges = [5.200569090409127e-32,9.671435483767684e-08,2.4892887059322484e-07,4.569848525313419e-07,
                           7.492699749673722e-07,1.0485155126705188e-06,1.4364860764106342e-06,1.8256148002914287e-06,
                           2.232049272528429e-06,2.649213763466167e-06,3.0858606815615197e-06,3.497617192826489e-06,
                           3.89318553961831e-06,4.338290968347371e-06,4.742324901692376e-06,5.183316389101488e-06,
                           5.617745372163669e-06,6.0962545675783876e-06,6.5699074104090474e-06,7.08122783795812e-06,
                           7.52429602059228e-06,7.987264651999415e-06,8.499978094479046e-06,9.00945927103645e-06,
                           9.48918715923887e-06,1.002206153877218e-05,1.0566049307587644e-05,1.109377411476001e-05,
                           1.1661174988683458e-05,1.2217907132840465e-05,1.2751011196815746e-05,1.3326744341879958e-05,
                           1.3895783675881723e-05,1.4483188447137591e-05,1.5108272221316367e-05,1.565608851829929e-05,
                           1.6245628733471537e-05,1.692609543876387e-05,1.7534766391271217e-05,1.8194227093489754e-05,
                           1.885847016052757e-05,1.9608162373561475e-05,2.0305119860518462e-05,2.0989767643291664e-05,
                           2.17318778096283e-05,2.2483028955615657e-05,2.3252420059422437e-05,2.4039827485335723e-05,
                           2.4898633720005323e-05,2.569759940253063e-05,2.64967855781534e-05,2.7313989457196204e-05,
                           2.8160080825333952e-05,2.9077749405793876e-05,3.0003047323130787e-05,3.093057262038859e-05,
                           3.197709314742343e-05,3.294270192793348e-05,3.397063846587853e-05,3.5048157992158514e-05,
                           3.6192660360727975e-05,3.724558168535201e-05,3.842318828025642e-05,3.960168057927282e-05,
                           4.091097768141406e-05,4.218556841555804e-05,4.3468363487125345e-05,4.476798606629913e-05,
                           4.612244628015452e-05,4.748016437422167e-05,4.9001743181856855e-05,5.0497877474350174e-05,
                           5.212737783754048e-05,5.380737713720987e-05,5.5492643782160756e-05,5.7253302934048196e-05,
                           5.931050805700038e-05,6.117073118135662e-05,6.31278516937437e-05,6.479490317736955e-05,
                           6.68708539571412e-05,6.90623591426261e-05,7.134130197767043e-05,7.375020060984532e-05,
                           7.599405754975308e-05,7.823827514770703e-05,8.069950102373653e-05,8.342284042587579e-05,
                           8.595337445438296e-05,8.881732173027919e-05,9.195164994561046e-05,9.50428345087013e-05,
                           9.811752081433416e-05,0.00010141374029691981,0.00010460649530842704,0.00010820456376981105,
                           0.00011220273684597232,0.00011613660353916533,0.00012021453534215138,0.00012462252390722377,
                           0.00012937605105351296,0.00013484361898455398,0.00014022181045475663,0.00014584311815305897,
                           0.00015165936109029664,0.0001583329464848468,0.00016575928889453072,0.0001730672525109688,
                           0.00018135564500634262,0.00018983013295667355,0.00019892912791466674,0.00020838148946199656,
                           0.00021966679659912814,0.00023202092393492725,0.0002447766664662951,0.00025920079717134864,
                           0.0002763188685650623,0.00029518220793319335,0.0003163538267482184,0.0003397883273782967,
                           0.0003646995528677145,0.00040143058822463755,0.0004427568663705476,0.0004926301177587684,
                           0.0005611880917717915,0.0006489478638623473,0.0007857963640658107,0.001071003039707246,0.005421533858295953]

    def assign_bucket(self, value, edges):
        """为单个值分配桶索引"""
        if len(edges) < 2:
            return 0

        # 处理小于最小值的情况
        if value < edges[0]:
            return 0

        # 处理大于最大值的情况
        if value >= edges[-1]:
            return len(edges) - 2
        
        # 查找合适的桶
        for i in range(1, len(edges)):
            if edges[i-1] <= value < edges[i]:
                return i - 1

    # @profile
    def __call__(self, data_dict):
        # with open(file_path, 'rb') as f:
        #     variances_tracks = pkl.load(f)
        # motion_variances = torch.from_numpy(np.load(data_dict["audio_feature_path"].human))
        variance_expr = np.array(data_dict["variance_expr"])
        variance_head = np.array(data_dict["variance_head"])

        variance_expr = variance_expr[data_dict["sample_indexes_audio"]].mean()
        variance_head = variance_head[data_dict["sample_indexes_audio"]].mean()

        expr_bucket = self.assign_bucket(variance_expr, self.expr_edges)
        head_bucket = self.assign_bucket(variance_head, self.head_edges)

        data_dict["motion_indexes"] = torch.from_numpy(np.array([expr_bucket, head_bucket]))

        return data_dict


class GenerateAudioFeaturesSonic:

    # def linear_interpolation(self, features, input_fps=50, output_fps=60, output_len=None):
    #     features = features.transpose(1, 2)
    #     seq_len = features.shape[2] / float(input_fps)
    #     if output_len is None:
    #         output_len = int(seq_len * output_fps) 
    #     output_features = F.interpolate(features,size=output_len, align_corners=True, mode='linear')
    #     return output_features.transpose(1, 2)

    def __call__(self, data_dict):

        # human_audio_feature = np.load(data_dict["audio_feature_path"].human)
        # bg_audio_feature = np.load(data_dict["audio_feature_path"].background)
        # origin_audio_feature = np.load(data_dict["audio_feature_path"].origin)

        # [NOTE] currently only use human audio feature for lipsync
        human_audio_feature = torch.from_numpy(np.load(data_dict["audio_feature_path"].human))
        human_audio_feature = torch.cat([torch.zeros_like(human_audio_feature[:,:4]), human_audio_feature, torch.zeros_like(human_audio_feature[:,:6])], 1) # 多padding一些, 避免fps精度转换损失

        # human_audio_feature: 1, 50fps*t, 5, 384 [NOTE: 0407后续数据都为25fps视频数据, 不用进行插值了]
        # human_audio_feature = human_audio_feature.reshape(human_audio_feature.shape[0],human_audio_feature.shape[1], -1)
        # human_audio_feature = self.linear_interpolation(human_audio_feature, input_fps=50, output_fps=data_dict['fps']*2) # fps transform: 50->60
        # human_audio_feature = human_audio_feature.reshape(human_audio_feature.shape[0],human_audio_feature.shape[1],5,384).contiguous()

        # Get audio features extractly corresponding to the video frames
        # start_frame_id, end_frame_id = data_dict["sample_indexes_audio"][0], data_dict["sample_indexes_audio"][-1]
        # human_audio_feature_valid = human_audio_feature[:, start_frame_id*2:end_frame_id*2+2]

        # # human_audio_feature: 1, N, 5, 384
        # pos_idx_ranges = self.split_audio_sequence(human_audio_feature_valid.shape[1], num_frames=len(data_dict["sample_indexes_audio"]))

        # human_audio_feature_valid, audio_context_lens = self.split_tensor_with_padding(
        #     human_audio_feature_valid, pos_idx_ranges, expand_length=4
        # )  # [b,21,9+8,5,384] # 专门计算了audio_context_lens

        # human_audio_feature_valid = human_audio_feature_valid.flatten(1,2).flatten(2) # [b,(21)*(9+8),(5*384)]
        # human_audio_feature_valid = human_audio_feature_valid # .flatten(3) # [b,(21)*(9+8),(5*384)]

        human_audio_feature_valid = []
        for i in data_dict["sample_indexes_audio"]:
            # [1, N, 5, 384]
            audio_clip = human_audio_feature[:,i*2*1:i*2*1+10].unsqueeze(0) # 1, 1, 10, 5, 384
            human_audio_feature_valid.append(audio_clip)

        human_audio_feature_valid = torch.cat(human_audio_feature_valid, dim=1) # b, f, w, 5, 384

        if random.random() < 0.1: # for audio guidance
            data_dict["audio_scale"] = torch.tensor([0.])
        else:
            data_dict["audio_scale"] = torch.tensor([1.])

        data_dict["human_audio_feature"] = human_audio_feature_valid
        # data_dict["human_audio_feature"] = torch.zeros(1, len(valid_range), 10, 5, 384)
        return data_dict

class SampleImages:
    def __init__(
        self,
        num_frames=1,
        stride=None,
        sample_type="continuous",
    ):
        self.num_frames = num_frames
        self.stride = stride
        self.sample_type = sample_type

    def __call__(self, data_dict):
        video = data_dict["video"]
        sample_indexes = self.get_sample_indexes(data_dict, self.num_frames, self.sample_type)
        images = video_utils.sample_video(video, sample_indexes, method=2)
        images = torch.from_numpy(images).permute(0, 3, 1, 2).contiguous()
        data_dict["images"] = images
        data_dict["sample_indexes_audio"] = np.clip(sample_indexes - data_dict["audio_video_offset"], 0,  data_dict["video_length"]-1).astype(np.int32) # 范围截断
        return data_dict

    def get_sample_indexes(self, data_dict, num_frames, sample_type='continuous'):
        if "video_valid_range" in data_dict:
            valid_range = data_dict["video_valid_range"]
            valid_range = [int(idx) for idx in valid_range]
        else:
            valid_range = (0, data_dict["video_length"])

        if "audio_valid_range" in data_dict:
            audio_valid_range = data_dict["audio_valid_range"]
            # if audio_valid_range[0] != -1: # it should not appear
            audio_valid_range = [int(idx) for idx in audio_valid_range]
            # audio_valid_range = [math.ceil(audio_valid_range[0]*data_dict['fps']/25), math.floor(audio_valid_range[1]*data_dict['fps']/25)] # fps transform
            valid_range = [max(valid_range[0], audio_valid_range[0]), \
                            min(valid_range[1], audio_valid_range[1])] # audio valid range right closed
            # double check video range (audio here supposed to be normally sampled)
        if self.stride:
            stride = self.stride
        else:
            stride = max(math.ceil(data_dict.get("fps", 25) / 15), 1)

        if sample_type == 'continuous':
            video_length = valid_range[1] - valid_range[0] - 5 # # [NOTE]: 跳过后5帧
            # assert num_frames <= video_length
            sample_length = min(video_length, math.ceil((num_frames - 1) * stride) + 1)
            start_idx = valid_range[0] + random.randint(0, video_length - sample_length) + 5 # [NOTE]: 跳过前5帧
            sample_indexes = np.linspace(
                start_idx, min(start_idx + sample_length - 1, data_dict["video_length"]-1), num_frames, dtype=int) # [NOTE]: 修改保证video sample不超过video长度
        elif sample_type == "jump_chunks_from_first": # for long video generatio
            seg1_len = 9
            seg2_len = num_frames - 9
            jump_chunk_size = 12
            max_jump_chunk_num = 10
            SKIP_HEAD = 5
            SKIP_TAIL = 5
            usable_start = valid_range[0] + SKIP_HEAD
            # valid_range[1] 一般代表右端（原代码对 video_length - 1 有额外限制），这里把 usable_end 视作包含的最后帧索引
            usable_end = valid_range[1] - 1 - SKIP_TAIL
            # 保护性处理：确保 usable_end 不超出真实视频长度
            max_frame_idx = int(data_dict.get("video_length", 0)) - 1
            usable_end = min(usable_end, max_frame_idx - SKIP_TAIL)  # 再次确保安全
            # 确保采样的total_min比可用span大
            total_min = math.ceil(num_frames * stride)  # k=0 时占用帧数（两段相邻）
            usable_span = usable_end - usable_start + 1
            if usable_end < usable_start or usable_span < total_min:
                print("usable interval too small, back to continuous sampling.")
                return self.get_sample_indexes(data_dict, num_frames, sample_type='continuous')
            else:
                # 还有余量可以分配给 gap（以帧为单位）
                max_gap_frames = usable_span - total_min
                max_k = min(max_gap_frames // math.ceil(jump_chunk_size * stride), max_jump_chunk_num)  # k 的最大值
                k = random.randint(0, max_k)  # 随机选择 k（包含 0）
                gap_frames = k * math.ceil(jump_chunk_size * stride)
                # 对于固定 gap_frames，start1 的合法范围：
                # 设 s1 为第一段起点，则要求 s1 >= usable_start 且
                # s1 + (seg1_len - 1) + 1 + gap_frames + (seg2_len - 1) <= usable_end
                # => s1 <= usable_end - (seg1_len + gap_frames + seg2_len - 1)
                s1_min = usable_start
                s1_max = usable_end - (math.ceil(seg1_len * stride) + gap_frames + math.ceil(seg2_len * stride) - 1)
                if s1_max < s1_min:
                    # 理论上不应该出现（因为 max_k 计算保证了可行性），但再做保险处理
                    s1_max = s1_min
                start1 = random.randint(s1_min, s1_max)
                start2 = start1 + math.ceil(seg1_len * stride) + gap_frames  # 因为 start2 = last_index_of_seg1 + 1 + gap_frames
                if start2 + math.ceil(seg2_len * stride) - 1 > usable_end:
                    print("usable interval too small, back to continuous sampling.")
                    return self.get_sample_indexes(data_dict, num_frames, sample_type='continuous')
                else:
                    # 构造两个连续段（步长为1）
                    sample_length1 = math.ceil((seg1_len - 1) * stride) + 1
                    sample_length2 = math.ceil((seg2_len - 1) * stride) + 1
                    seg1_idx = np.linspace(
                        start1, min(start1 + sample_length1 - 1, data_dict["video_length"]-1), seg1_len, dtype=int)
                    seg2_idx = np.linspace(
                        start2, min(start2 + sample_length2 - 1, data_dict["video_length"]-1), seg2_len, dtype=int)
                    sample_indexes = np.concatenate([seg1_idx, seg2_idx]).astype(int)
            assert len(sample_indexes) == num_frames
        else:
            raise NotImplementedError

        return sample_indexes

class SampleFaceBoundingBoxesDummy:

    # @profile
    def __call__(self, data_dict):
        face_masks = torch.zeros_like(data_dict["images"])
        face_masks[:, :, :, :] = 1 # [NOTE]: 试验全部mask, TODO: delete
        data_dict["face_masks"] = face_masks
        # if face_masks.sum() == 0:
        #     print("mask zero in dataloader...")
        return data_dict


class SampleFaceBoundingBoxes:

    def get_union_bbox(self, bbox_list):
        """
        输入: bbox_list 是一个 list，里面的每个元素是一个 list，格式为 [x1, y1, x2, y2]
        输出: 并集区域的 [x1, y1, x2, y2]
        """
        if not bbox_list:
            return None  # 或者 raise ValueError("空的 bbox 列表")

        # 拆分所有坐标
        x1_list = [bbox[0] for bbox in bbox_list]
        y1_list = [bbox[1] for bbox in bbox_list]
        x2_list = [bbox[2] for bbox in bbox_list]
        y2_list = [bbox[3] for bbox in bbox_list]

        # 分别取极值
        x1_union = min(x1_list)
        y1_union = min(y1_list)
        x2_union = max(x2_list)
        y2_union = max(y2_list)

        return [x1_union, y1_union, x2_union, y2_union]

    def __call__(self, data_dict):
        # if "audio_video_offset" in data_dict:
        #     sample_indexes_video = data_dict["sample_indexes_audio"] + data_dict["audio_video_offset"]
        # else:
        #     sample_indexes_video = data_dict["sample_indexes_audio"]
        
        # with open(data_dict["face_info_path"], 'rb') as f:
        #     pass

        face_masks = torch.zeros_like(data_dict["images"])
        audio_bbox = data_dict["audio_bbox"]
        bbox_sample_offseted = data_dict["sample_indexes_audio"] - data_dict['audio_valid_range'][0]
        audio_bbox_sampled = [audio_bbox[i] for i in bbox_sample_offseted]
        x1, y1, x2, y2 = self.get_union_bbox(audio_bbox_sampled)
        x1, y1, x2, y2 = self.resize_bbox(x1, y1, x2, y2, 1.75)
        x1, y1, x2, y2 = max(int(x1), 0), max(int(y1), 0), max(int(x2), 0), max(int(y2), 0)
        face_masks[:, :, y1:y2, x1:x2] = 1
        # face_masks[:, :, :, :] = 1 # [NOTE]: 试验全部mask, TODO: delete
        data_dict["face_masks"] = face_masks
        if face_masks.sum() == 0:
            print("mask zero in dataloader...")
        # import cv2
        # data_dict["images"][:, :, y1:y2, x1:x2] = 1
        # cv2.imwrite("test.jpg", data_dict["images"][0].permute(1,2,0).numpy())
        return data_dict

    def resize_bbox(self, x1, y1, x2, y2, R):
        # 计算原始中心点
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # 计算新的宽度和高度
        new_width = (x2 - x1) * R
        new_height = (y2 - y1) * R
        
        # 计算新的角点坐标
        x1_new = cx - new_width / 2
        y1_new = cy - new_height / 2
        x2_new = cx + new_width / 2
        y2_new = cy + new_height / 2
        
        return int(x1_new), int(y1_new), int(x2_new), int(y2_new)

class SampleLipMasks:

    def __init__(
        self
    ):
        self.outer_lip_index = list(range(48, 60))
        self.face_index = list(range(1, 16))

    def __call__(self, data_dict):
        # Load kps
        data_path = str(data_dict["data_path"])
        landmarks_dir = os.path.join(data_path, "v0.0.1", "face", "landmarks")
        landmark_info_path = os.path.join(landmarks_dir, str(data_dict["clip_info"].id)+".pkl")
        with open(landmark_info_path, "rb") as f:
            landmark_info = pkl.load(f)
        kps_all = landmark_info[data_dict["track_id"]]['kps']
        kps_all_conf = landmark_info[data_dict["track_id"]]['kps_score']

        # Sample indexes
        sample_indexes_offseted = data_dict["sample_indexes_audio"] - data_dict['audio_valid_range'][0]
        kps_all = [kps_all[i] for i in sample_indexes_offseted]
        kps_all_conf = [kps_all_conf[i] for i in sample_indexes_offseted]

        # Sample bboxes as substitution
        audio_bbox = data_dict["audio_bbox"]
        audio_bbox_sampled = [audio_bbox[i] for i in sample_indexes_offseted]

        # get lip masks
        lip_masks = torch.zeros_like(data_dict["images"])
        f, channels, H, W = lip_masks.shape

        # 膨胀核大小与迭代次数
        kernel_size = 40
        num_iterations = 1
        kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)

        for i, (lm, lm_conf, bbox) in enumerate(zip(kps_all, kps_all_conf, audio_bbox_sampled)):

            use_full_mask = False
            lip_kps_cur = lm[self.outer_lip_index]
            lip_kps_conf_cur = lm_conf[self.outer_lip_index]
            # face_kps_cur = lm[self.face_index]
            # face_kps_conf_cur = lm_conf[self.face_index]

            if lip_kps_conf_cur.min() > 0.6:
                kps_cur = lip_kps_cur
            else:
                x1, y1, x2, y2 = bbox
                face_kps = [[x1, y1], [x1, y2], [x2, y2], [x2, y1]]
                kps_cur = np.array(face_kps)

            # lm: (12,2) 的浮点坐标，先转为整数
            pts = np.round(kps_cur).astype(np.int32)
            # 构造单通道的 numpy mask
            mask_np = np.zeros((H, W), dtype=np.uint8)
            # 用多边形填充（注意 OpenCV 要求 pts 必须是形状 (n,1,2) 或列表形式）
            cv2.fillPoly(mask_np, [pts], color=1)
            # 膨胀操作，扩大区域
            mask_np = cv2.dilate(mask_np, kernel, iterations=num_iterations)
            # 转回 torch，并扩展到 3 通道
            m = torch.from_numpy(mask_np).float()            # (H, W)
            m = m.unsqueeze(0).unsqueeze(0)                  # (1, 1, H, W)
            m = m.expand(1, channels, H, W)                  # (1, C, H, W)
            
            # 将第 i 帧的 mask 写入
            lip_masks[i:i+1, :, :, :] = m

        data_dict["lip_masks"] = lip_masks

        return data_dict


"""
import torch
import torchvision.transforms.functional as F
from PIL import Image, ImageDraw, ImageFont

def draw_points_with_index(image_tensor, points, save_path):
    
    # Args:
    #     image_tensor (torch.Tensor): (3, H, W) 图像，像素值范围 0-1。
    #     points (torch.Tensor): (68, 2) 点坐标，xy顺序。
    #     save_path (str): 保存路径。
    
    # 转成 PIL Image
    image_pil = F.to_pil_image(image_tensor.cpu().clamp(0, 1))
    draw = ImageDraw.Draw(image_pil)
    
    # 可以选择字体，这里用默认字体
    try:
        font = ImageFont.truetype("arial.ttf", size=12)
    except:
        font = ImageFont.load_default()

    for idx, (x, y) in enumerate(points):
        radius = 2
        # 画点
        leftUpPoint = (x - radius, y - radius)
        rightDownPoint = (x + radius, y + radius)
        draw.ellipse([leftUpPoint, rightDownPoint], fill=(255, 0, 0))
        # 写序号，稍微偏移一点，避免遮住点
        draw.text((x + 3, y - 3), str(idx), fill=(0, 255, 0), font=font)

    image_pil.save(save_path)

draw_points_with_index(lip_masks[10], kps_all[10], 'output_with_index.png')

"""