import gc
import logging

from utils.dataset import ShardingLMDBDataset, cycle
from utils.dataset import TextDataset
from utils.distributed import EMA_ACCELERATE, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from utils.misc import (
    set_seed,
    merge_dict_list
)
import torch.distributed as dist
from omegaconf import OmegaConf
from model import CausVid, DMD, SiD, CausVid_AI2V, DMDAI2V_CausVid, DMDAI2V_SelfForcing
import torch
import wandb
import time
import os

# new
from wan.data.dataset_audio_visual_mulres_chinesewav2vec_multask_mulref import AudioVisualDataset, get_random_mask
from wan.data.bucket_sampler import (ASPECT_RATIO_512, 
                                     ASPECT_RATIO_RANDOM_CROP_512, 
                                     AspectRatioBatchSingletaskVideoSampler, 
                                     AspectRatioBatchMultitaskVideoSampler,
                                     get_closest_ratio)
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import random
from functools import partial
from torchvision import transforms
from einops import rearrange
from utils.utils_vxf import save_videos_grid
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# from diffusers.training_utils import EMAModel
import math
from diffusers.optimization import get_scheduler
from torchvision.io import write_video

# from third_party.pytorchface.face_id_loss import FaceIdLoss
from third_party.pytorchface.face_tools import FaceAnalysis, Face
import cv2

import trainer.reward.reward_fn as reward_fn
import json
# import random

def resize_mask(mask, latent, process_first_frame_only=True):
    latent_size = latent.size()
    batch_size, channels, num_frames, height, width = mask.shape

    if process_first_frame_only:
        target_size = list(latent_size[2:])
        target_size[0] = 1
        first_frame_resized = F.interpolate(
            mask[:, :, 0:1, :, :],
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
        
        target_size = list(latent_size[2:])
        target_size[0] = target_size[0] - 1
        if target_size[0] != 0:
            remaining_frames_resized = F.interpolate(
                mask[:, :, 1:, :, :],
                size=target_size,
                mode='trilinear',
                align_corners=False
            )
            resized_mask = torch.cat([first_frame_resized, remaining_frames_resized], dim=2)
        else:
            resized_mask = first_frame_resized
    else:
        target_size = list(latent_size[2:])
        resized_mask = F.interpolate(
            mask,
            size=target_size,
            mode='trilinear',
            align_corners=False
        )
    return resized_mask

def collate_fn(examples, args=None):
    
    def get_length_to_frame_num(token_length):
        if args.image_sample_size > args.video_sample_size:
            sample_sizes = list(range(args.video_sample_size, args.image_sample_size + 1, 128))

            if sample_sizes[-1] != args.image_sample_size:
                sample_sizes.append(args.image_sample_size)
        else:
            sample_sizes = [args.image_sample_size]
        
        length_to_frame_num = {
            sample_size: min(token_length / sample_size / sample_size, args.video_sample_n_frames) // args.vae_temporal_compression_ratio * args.vae_temporal_compression_ratio + 1 for sample_size in sample_sizes
        }

        return length_to_frame_num

    def get_random_downsample_ratio(sample_size, image_ratio=[],
                                    all_choices=False, rng=None):
        def _create_special_list(length):
            if length == 1:
                return [1.0]
            if length >= 2:
                first_element = 0.90
                remaining_sum = 1.0 - first_element
                other_elements_value = remaining_sum / (length - 1)
                special_list = [first_element] + [other_elements_value] * (length - 1)
                return special_list
                
        if sample_size >= 1536:
            number_list = [1, 1.25, 1.5, 2, 2.5, 3] + image_ratio 
        elif sample_size >= 1024:
            number_list = [1, 1.25, 1.5, 2] + image_ratio
        elif sample_size >= 768:
            number_list = [1, 1.25, 1.5] + image_ratio
        elif sample_size >= 512:
            number_list = [1] + image_ratio
        else:
            number_list = [1]

        if all_choices:
            return number_list

        number_list_prob = np.array(_create_special_list(len(number_list)))
        if rng is None:
            return np.random.choice(number_list, p = number_list_prob)
        else:
            return rng.choice(number_list, p = number_list_prob)

    # Get token length
    target_token_length = args.video_sample_n_frames * args.token_sample_size * args.token_sample_size
    length_to_frame_num = get_length_to_frame_num(target_token_length)

    # Create new output
    new_examples                 = {}
    new_examples["target_token_length"] = target_token_length
    new_examples["pixel_values"] = []
    new_examples["text"]         = []
    new_examples["wav2vec_embedding"] = []
    new_examples["face_mask"]    = []
    # new_examples["lip_masks"]    = []
    # Used in Inpaint mode 
    if args.train_mode != "normal":
        new_examples["mask_pixel_values"] = []
        new_examples["mask"] = []
        new_examples["clip_pixel_values"] = []

    # Get downsample ratio in image and videos
    pixel_value     = examples[0]["images"]
    # data_type       = examples[0]["data_type"]
    f, c, h, w      = pixel_value.shape
    # if data_type == 'image':
    #     random_downsample_ratio = 1 if not args.random_hw_adapt else get_random_downsample_ratio(args.image_sample_size, image_ratio=[args.image_sample_size / args.video_sample_size])

    #     aspect_ratio_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
    #     aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.image_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}
        
    #     batch_video_length = args.video_sample_n_frames + sample_n_frames_bucket_interval
    # else:
    if args.random_hw_adapt: # False
        if args.training_with_video_token_length:
            local_min_size = np.min(np.array([np.mean(np.array([np.shape(example["images"])[1], np.shape(example["images"])[2]])) for example in examples]))
            # The video will be resized to a lower resolution than its own.
            choice_list = [length for length in list(length_to_frame_num.keys()) if length < local_min_size * 1.25]
            if len(choice_list) == 0:
                choice_list = list(length_to_frame_num.keys())
            local_video_sample_size = np.random.choice(choice_list)
            batch_video_length = length_to_frame_num[local_video_sample_size]
            random_downsample_ratio = args.video_sample_size / local_video_sample_size
        else:
            random_downsample_ratio = get_random_downsample_ratio(args.video_sample_size)
            batch_video_length = args.video_sample_n_frames + args.vae_temporal_compression_ratio
    else:
        random_downsample_ratio = 1
        batch_video_length = args.video_sample_n_frames + args.vae_temporal_compression_ratio

    aspect_ratio_sample_size = {key : [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
    aspect_ratio_random_crop_sample_size = {key : [x / 512 * args.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_RANDOM_CROP_512[key]] for key in ASPECT_RATIO_RANDOM_CROP_512.keys()}

    closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
    closest_size = [int(x / 16) * 16 for x in closest_size]

    # [NOTE]: by ys
    assert not args.random_ratio_crop

    if args.random_ratio_crop: # False
        random_sample_size = aspect_ratio_random_crop_sample_size[
            np.random.choice(list(aspect_ratio_random_crop_sample_size.keys()), p = ASPECT_RATIO_RANDOM_CROP_PROB)
        ]
        random_sample_size = [int(x / 16) * 16 for x in random_sample_size]

    for example in examples:
        if args.random_ratio_crop: # False
            # To 0~1
            pixel_values = torch.from_numpy(example["images"]).permute(0, 3, 1, 2).contiguous()
            pixel_values = pixel_values / 255.

            # Get adapt hw for resize
            b, c, h, w = pixel_values.size()
            th, tw = random_sample_size
            if th / tw > h / w:
                nh = int(th)
                nw = int(w / h * nh)
            else:
                nw = int(tw)
                nh = int(h / w * nw)
            
            transform = transforms.Compose([
                transforms.Resize([nh, nw]),
                transforms.CenterCrop([int(x) for x in random_sample_size]),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])
            # print("Error! Not support random crop!")
        else:
            # To 0~1
            pixel_values = example["images"].contiguous() # torch.from_numpy(example["images"]).permute(0, 3, 1, 2).contiguous()
            # pixel_values = pixel_values / 255.

            # Get adapt hw for resize
            closest_size = list(map(lambda x: int(x), closest_size))
            if closest_size[0] / h > closest_size[1] / w:
                resize_size = closest_size[0], int(w * closest_size[0] / h)
            else:
                resize_size = int(h * closest_size[1] / w), closest_size[1]
            
            # TODO: 可能没必要再resize一遍了
            transform = transforms.Compose([
                transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
                transforms.CenterCrop(closest_size),
                # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ])

            face_masks = example["face_masks"]
            face_masks = torch.round((face_masks + 1.) / 2. * 255)

            # lip_masks = example["lip_masks"]
            # lip_masks = torch.round((lip_masks + 1.) / 2. * 255)

        new_examples["pixel_values"].append(transform(pixel_values))
        new_examples["wav2vec_embedding"].append(example["wav2vec_embedding"])
        new_examples["text"].append(example["prompt"])
        new_examples["face_mask"].append(transform(face_masks))
        # new_examples["lip_masks"].append(transform(lip_masks))

        batch_video_length = int(min(batch_video_length, len(pixel_values)))

        # Magvae needs the number of frames to be 4n + 1.
        batch_video_length = (batch_video_length - 1) // args.vae_temporal_compression_ratio * args.vae_temporal_compression_ratio + 1

        if batch_video_length <= 0:
            batch_video_length = 1

        if args.train_mode != "normal":
            mask = get_random_mask(new_examples["pixel_values"][-1].size(), image_start_only=False) # [NOTE]: by hys, I2V first few frames
            mask_pixel_values = new_examples["pixel_values"][-1] * (1 - mask) 
            # Wan 2.1 use 0 for masked pixels
            # + torch.ones_like(new_examples["pixel_values"][-1]) * -1 * mask
            new_examples["mask_pixel_values"].append(mask_pixel_values)
            new_examples["mask"].append(mask)

            clip_pixel_values = new_examples["pixel_values"][-1][0].permute(1, 2, 0).contiguous()
            clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
            new_examples["clip_pixel_values"].append(clip_pixel_values)

    # Limit the number of frames to the same
    new_examples["pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["pixel_values"]])
    # latent_length = (batch_video_length - 1) // 4 + 1 # Audio feature is latent-wise time
    new_examples["wav2vec_embedding"] = torch.stack([example[:batch_video_length] for example in new_examples["wav2vec_embedding"]])
    new_examples["face_mask"] = torch.stack([example[:batch_video_length, :1] for example in new_examples["face_mask"]]) # remain one channel
    # new_examples["lip_masks"] = torch.stack([example[:batch_video_length, :1] for example in new_examples["lip_masks"]]) # remain one channel

    if args.train_mode != "normal": # inp
        new_examples["mask_pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["mask_pixel_values"]])
        new_examples["mask"] = torch.stack([example[:batch_video_length] for example in new_examples["mask"]])
        new_examples["clip_pixel_values"] = torch.stack([example for example in new_examples["clip_pixel_values"]])

    # Encode prompts when enable_text_encoder_in_dataloader=True
    # if False: # args.enable_text_encoder_in_dataloader: # False
    #     prompt_ids = tokenizer(
    #         new_examples['text'], 
    #         max_length=args.tokenizer_max_length, 
    #         padding="max_length", 
    #         add_special_tokens=True, 
    #         truncation=True, 
    #         return_tensors="pt"
    #     )
    #     encoder_hidden_states = text_encoder(
    #         prompt_ids.input_ids
    #     )[0]
    #     new_examples['encoder_attention_mask'] = prompt_ids.attention_mask
    #     new_examples['encoder_hidden_states'] = encoder_hidden_states

    return new_examples


class AI2VIDScoreDistillationTrainer:
    def __init__(self, config, accelerator):

        # id_loss_handler = FaceIdLoss(device=accelerator.device)

        # test_img_batch_t = cv2.imread("/gemini/platform/public/aigc/human_guozz2/code/hys/wan22/examples/examples/imgs/chinese_squre.png")
        # test_img_batch_t = torch.from_numpy(test_img_batch_t).unsqueeze(0).repeat(81,1,1,1).permute(0,3,1,2)
        # test_img_batch_t = test_img_batch_t.to(accelerator.device)
        # test_img_batch_t1 = (test_img_batch_t / 255.) * 2 - 1

        # test_img_batch_t = cv2.imread("/gemini/platform/public/aigc/human_guozz2/code/hys/longtalker_0319/asset/test2_169.jpg")
        # test_img_batch_t = torch.from_numpy(test_img_batch_t).unsqueeze(0).repeat(81,1,1,1).permute(0,3,1,2)
        # test_img_batch_t = test_img_batch_t.to(accelerator.device)
        # test_img_batch_t2 = (test_img_batch_t / 255.) * 2 - 1

        ### images_batch_t: [-1, 1], B*T, 3, H, W
        # id_loss_handler.face_id_loss(test_img_batch_t1, test_img_batch_t2)

        # reward loss function
        if config.enable_reward_loss:
            reward_fn_kwargs = {}
            if config.reward_fn_kwargs is not None:
                reward_fn_kwargs = config.reward_fn_kwargs # json.loads(config.reward_fn_kwargs)
            # if accelerator.is_main_process:
            #     # Check if the model is downloaded in the main process.
            #     loss_fn = getattr(reward_fn, config.reward_fn)(device="cpu", dtype=torch.bfloat16, **reward_fn_kwargs)
            # accelerator.wait_for_everyone()
            self.loss_fn = getattr(reward_fn, config.reward_fn)(device=accelerator.device, dtype=torch.bfloat16, **reward_fn_kwargs)
            # accelerator.wait_for_everyone()

        if config.enable_face_loss:
            # self.id_loss_handler = FaceIdLoss(device=accelerator.device)
            self.face_app = FaceAnalysis(root="wan_models/face_models", device=accelerator.device)

        self.config = config
        self.accelerator = accelerator
        self.step = 0
        self.step_generator = 0
        self.step_critic = 0
        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        # torch.backends.cuda.matmul.allow_tf32 = True
        # torch.backends.cudnn.allow_tf32 = True
        # launch_distributed_job()
        global_rank = accelerator.process_index
        self.global_rank = global_rank
        # self.world_size = dist.get_world_size()
        # self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        self.dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            self.dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            self.dtype = torch.bfloat16
        self.device = accelerator.device
        self.is_main_process = accelerator.is_main_process
        self.causal = config.causal
        self.disable_wandb = config.disable_wandb
        self.disable_tensorboard = config.disable_tensorboard

        fsdp_plugin = accelerator.state.fsdp_plugin if hasattr(accelerator.state, "fsdp_plugin") else None
        if fsdp_plugin is not None:
            from torch.distributed.fsdp import ShardingStrategy
            zero_stage = 0
            if fsdp_plugin.sharding_strategy is ShardingStrategy.FULL_SHARD:
                fsdp_stage = 3
            elif fsdp_plugin.sharding_strategy is None: # The fsdp_plugin.sharding_strategy is None in FSDP 2.
                fsdp_stage = 3
            elif fsdp_plugin.sharding_strategy is ShardingStrategy.SHARD_GRAD_OP:
                fsdp_stage = 2
            else:
                fsdp_stage = 0
            print(f"Using FSDP stage: {fsdp_stage}")
            self.use_fsdp = True
            if fsdp_stage == 3:
                print(f"Auto set save_state to True because fsdp_stage == 3")
                self.save_state = True

        # if fsdp_stage != 0:
        #     def save_model_hook(models, weights, output_dir):
        #         accelerate_state_dict = accelerator.get_state_dict(models[-1], unwrap=True)
        #         if accelerator.is_main_process:
        #             from safetensors.torch import save_file
        #             safetensor_save_path = os.path.join(output_dir, f"diffusion_pytorch_model.safetensors")
        #             save_file(accelerate_state_dict, safetensor_save_path, metadata={"format": "pt"})
        #     accelerator.register_save_state_pre_hook(save_model_hook)

        # use a random seed for the training
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + global_rank)

        if self.is_main_process and not self.disable_wandb:
            wandb.login(host=config.wandb_host, key=config.wandb_key)
            wandb.init(
                config=OmegaConf.to_container(config, resolve=True),
                name=config.config_name,
                mode="online",
                entity=config.wandb_entity,
                project=config.wandb_project,
                dir=config.wandb_save_dir
            )

        self.writer = None
        if self.is_main_process and not self.disable_tensorboard:
            # Add timestamp to tensorboard log directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_tensorboard_dir = getattr(config, 'tensorboard_log_dir', os.path.join(config.logdir, 'tensorboard'))
            tensorboard_log_dir = os.path.join(base_tensorboard_dir, f"run_{timestamp}")
            
            os.makedirs(tensorboard_log_dir, exist_ok=True)
            self.writer = SummaryWriter(log_dir=tensorboard_log_dir)
            
            # Log config as text
            config_str = OmegaConf.to_yaml(config)
            self.writer.add_text('config', config_str, 0)

        self.output_path = config.logdir
        self.distribution_loss = config.distribution_loss

        # Step 2: Initialize the model and optimizer
        if config.distribution_loss == "causvid":
            self.model = CausVid(config, device=self.device)
        elif config.distribution_loss == "causvid_ai2v":
            self.model = CausVid_AI2V(config, device=self.device)
        elif config.distribution_loss == "dmd":
            self.model = DMD(config, device=self.device)
        elif config.distribution_loss == "sid":
            self.model = SiD(config, device=self.device)
        elif config.distribution_loss == "dmd_ai2v_causvid":
            self.model = DMDAI2V_CausVid(config, device=self.device)
        elif config.distribution_loss == "dmd_ai2v_selfforcing":
            self.model = DMDAI2V_SelfForcing(config, device=self.device, accelerator=accelerator)
        else:
            raise ValueError("Invalid distribution matching loss")

        if fsdp_stage != 0:
            # assert False, "only support hybrid_full sharding now."
            # from functools import partial
            from utils.fsdp import shard_model
            # from utils.fuser import set_multi_gpus_devices
            shard_fn = partial(shard_model, device_id=accelerator.device, param_dtype=self.dtype)
            self.model.text_encoder.text_encoder = shard_fn(self.model.text_encoder.text_encoder)

        # accelerator.wait_for_everyone()

        # Save pretrained model state_dicts to CPU
        # self.fake_score_state_dict_cpu = self.model.fake_score.state_dict()

        # self.model.generator = fsdp_wrap(
        #     self.model.generator,
        #     sharding_strategy=config.sharding_strategy,
        #     mixed_precision=config.mixed_precision,
        #     wrap_strategy=config.generator_fsdp_wrap_strategy,
        #     cpu_offload=getattr(config, "model_cpu_offload", False),
        # )

        # self.model.real_score = fsdp_wrap(
        #     self.model.real_score,
        #     sharding_strategy=config.sharding_strategy,
        #     mixed_precision=config.mixed_precision,
        #     wrap_strategy=config.real_score_fsdp_wrap_strategy,
        #     cpu_offload=getattr(config, "model_cpu_offload", False),
        # )

        # self.model.fake_score = fsdp_wrap(
        #     self.model.fake_score,
        #     sharding_strategy=config.sharding_strategy,
        #     mixed_precision=config.mixed_precision,
        #     wrap_strategy=config.fake_score_fsdp_wrap_strategy,
        #     cpu_offload=getattr(config, "model_cpu_offload", False),
        # )

        # [NOTE]: for text encoder, it should be tested that what sharding strategy is better.
        # self.model.text_encoder = fsdp_wrap(
        #     self.model.text_encoder,
        #     sharding_strategy=config.sharding_strategy,
        #     mixed_precision=config.mixed_precision,
        #     wrap_strategy=config.text_encoder_fsdp_wrap_strategy,
        #     cpu_offload=getattr(config, "text_encoder_cpu_offload", False)
        # )
        
        self.model.vae = self.model.vae.to(
            device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = torch.optim.AdamW(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.lr_critic if hasattr(config, "lr_critic") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Step 3: Initialize the dataloader
        # world_size = dist.get_world_size() 
        # local_rank = dist.get_rank()
        # self.local_rank = local_rank
        # if local_rank == 0:
        #     print(f"num of all gpus: {world_size}")
        world_size = accelerator.num_processes
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1:
            local_rank = env_local_rank
        self.local_rank = local_rank
        if self.local_rank == 0:
            print(f"num of all gpus: {world_size}")

        torch_rng = torch.Generator().manual_seed(config.seed + local_rank)

        # Step 3: Initialize the dataloader
        # dataset = TensorDataset(config.base_paths, config.metadata_paths)

        train_dataset = AudioVisualDataset(
            video_sample_size=config.video_sample_size,
            video_sample_stride=config.video_sample_stride,
            video_sample_n_frames=config.video_sample_n_frames,
            video_repeat=1,
            image_sample_size=config.video_sample_size,
            enable_bucket=True, enable_inpaint=True,
            dataset_paths_txt=config.dataset_paths_txt
        )

        aspect_ratio_sample_size = {key : [x / 512 * config.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        # batch_sampler = AspectRatioBatchSingletaskVideoSampler(
        #     sampler_ai2v=RandomSampler(train_dataset.ai2v_dataset, generator=torch_rng), 
        #     ai2v_dataset=train_dataset.ai2v_dataset, 
        #     batch_size=config.batch_size, drop_last=True,
        #     aspect_ratios=aspect_ratio_sample_size,
        # )
        batch_sampler_generator = torch.Generator().manual_seed(config.seed)
        batch_sampler = AspectRatioBatchMultitaskVideoSampler(
            sampler_ai2v=RandomSampler(train_dataset.ai2v_dataset, generator=batch_sampler_generator), 
            sampler_i2v=RandomSampler(train_dataset.i2v_dataset, generator=batch_sampler_generator), 
            ai2v_dataset=train_dataset.ai2v_dataset, 
            i2v_dataset=train_dataset.i2v_dataset,
            batch_size=config.batch_size, drop_last=True,
            aspect_ratios=aspect_ratio_sample_size,
            ai2v_ratio=0.8,
        )

        def worker_init_fn(_seed):
            _seed = _seed * 256
            def _worker_init_fn(worker_id):
                print(f"worker_init_fn with {_seed + worker_id}")
                np.random.seed(_seed + worker_id)
                random.seed(_seed + worker_id)
            return _worker_init_fn

        collate_fn_configed = partial(collate_fn, args=config)

        self.dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn_configed,
            persistent_workers=True if config.dataloader_num_workers != 0 else False,
            num_workers=config.dataloader_num_workers,
            # shuffle=True,
            worker_init_fn=worker_init_fn(config.seed + local_rank)
        )

        num_update_steps_per_epoch = math.ceil(len(train_dataset) / config.accumulation_steps)
        self.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch

        self.lr_scheduler_generator = get_scheduler(
            config.lr_scheduler,
            optimizer=self.generator_optimizer,
            num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=self.max_train_steps * accelerator.num_processes,
        )

        self.lr_scheduler_critic = get_scheduler(
            config.lr_scheduler,
            optimizer=self.critic_optimizer,
            num_warmup_steps=config.lr_warmup_steps * accelerator.num_processes,
            num_training_steps=self.max_train_steps * accelerator.num_processes,
        )

        ##############################################################################################################
        # 7. (If resuming) Load the model and optimizer, lr_scheduler, ema's statedicts
        if getattr(config, "resume_from_checkpoint", True):
            # if args.resume_from_checkpoint != "latest":
            #     path = os.path.basename(args.resume_from_checkpoint)
            # else:
            # Get the most recent checkpoint
            dirs = os.listdir(self.output_path)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("_")[-1]))
            path = dirs[-1] if len(dirs) > 0 else None

            if path is None:
                # no previous ckpts
                accelerator.print(
                    f"Checkpoint does not exist. Starting a new training run."
                )
                self.step = 0
                if getattr(config, "generator_ckpt", False):
                    if os.path.exists(config.generator_ckpt):
                        print(f"Loading pretrained generator from {config.generator_ckpt}")
                        state_dict = torch.load(config.generator_ckpt, map_location="cpu")
                        if "generator" in state_dict:
                            generator_state_dict = state_dict["generator"]
                            # self.accelerator.unwrap_model(self.model.generator).load_state_dict(
                            #     generator_state_dict, strict=True
                            # )
                            # print(generator_state_dict.keys())
                            self.model.generator.load_state_dict(
                                generator_state_dict, strict=True
                            )

                        if "model" in state_dict:
                            generator_state_dict = state_dict["model"]
                            self.model.generator.load_state_dict(
                                generator_state_dict, strict=True
                            )

                        if "critic" in state_dict:
                            critic_state_dict = state_dict["critic"]
                            self.model.fake_score.load_state_dict(
                                critic_state_dict, strict=True
                            )

                        print(f"Model loaded from {config.generator_ckpt}")
                    else:
                        print(f"Loading pretrained generator from {config.generator_ckpt} in seperation.")
                        generator_ckpt_path = config.generator_ckpt[:-3] + "_gen.pt"
                        critic_ckpt_path = config.generator_ckpt[:-3] + "_critic.pt"
                        # ema_ckpt_path = config.generator_ckpt[:-3] + "_ema.pt"
                        # 先不考虑EMA加载
                        if os.path.exists(generator_ckpt_path):
                            generator_state_dict = torch.load(generator_ckpt_path, map_location="cpu")["generator"]
                            self.model.generator.load_state_dict(
                                generator_state_dict, strict=True
                            )
                            generator_state_dict = None
                        if os.path.exists(critic_ckpt_path):
                            critic_state_dict = torch.load(critic_ckpt_path, map_location="cpu")["critic"]
                            self.model.fake_score.load_state_dict(
                                critic_state_dict, strict=True
                            )
                            critic_state_dict = None
                        print(f"Model loaded from {config.generator_ckpt} in seperation.")
            else:
                # load from trained ckpts
                self.step = int(path.split("_")[-1])
                accelerator.print(f"Resuming from checkpoint {path}")
                generator_ckpt_combine_path = os.path.join(path, "model.pt")
                generator_ckpt_path = os.path.join(path, "model_gen.pt")
                critic_ckpt_path = os.path.join(path, "model_critic.pt")
                if os.path.exists(generator_ckpt_combine_path):
                    print(f"Loading pretrained generator from {generator_ckpt_combine_path}")
                    state_dict = torch.load(generator_ckpt_combine_path, map_location="cpu")
                    if "generator" in state_dict:
                        generator_state_dict = state_dict["generator"]
                        self.model.generator.load_state_dict(
                            generator_state_dict, strict=True
                        )

                    if "model" in state_dict:
                        generator_state_dict = state_dict["model"]
                        self.model.generator.load_state_dict(
                            generator_state_dict, strict=True
                        )

                    if "critic" in state_dict:
                        critic_state_dict = state_dict["critic"]
                        self.model.fake_score.load_state_dict(
                            critic_state_dict, strict=True
                        )

                    print(f"Model loaded from {generator_ckpt_combine_path}")
                else:
                    print(f"Loading pretrained generator from {generator_ckpt_path}, {critic_ckpt_path} in seperation.")
                    # ema_ckpt_path = config.generator_ckpt[:-3] + "_ema.pt"
                    # 先不考虑EMA加载
                    if os.path.exists(generator_ckpt_path):
                        generator_state_dict = torch.load(generator_ckpt_path, map_location="cpu")["generator"]
                        self.model.generator.load_state_dict(
                            generator_state_dict, strict=True
                        )
                        generator_state_dict = None
                        print(f"Model loaded from {generator_ckpt_path}.")
                    if os.path.exists(critic_ckpt_path):
                        critic_state_dict = torch.load(critic_ckpt_path, map_location="cpu")["critic"]
                        self.model.fake_score.load_state_dict(
                            critic_state_dict, strict=True
                        )
                        critic_state_dict = None
                        print(f"Model loaded from {critic_ckpt_path}.")
                    print(f"Model loaded from {generator_ckpt_path}, {critic_ckpt_path} in seperation.")

        # print("before prepare")
        # accelerator.wait_for_everyone()
        self.model.generator, self.model.real_score, self.model.fake_score, self.generator_optimizer, \
            self.critic_optimizer, self.dataloader, self.lr_scheduler_generator, self.lr_scheduler_critic = accelerator.prepare(
            self.model.generator, self.model.real_score, self.model.fake_score, self.generator_optimizer, \
                self.critic_optimizer, self.dataloader, self.lr_scheduler_generator, self.lr_scheduler_critic
        )
        # print("after prepare")

        if self.local_rank == 0:
            print("DATASET SIZE %d" % len(train_dataset))

        ##############################################################################################################
        # 6. Set up EMA parameter containers
        rename_param = (
            lambda name: name.replace("_fsdp_wrapped_module.", "")
            .replace("_checkpoint_wrapped_module.", "")
            .replace("_orig_mod.", "")
        )
        self.name_to_trainable_params = {}
        for n, p in self.model.generator.named_parameters():
            if not p.requires_grad:
                continue

            renamed_n = rename_param(n)
            self.name_to_trainable_params[renamed_n] = p

        ema_weight = config.ema_weight
        self.generator_ema = None
        if (ema_weight is not None) and (ema_weight > 0.0):
            print(f"Setting up EMA with weight {ema_weight}")
            self.generator_ema = EMA_ACCELERATE(self.model.generator, decay=ema_weight)

        ##############################################################################################################

        # Let's delete EMA params for early steps to save some computes at training and inference
        if self.step < config.ema_start_step:
            self.generator_ema = None

        self.max_grad_norm_generator = getattr(config, "max_grad_norm_generator", 10.0)
        self.max_grad_norm_critic = getattr(config, "max_grad_norm_critic", 10.0)
        self.previous_time = None

        self.accumulation_steps = config.accumulation_steps

    def save(self):
        print("Start gathering distributed model states...")
        self.accelerator.wait_for_everyone()

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)

        # save generator_state_dict
        generator_state_dict = self.accelerator.get_state_dict(self.model.generator, unwrap=True)
        state_dict_gen = {
            "generator": generator_state_dict,
        }
        if self.is_main_process:
            torch.save(state_dict_gen, os.path.join(self.output_path,
                    f"checkpoint_model_{self.step:06d}", "model_gen.pt"))
        generator_state_dict = None
        state_dict_gen = None
        gc.collect()

        # save critic_state_dict
        critic_state_dict = self.accelerator.get_state_dict(self.model.fake_score, unwrap=True)
        state_dict_critic = {
            "critic": critic_state_dict,
        }
        if self.is_main_process:
            torch.save(state_dict_critic, os.path.join(self.output_path,
                    f"checkpoint_model_{self.step:06d}", "model_critic.pt"))
        critic_state_dict = None
        state_dict_critic = None
        gc.collect()

        if self.config.ema_start_step < self.step:
            state_dict_ema = {
                "generator_ema": self.generator_ema.state_dict(),
            }
            # if self.is_main_process:
            torch.save(state_dict_ema, os.path.join(self.output_path,
                    f"checkpoint_model_{self.step:06d}", f"model_ema_rank_{self.global_rank}.pt"))

        print("Model saved to", os.path.join(self.output_path,
                f"checkpoint_model_{self.step:06d}", "model_xxx.pt"))
        self.accelerator.wait_for_everyone()

        # self.accelerator.wait_for_everyone()
        # print("Start gathering distributed model states...")
        # generator_state_dict = fsdp_state_dict(
        #     self.model.generator)
        # critic_state_dict = fsdp_state_dict(
        #     self.model.fake_score)

        # if self.config.ema_start_step < self.step:
        #     state_dict = {
        #         "generator": generator_state_dict,
        #         "critic": critic_state_dict,
        #         "generator_ema": self.generator_ema.state_dict(),
        #     }
        # else:
        #     state_dict = {
        #         "generator": generator_state_dict,
        #         "critic": critic_state_dict,
        #     }

        # if self.is_main_process:
        #     os.makedirs(os.path.join(self.output_path,
        #                 f"checkpoint_model_{self.step:06d}"), exist_ok=True)
        #     torch.save(state_dict, os.path.join(self.output_path,
        #                f"checkpoint_model_{self.step:06d}", "model.pt"))
        #     print("Model saved to", os.path.join(self.output_path,
        #           f"checkpoint_model_{self.step:06d}", "model.pt"))

        # TODO: test model saving with accelerate

    def fwdbwd_one_step(self, batch, train_generator):
        # self.model.eval()  # prevent any randomness (e.g. dropout)

        if train_generator:
            self.step_generator += 1
        else:
            self.step_critic += 1

        # if self.step % 20 == 0:
        torch.cuda.empty_cache()

        # [NOTE]: 没有传入参数，如需要得专门传
        # accumulation_steps = getattr(self, "accumulation_steps", 1)

        # Step 1: Get the next batch of text prompts
        # text_prompts = batch["prompts"]
        # if self.config.i2v:
        #     clean_latent = None
        #     image_latent = batch["ode_latent"][:, -1][:, 0:1, ].to(
        #         device=self.device, dtype=self.dtype)
        # else:
        #     clean_latent = None
        #     image_latent = None

        # batch_size = len(text_prompts)
        # if self.config.distribution_loss == "dmd_ai2v_selfforcing":
        #     new_batch = self.adapt_batch_sf(batch)
        # else:
        new_batch = self.adapt_batch(batch)
        clean_latent = new_batch["clean_latents"]
        conditional_dict = new_batch["conditional_dict"]
        unconditional_dict = new_batch["unconditional_dict"]

        wav2vec_embedding = new_batch["wav2vec_embedding"]
        face_mask = new_batch["face_mask"]
        ref_target_masks = new_batch["ref_target_masks"]
        clip_context = new_batch["clip_context"]
        inpaint_latents = new_batch["inpaint_latents"]

        image_latent = clean_latent[:, 0:1, ] # [NOTE]: check this, i move this from the next next row
        gt_ref_images = new_batch["pixel_values"]
        # clean_latent = clean_latent.permute(0, 2, 1, 3, 4) # TODO: CHECK THIS
        
        # batch_size = clean_latent.shape[0]
        """
        [TODO]: check 为什么要预设一个固定的shape?
        image_or_video_shape = list(self.config.image_or_video_shape)
        image_or_video_shape[0] = batch_size
        """

        image_or_video_shape = clean_latent.shape

        # for self-forcing training
        if self.distribution_loss == "dmd_ai2v_selfforcing":
            clean_latent = None

        # Step 2: Extract the conditional infos
        # with torch.no_grad():
        #     conditional_dict = self.model.text_encoder(
        #         text_prompts=text_prompts)

        #     if not getattr(self, "unconditional_dict", None):
        #         unconditional_dict = self.model.text_encoder(
        #             text_prompts=[self.config.negative_prompt] * batch_size)
        #         unconditional_dict = {k: v.detach()
        #                               for k, v in unconditional_dict.items()}
        #         self.unconditional_dict = unconditional_dict  # cache the unconditional_dict
        #     else:
        #         unconditional_dict = self.unconditional_dict

        # Step 3: Store gradients for the generator (if training the generator)
        
        if train_generator:

            # critic_loss, critic_log_dict = self.model.critic_loss(
            #     image_or_video_shape=image_or_video_shape,
            #     conditional_dict=conditional_dict,
            #     unconditional_dict=unconditional_dict,
            #     clean_latent=clean_latent,
            #     # initial_latent=image_latent if self.config.i2v else None,
            #     wav2vec_embedding=wav2vec_embedding,
            #     face_mask=face_mask,
            #     ref_target_masks=ref_target_masks,
            #     clip_context=clip_context,
            #     inpaint_latents=inpaint_latents,
            # )
            VISUALIZE_GEN = self.step_generator % 5 == 0

            generator_loss, generator_log_dict = self.model.generator_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                initial_latent=image_latent if self.config.i2v else None,
                wav2vec_embedding=wav2vec_embedding,
                face_mask=face_mask,
                ref_target_masks=ref_target_masks,
                clip_context=clip_context,
                inpaint_latents=inpaint_latents,
            )
            

            # torch.cuda.empty_cache()
            # calc face id loss
            # if self.config.enable_face_loss or self.config.enable_reward_loss:
            #     def vae_decode_fn(latent):
            #         # 注意：checkpoint 要求输入是 tensor(s)，并且内部最好不要有 in-place 操作
            #         return self.model.vae.decode_to_pixel(latent[:, :int(self.config.num_decode_latents)].to(torch.bfloat16))

            #     output_gen = generator_log_dict["dmdtrain_clean_latent"]
            #     output_video = torch.utils.checkpoint.checkpoint(
            #         vae_decode_fn,
            #         output_gen,
            #         use_reentrant=False,
            #     )            

            # if self.config.enable_face_loss and self.step_generator > 500:
            #     # ID loss calculation
            #     torch.cuda.empty_cache()
            #     with torch.no_grad():
            #         ref_image = self.model.vae.decode_to_pixel(image_latent.detach().to(torch.bfloat16))
            #         ref_image = ref_image[0].flip(1) # .repeat(output_frames.shape[0],1,1,1)
            #     # ID loss calc
            #     output_frames = output_video[0, ...].flip(1)
            #     face_loss = self.id_loss_handler.face_id_loss_with_ref(output_frames, ref_image=ref_image)
            #     generator_loss = generator_loss + face_loss
            #     generator_log_dict.update({"face_loss": face_loss})

            def vae_decode_fn(latent, grad_target):
                # 注意：checkpoint 要求输入是 tensor(s)，并且内部最好不要有 in-place 操作
                return self.model.vae.decode_to_pixel(latent.to(torch.bfloat16), grad_target=grad_target)

            output_gen = generator_log_dict["dmdtrain_clean_latent"] # torch.Size([1, 48, 12, 56, 56])
            total_latents = output_gen.shape[2]
            grad_start = random.randint(1, total_latents-1-2)
            grad_target = [grad_start, grad_start+1] # decode [grad_start, grad_start] 闭区间内的latent使用梯度
            decoded_range = [(grad_target[0]-1)*4+1, (grad_target[1]+1-1)*4+1]

            output_video = torch.utils.checkpoint.checkpoint(
                vae_decode_fn,
                output_gen,
                grad_target,
                use_reentrant=False,
            )
            output_video = output_video[:, decoded_range[0]:decoded_range[1]]
            output_frames = output_video.view(-1, output_video.shape[2],output_video.shape[3],output_video.shape[4]).flip(1)

            gt_start = 0
            gt_end = gt_ref_images.shape[2]
            N = decoded_range[1] - decoded_range[0]
            gt_images_sample_indexes = random.choices(range(gt_start, gt_end), k=N)
            gt_ref_images = gt_ref_images[0, :].permute(1,0,2,3) # after: b, 3, h, w
            gt_ref_images = gt_ref_images[gt_images_sample_indexes].contiguous()

            valid_face_loss = True
            # calc gt images face embeddings online, with no grad
            with torch.no_grad():
                gt_id_embedding = []
                gt_id_mask = []
                for f in gt_ref_images:
                    f = f.float() #(3, h, w) -1 1
                    # g = ((f + 1) / 2 * 255).permute(1,2,0).cpu().numpy()
                    # cv2.imwrite("test.jpg", g) # 反色rgb

                    bboxes, kpss = self.face_app.detection_model.detect(f)
                    if bboxes.shape[0] > 0:
                        indexed_bboxes = [(i, x) for i, x in enumerate(bboxes)]
                        sorted_bboxes = sorted(indexed_bboxes, key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]))
                        max_index, max_bbox = sorted_bboxes[-1]
                        kps = kpss[max_index]
                        face = Face(bbox=bboxes[max_index][0:4], kps=kps, det_score=bboxes[max_index][4])
                        gt_id_embedding.append(self.face_app.arcface_model.get(f, face))
                        gt_id_mask.append(1)
                    else:
                        gt_id_embedding.append(torch.zeros(512).to(self.device))
                        gt_id_mask.append(0)

                gt_id_embedding = torch.stack(gt_id_embedding).unsqueeze(0)
                if sum(gt_id_mask) == 0:
                    valid_face_loss = False
                gt_id_mask = torch.tensor(gt_id_mask).unsqueeze(0).to(gt_id_embedding.device)
    
            # calc pred images face embeddings online
            pred_id_embedding = []
            pred_id_mask = []
            for f in output_frames:
                f = f.float() #(3, h, w) -1 1
                # g = ((f + 1) / 2 * 255).permute(1,2,0).cpu().numpy()
                # cv2.imwrite("test.jpg", g) # 反色rgb

                bboxes, kpss = self.face_app.detection_model.detect(f)
                if bboxes.shape[0] > 0:
                    indexed_bboxes = [(i, x) for i, x in enumerate(bboxes)]
                    sorted_bboxes = sorted(indexed_bboxes, key=lambda item: (item[1][2] - item[1][0]) * (item[1][3] - item[1][1]))
                    max_index, max_bbox = sorted_bboxes[-1]
                    kps = kpss[max_index]
                    face = Face(bbox=bboxes[max_index][0:4], kps=kps, det_score=bboxes[max_index][4])
                    pred_id_embedding.append(self.face_app.arcface_model.get(f, face))
                    pred_id_mask.append(1)
                else:
                    pred_id_embedding.append(torch.zeros(512).to(self.device))
                    pred_id_mask.append(0)

            pred_id_embedding = torch.stack(pred_id_embedding).unsqueeze(0)
            if sum(pred_id_mask) == 0:
                valid_face_loss = False
            pred_id_mask = torch.tensor(pred_id_mask).unsqueeze(0).to(pred_id_embedding.device)

            if self.step_generator < 500:
                valid_face_loss = False

            if valid_face_loss:
                face_score = self.face_app.pool_embedding_loss(pred_id_embedding, gt_id_embedding, pred_id_mask)
                face_loss = (1 - face_score) * 0.05
            else:
                face_loss = 0.
                # print("no face detected in either GT or Pred. Skip Face loss.")
            generator_loss = generator_loss + face_loss

            if self.config.enable_reward_loss:
                # reward loss calc
                output_video = output_video.clamp(-1, 1)
                output_video = (output_video / 2 + 0.5).clamp(0, 1)  # [-1, 1] -> [0, 1]
                # def reward_fn(output_video, text):
                #     # 注意：checkpoint 要求输入是 tensor(s)，并且内部最好不要有 in-place 操作
                #     return self.loss_fn(output_video, text)
                # loss_reward, reward = torch.utils.checkpoint.checkpoint(
                #     reward_fn,
                #     output_video.permute(0,2,1,3,4),
                #     batch['text'][0],
                #     use_reentrant=False,
                # )        
                # print(batch['text'])
                # output_video: b t c h w
                if self.config.random_one_latent_reward:
                    num_latents = output_video.shape[1]
                    seleceted_latent_idx = random.randrange(num_latents)
                    output_video = output_video[:, seleceted_latent_idx:seleceted_latent_idx+1]
                loss_reward, reward = self.loss_fn(output_video.permute(0,2,1,3,4), batch['text'])
                generator_loss = generator_loss + loss_reward * 0.05
                generator_log_dict.update({"reward": reward})

            torch.cuda.empty_cache()

            generator_loss = generator_loss / self.accumulation_steps
            # print("before backward")
            generator_loss.backward()
            generator_grad_norm = self.model.generator.clip_grad_norm_(
                self.max_grad_norm_generator)

            generator_log_dict.update({"generator_loss": generator_loss,
                                       "generator_grad_norm": generator_grad_norm,
                                       "face_loss": face_loss,})

            update_params_flag_generator = False
            update_params_flag_critic = False
            if self.step_generator % self.accumulation_steps == 0:
                update_params_flag_generator = True

            if VISUALIZE_GEN and self.is_main_process:
                # Visualize the input, output, and ground truth
                output_video_dir = os.path.join(self.config.logdir, "dmd_train_visualize/")
                os.makedirs(output_video_dir, exist_ok=True)

                if self.distribution_loss != "dmd_ai2v_selfforcing":
                    gen_input_noisy = generator_log_dict["dmdtrain_gen_input_noisy"]
                    gen_input_video = self.model.vae.decode_to_pixel(gen_input_noisy.to(torch.bfloat16))
                    gen_input_video = rearrange(gen_input_video, 'b t c h w -> b t h w c').cpu()
                    gen_input_video = 255.0 * (gen_input_video.cpu().numpy() * 0.5 + 0.5)
                    gen_input_video_path = os.path.join(output_video_dir, f"gen_input_{self.step}.mp4")
                    write_video(gen_input_video_path, gen_input_video[0], fps=16)

                input_noisy = generator_log_dict["dmdtrain_noisy_latent"]
                output_gen = generator_log_dict["dmdtrain_clean_latent"].detach()
                pred_real = generator_log_dict["dmdtrain_pred_real_image"]
                pred_fake = generator_log_dict["dmdtrain_pred_fake_image"]
                    
                input_video = self.model.vae.decode_to_pixel(input_noisy.to(torch.bfloat16))
                output_video = self.model.vae.decode_to_pixel(output_gen.to(torch.bfloat16))
                # torch.Size([1, 81, 3, 512, 512]), -1, 1
                pred_real_video = self.model.vae.decode_to_pixel(pred_real.to(torch.bfloat16))
                pred_fake_video = self.model.vae.decode_to_pixel(pred_fake.to(torch.bfloat16))
                    
                input_video = rearrange(input_video, 'b t c h w -> b t h w c').cpu()
                output_video = rearrange(output_video, 'b t c h w -> b t h w c').cpu()
                pred_fake_video = rearrange(pred_fake_video, 'b t c h w -> b t h w c').cpu()
                pred_real_video = rearrange(pred_real_video, 'b t c h w -> b t h w c').cpu()

                input_video = 255.0 * (input_video.cpu().numpy() * 0.5 + 0.5)
                output_video = 255.0 * (output_video.cpu().numpy() * 0.5 + 0.5)
                pred_fake_video = 255.0 * (pred_fake_video.cpu().numpy() * 0.5 + 0.5)
                pred_real_video = 255.0 * (pred_real_video.cpu().numpy() * 0.5 + 0.5)

                input_video_path = os.path.join(output_video_dir, f"input_{self.step}.mp4")
                output_video_path = os.path.join(output_video_dir, f"output_{self.step}.mp4")
                pred_fake_video_path = os.path.join(output_video_dir, f"pred_fake_{self.step}.mp4")
                pred_real_video_path = os.path.join(output_video_dir, f"pred_real_{self.step}.mp4")
                
                write_video(input_video_path, input_video[0], fps=16)
                write_video(output_video_path, output_video[0], fps=16)
                write_video(pred_fake_video_path, pred_fake_video[0], fps=16)
                write_video(pred_real_video_path, pred_real_video[0], fps=16)

            torch.cuda.empty_cache()
            return generator_log_dict, update_params_flag_generator, update_params_flag_critic
        else:
            # generator_log_dict = {}
            # Step 4: Store gradients for the critic (if training the critic)
            critic_loss, critic_log_dict = self.model.critic_loss(
                image_or_video_shape=image_or_video_shape,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                clean_latent=clean_latent,
                # initial_latent=image_latent if self.config.i2v else None,
                wav2vec_embedding=wav2vec_embedding,
                face_mask=face_mask,
                ref_target_masks=ref_target_masks,
                clip_context=clip_context,
                inpaint_latents=inpaint_latents,
            )

            critic_loss = critic_loss / self.accumulation_steps
            critic_loss.backward()
            critic_grad_norm = self.model.fake_score.clip_grad_norm_(
                self.max_grad_norm_critic)

            critic_log_dict.update({"critic_loss": critic_loss,
                                    "critic_grad_norm": critic_grad_norm})

            update_params_flag_generator = False
            update_params_flag_critic = False
            if self.step_critic % self.accumulation_steps == 0:
                update_params_flag_critic = True

            return critic_log_dict, update_params_flag_generator, update_params_flag_critic

    # def adapt_batch_sf(self, batch):
    #     # Convert images to latent space
    #     pixel_values = batch["pixel_values"].to(self.device, self.dtype)
    #     wav2vec_embedding = batch["wav2vec_embedding"]

    #     training_with_video_token_length = False
    #     random_frame_crop = False
    #     keep_all_node_same_token_length = False
    #     assert training_with_video_token_length == False, "not support training_with_video_token_length"

    #     if self.config.train_mode != "normal":
    #         clip_pixel_values = batch["clip_pixel_values"].to(self.dtype)
    #         mask_pixel_values = batch["mask_pixel_values"].to(self.device, self.dtype)
    #         mask = batch["mask"].to(self.device, self.dtype)
    #         face_mask = batch["face_mask"].to(self.device, self.dtype)

    #     # Make the inpaint latents to be zeros.
    #     if self.config.train_mode != "normal":
    #         t2v_flag = [(_mask == 1).all() for _mask in mask]
    #         new_t2v_flag = []
    #         for _mask in t2v_flag:
    #             if _mask and np.random.rand() < 0.90:
    #                 new_t2v_flag.append(0)
    #             else:
    #                 new_t2v_flag.append(1)
    #         t2v_flag = torch.from_numpy(np.array(new_t2v_flag)).to(self.device, dtype=self.dtype)

    #     if self.config.low_vram: # [NOTE]: 不理解为啥和下面有两个差不多的代码
    #         torch.cuda.empty_cache()
    #         self.model.vae.to(self.device)
    #         self.model.clip_image_encoder.to(self.device)
    #         self.model.text_encoder.cpu()
            

    #     with torch.no_grad():
    #         # [NOTE]: 0808, modified to here. 下周需要继续对齐vae的输入输出维度，以及和self forcing原本的对齐
    #         pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
    #         clean_latents = self.model.vae.encode_to_latent(pixel_values).to(self.device, self.dtype) # input: b, c=3, F, H, W output: b, f, c=16, h, w

    #         if self.config.train_mode != "normal":
    #             mask = rearrange(mask, "b f c h w -> b c f h w") # 1, 1, 81, h, w
    #             # TODO: what is this for?
    #             mask = torch.concat(
    #                 [
    #                     torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2), 
    #                     mask[:, :, 1:]
    #                 ], dim=2
    #             ) # 1,1,84,h,w
    #             mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]) # 1, 21, 4, h, w
    #             mask = mask.transpose(1, 2) # 1, 4, 21, h, w
    #             mask = resize_mask(1 - mask, clean_latents.permute(0,2,1,3,4))

    #             # resize face mask to 1, 1, 21, h, w
    #             face_mask = rearrange(face_mask, "b f c h w -> b c f h w")
    #             # lip_mask = rearrange(lip_mask, "b f c h w -> b c f h w")

    #             # Encode inpaint latents.
    #             mask_pixel_values = mask_pixel_values.permute(0, 2, 1, 3, 4)
    #             mask_latents = self.model.vae.encode_to_latent(mask_pixel_values).to(self.device, self.dtype).permute(0, 2, 1, 3, 4)

    #             # mask = mask.transpose(1, 2) # 1, 4, 21, h, w
    #             inpaint_latents = torch.concat([mask, mask_latents], dim=1) # output: b, f, c=16+4, h, w
    #             inpaint_latents = t2v_flag[:, None, None, None, None] * inpaint_latents

    #             clip_context = []
    #             for clip_pixel_value in clip_pixel_values:
    #                 clip_image = Image.fromarray(np.uint8(clip_pixel_value.float().cpu().numpy()))
    #                 clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(self.model.clip_image_encoder.device, self.dtype)
    #                 _clip_context = self.model.clip_image_encoder([clip_image[:, None, :, :]])

    #                 rng = None
    #                 if rng is None:
    #                     zero_init_clip_in = np.random.choice([True, False], p=[0.1, 0.9])
    #                 else:
    #                     zero_init_clip_in = rng.choice([True, False], p=[0.1, 0.9])
    #                 clip_context.append(_clip_context if not zero_init_clip_in else torch.zeros_like(_clip_context))
                    
    #             clip_context = torch.cat(clip_context)


    #     if self.config.low_vram:
    #         self.model.vae.to('cpu')
    #         self.model.clip_image_encoder.to('cpu')
    #         torch.cuda.empty_cache()
    #         self.model.text_encoder.to(self.device)

    #     with torch.no_grad():
    #         # [NOTE]: 维度两边需要对齐
    #         bsz = clean_latents.shape[0]        
    #         conditional_dict = self.model.text_encoder(text_prompts=batch['text'])

    #         if not getattr(self, "unconditional_dict", None):
    #             unconditional_dict = self.model.text_encoder(
    #                 text_prompts=[self.config.negative_prompt] * bsz)
    #             unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
    #             self.unconditional_dict = unconditional_dict
    #         else:
    #             unconditional_dict = self.unconditional_dict

    #         # prompt_embeds = text_encoder(batch['text'], accelerator.device)[0] # [NOTE]: 输入输出都为一个列表，推理的是否pos neg可以写到一起
    #         # prompt_embeds = [prompt_embeds]

    #     if self.config.low_vram:
    #         self.model.text_encoder.cpu()
    #         torch.cuda.empty_cache()

    #     ref_target_masks = torch.ones_like(face_mask).to(face_mask)
    #     ref_target_masks = ref_target_masks[0, 0, 0:3] # 3, h, w

    #     new_batch = {
    #         "clean_latents": clean_latents,
    #         "conditional_dict": conditional_dict,
    #         "unconditional_dict": unconditional_dict,
    #         "wav2vec_embedding": wav2vec_embedding,
    #         "face_mask": face_mask,
    #         "ref_target_masks": ref_target_masks,
    #         "clip_context": clip_context,
    #         "inpaint_latents": inpaint_latents,

    #     }

    #     return new_batch

    def adapt_batch(self, batch):
        # Convert images to latent space
        pixel_values = batch["pixel_values"].to(self.device, self.dtype)
        wav2vec_embedding = batch["wav2vec_embedding"]

        training_with_video_token_length = False
        random_frame_crop = False
        keep_all_node_same_token_length = False
        assert training_with_video_token_length == False, "not support training_with_video_token_length"

        if self.config.train_mode != "normal":
            clip_pixel_values = batch["clip_pixel_values"].to(self.dtype)
            mask_pixel_values = batch["mask_pixel_values"].to(self.device, self.dtype)
            mask = batch["mask"].to(self.device, self.dtype)
            face_mask = batch["face_mask"].to(self.device, self.dtype)

        # Make the inpaint latents to be zeros.
        if self.config.train_mode != "normal":
            t2v_flag = [(_mask == 1).all() for _mask in mask]
            new_t2v_flag = []
            for _mask in t2v_flag:
                if _mask and np.random.rand() < 0.90:
                    new_t2v_flag.append(0)
                else:
                    new_t2v_flag.append(1)
            t2v_flag = torch.from_numpy(np.array(new_t2v_flag)).to(self.device, dtype=self.dtype)

        if self.config.low_vram: # [NOTE]: 不理解为啥和下面有两个差不多的代码
            torch.cuda.empty_cache()
            self.model.vae.to(self.device)
            self.model.clip_image_encoder.to(self.device)
            self.model.text_encoder.cpu()
            

        with torch.no_grad():
            # [NOTE]: 0808, modified to here. 下周需要继续对齐vae的输入输出维度，以及和self forcing原本的对齐
            pixel_values = pixel_values.permute(0, 2, 1, 3, 4)
            clean_latents = self.model.vae.encode_to_latent(pixel_values).to(self.device, self.dtype) # input: b, c=3, F, H, W output: b, f, c=16, h, w

            if self.config.train_mode != "normal":
                mask = rearrange(mask, "b f c h w -> b c f h w") # 1, 1, 81, h, w
                # TODO: what is this for?
                mask = torch.concat(
                    [
                        torch.repeat_interleave(mask[:, :, 0:1], repeats=4, dim=2), 
                        mask[:, :, 1:]
                    ], dim=2
                ) # 1,1,84,h,w
                mask = mask.view(mask.shape[0], mask.shape[2] // 4, 4, mask.shape[3], mask.shape[4]) # 1, 21, 4, h, w
                mask = mask.transpose(1, 2) # 1, 4, 21, h, w
                mask = resize_mask(1 - mask, clean_latents.permute(0,2,1,3,4))

                # resize face mask to 1, 1, 21, h, w
                face_mask = rearrange(face_mask, "b f c h w -> b c f h w")
                # lip_mask = rearrange(lip_mask, "b f c h w -> b c f h w")

                # Encode inpaint latents.
                mask_pixel_values = mask_pixel_values.permute(0, 2, 1, 3, 4)
                mask_latents = self.model.vae.encode_to_latent(mask_pixel_values).to(self.device, self.dtype).permute(0, 2, 1, 3, 4)

                # mask = mask.transpose(1, 2) # 1, 4, 21, h, w
                inpaint_latents = torch.concat([mask, mask_latents], dim=1) # output: b, f, c=16+4, h, w
                inpaint_latents = t2v_flag[:, None, None, None, None] * inpaint_latents

                clip_context = []
                for clip_pixel_value in clip_pixel_values:
                    clip_image = Image.fromarray(np.uint8(clip_pixel_value.float().cpu().numpy()))
                    clip_image = TF.to_tensor(clip_image).sub_(0.5).div_(0.5).to(self.model.clip_image_encoder.device, self.dtype)
                    _clip_context = self.model.clip_image_encoder([clip_image[:, None, :, :]])

                    rng = None
                    if rng is None:
                        zero_init_clip_in = np.random.choice([True, False], p=[0.1, 0.9])
                    else:
                        zero_init_clip_in = rng.choice([True, False], p=[0.1, 0.9])
                    clip_context.append(_clip_context if not zero_init_clip_in else torch.zeros_like(_clip_context))
                    
                clip_context = torch.cat(clip_context)


        if self.config.low_vram:
            self.model.vae.to('cpu')
            self.model.clip_image_encoder.to('cpu')
            torch.cuda.empty_cache()
            self.model.text_encoder.to(self.device)

        with torch.no_grad():
            # [NOTE]: 维度两边需要对齐
            bsz = clean_latents.shape[0]        
            conditional_dict = self.model.text_encoder(text_prompts=batch['text'])

            if not getattr(self, "unconditional_dict", None):
                unconditional_dict = self.model.text_encoder(
                    text_prompts=[self.config.negative_prompt] * bsz)
                unconditional_dict = {k: v.detach() for k, v in unconditional_dict.items()}
                self.unconditional_dict = unconditional_dict
            else:
                unconditional_dict = self.unconditional_dict

            # prompt_embeds = text_encoder(batch['text'], accelerator.device)[0] # [NOTE]: 输入输出都为一个列表，推理的是否pos neg可以写到一起
            # prompt_embeds = [prompt_embeds]

        if self.config.low_vram:
            self.model.text_encoder.cpu()
            torch.cuda.empty_cache()

        ref_target_masks = torch.ones_like(face_mask).to(face_mask)
        ref_target_masks = ref_target_masks[0, 0, 0:3] # 3, h, w

        new_batch = {
            "clean_latents": clean_latents,
            "conditional_dict": conditional_dict,
            "unconditional_dict": unconditional_dict,
            "wav2vec_embedding": wav2vec_embedding,
            "face_mask": face_mask,
            "ref_target_masks": ref_target_masks,
            "clip_context": clip_context,
            "inpaint_latents": inpaint_latents,
            'pixel_values': pixel_values,

        }

        return new_batch

    def train(self):
        self.model.generator.model.train()
        self.model.fake_score.model.train()

        start_step = self.step
        # if self.is_main_process:
        #     pbar = tqdm(total=999999, desc="Training", 
        #             unit="iter", ncols=100, 
        #             bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        pbar = tqdm(
            range(0, self.max_train_steps),
            initial=start_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
        )

        # self.dataloader.batch_sampler.epoch = self.dataloader.batch_sampler.epoch + 1
        # print("dataloader epoch:", self.dataloader.batch_sampler.epoch)
        local_dataloader_iterator = iter(self.dataloader)
        # print("accumulation_steps:", self.accumulation_steps)
        # cnt = 0
        train_gen_loss = 0.0
        train_critic_loss = 0.0
        can_save = True
        while True:
            # print(f"count step: {cnt}")
            # cnt += 1
            # if train_generator:
            #     self.step_generator += 1
            # else:
            #     self.step_critic += 1
            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
            # Train the generator
            if TRAIN_GENERATOR:
                extras_list = []
                try:
                    batch = next(local_dataloader_iterator)
                except StopIteration:
                    # 当数据取完时，重新设置迭代器
                    # print("数据迭代完，重新打乱并开始新的循环...")
                    # self.dataloader.batch_sampler.epoch = self.dataloader.batch_sampler.epoch + 1
                    print("数据迭代完，开始新的循环...")
                    local_dataloader_iterator = iter(self.dataloader)  # 重新创建迭代器
                    batch = next(local_dataloader_iterator)

                if self.step == start_step:
                    pixel_values, texts = batch['pixel_values'].cpu(), batch['text']
                    pixel_values = rearrange(pixel_values, "b f c h w -> b c f h w")
                    os.makedirs(os.path.join(self.config.logdir, "sanity_check"), exist_ok=True)
                    for idx, (pixel_value, text) in enumerate(zip(pixel_values, texts)):
                        pixel_value = pixel_value[None, ...]
                        gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{idx}'
                        save_videos_grid(pixel_value, f"{self.config.logdir}/sanity_check/{gif_name[:10]}.gif", rescale=True)
                    if self.config.train_mode != "normal":
                        clip_pixel_values, mask_pixel_values, texts = batch['clip_pixel_values'].cpu(), batch['mask_pixel_values'].cpu(), batch['text']
                        mask_pixel_values = rearrange(mask_pixel_values, "b f c h w -> b c f h w")
                        for idx, (clip_pixel_value, pixel_value, text) in enumerate(zip(clip_pixel_values, mask_pixel_values, texts)):
                            pixel_value = pixel_value[None, ...]
                            Image.fromarray(np.uint8(clip_pixel_value)).save(f"{self.config.logdir}/sanity_check/clip_{gif_name[:10] if not text == '' else f'{idx}'}.png")
                            save_videos_grid(pixel_value, f"{self.config.logdir}/sanity_check/mask_{gif_name[:10] if not text == '' else f'{idx}'}.gif", rescale=True) 
                        # face mask
                        pixel_value_face_mask = batch['pixel_values'].cpu() * batch["face_mask"].cpu()
                        pixel_value_face_mask = rearrange(pixel_value_face_mask, "b f c h w -> b c f h w")
                        for idx, (pixel_value, text) in enumerate(zip(pixel_value_face_mask, texts)):
                            pixel_value = pixel_value[None, ...]
                            gif_name = '-'.join(text.replace('/', '').split()[:10]) if not text == '' else f'{idx}'
                            save_videos_grid(pixel_value, f"{self.config.logdir}/sanity_check/{gif_name[:10]}_face_mask.gif", rescale=True)

                extra, update_params_flag_generator, update_params_flag_critic = self.fwdbwd_one_step(batch, True)
                extras_list.append(extra)
                generator_log_dict = merge_dict_list(extras_list)

                if update_params_flag_generator:
                    # loss_generated = generator_log_dict["generator_loss"]
                    # # Gather the losses across all processes for logging (if we use distributed training).
                    # avg_loss = self.accelerator.gather(loss_generated.repeat(1)).mean()
                    # train_gen_loss += avg_loss.item() / self.accumulation_steps
                    self.generator_optimizer.step()
                    self.generator_optimizer.zero_grad() # set_to_none=True
                    self.lr_scheduler_generator.step()
                    if self.generator_ema is not None:
                        self.generator_ema.update(self.model.generator)
                    can_save = True
                    if self.is_main_process:
                        # Update progress bar
                        pbar.update(1)
                        pbar_str = f"generator loss: {generator_log_dict['generator_loss']:.4f}, critic_loss: -1"
                        if 'face_loss' in generator_log_dict:
                            pbar_str = pbar_str + f", face id loss: {generator_log_dict['face_loss']:.4f}"
                        if 'reward' in generator_log_dict:
                            pbar_str = pbar_str + f", reward: {generator_log_dict['reward']:.4f}"
                        pbar.write(pbar_str)

            # Train the critic
            extras_list = []
            try:
                batch = next(local_dataloader_iterator)
            except StopIteration:
                # 当数据取完时，重新设置迭代器
                # print("数据迭代完，重新打乱并开始新的循环...")
                # self.dataloader.batch_sampler.epoch = self.dataloader.batch_sampler.epoch + 1
                print("数据迭代完，开始新的循环...")
                local_dataloader_iterator = iter(self.dataloader)  # 重新创建迭代器
                batch = next(local_dataloader_iterator)
            extra, update_params_flag_generator, update_params_flag_critic = self.fwdbwd_one_step(batch, False)
            extras_list.append(extra)
            critic_log_dict = merge_dict_list(extras_list)
            if update_params_flag_critic:

                # loss_critic = generator_log_dict["generator_loss"]
                # # Gather the losses across all processes for logging (if we use distributed training).
                # avg_loss = self.accelerator.gather(loss_critic.repeat(1)).mean()
                # train_critic_loss += avg_loss.item() / self.accumulation_steps
                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad() # set_to_none=True
                self.lr_scheduler_critic.step()
                can_save = True
                if self.is_main_process:
                    pbar.update(1)
                    pbar.write(f"generator loss: -1, critic_loss: {critic_log_dict['critic_loss']:.4f}")

            """
            a = [(name, param) for name, param in self.model.generator.named_parameters()]
            for k, v in a:
                if v is not None:
                    if v.shape[0]> 0:
                        print(k, 2, torch.isnan(v.grad).any(), torch.isinf(v.grad).any())
                    # print(k, torch.isnan(v.grad).any(), torch.isinf(v.grad).any())
            """

            # Create EMA params (if not already created)
            if (self.step >= self.config.ema_start_step) and \
                    (self.generator_ema is None) and (self.config.ema_weight > 0):
                self.generator_ema = EMA_ACCELERATE(self.model.generator, decay=self.config.ema_weight)

            # Logging
            if self.step % self.config.gc_interval == 0:
                if dist.get_rank() == 0:
                    logging.info("DistGarbageCollector: Running GC.")
                gc.collect()
                torch.cuda.empty_cache()

            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if self.writer is not None:
                        self.writer.add_scalar("timing/per_iteration_time", current_time - self.previous_time, self.step)
                    self.previous_time = current_time

                if TRAIN_GENERATOR:
                    self.writer.add_scalar("generator_loss", generator_log_dict["generator_loss"].mean().item(), self.step)
                    self.writer.add_scalar("generator_grad_norm", generator_log_dict["generator_grad_norm"].mean().item(), self.step)
                    self.writer.add_scalar("dmdtrain_gradient_norm", generator_log_dict["dmdtrain_gradient_norm"].mean().item(), self.step)

                self.writer.add_scalar("critic_loss", critic_log_dict["critic_loss"].mean().item(), self.step)
                self.writer.add_scalar("critic_grad_norm", critic_log_dict["critic_grad_norm"].mean().item(), self.step)

            # Save the model
            if (not self.config.no_save) and (self.step - start_step) > 0 and self.step % self.config.log_iters == 0 and can_save:
                torch.cuda.empty_cache()
                self.save()
                can_save = False
                torch.cuda.empty_cache()

            # pbar.set_postfix({"generator loss":f"{generator_log_dict['generator_loss']:.4f}",
            #                 "critic_loss": f"{critic_log_dict['critic_loss']:.4f}"})
            # pbar.update(1)
            self.step += 1

        # Close progress bar
        # if self.is_main_process:
        self.accelerator.end_training()
        pbar.close()

    def generate_video(self, pipeline, prompts, image=None):
        batch_size = len(prompts)
        if image is not None:
            image = image.squeeze(0).unsqueeze(0).unsqueeze(2).to(device="cuda", dtype=torch.bfloat16)

            # Encode the input image as the first latent
            initial_latent = pipeline.vae.encode_to_latent(image).to(device="cuda", dtype=torch.bfloat16)
            initial_latent = initial_latent.repeat(batch_size, 1, 1, 1, 1)
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames - 1, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )
        else:
            initial_latent = None
            sampled_noise = torch.randn(
                [batch_size, self.model.num_training_frames, 16, 60, 104],
                device="cuda",
                dtype=self.dtype
            )

        video, _ = pipeline.inference(
            noise=sampled_noise,
            text_prompts=prompts,
            return_latents=True,
            initial_latent=initial_latent
        )
        current_video = video.permute(0, 1, 3, 4, 2).cpu().numpy() * 255.0
        return current_video

    def __del__(self):
        # Close tensorboard writer when trainer is destroyed
        if hasattr(self, 'writer') and self.writer is not None:
            self.writer.close()