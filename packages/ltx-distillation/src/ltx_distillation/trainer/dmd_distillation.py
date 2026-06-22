"""
DMD Distillation Training Script for LTX-2.

Usage:
    torchrun --nproc_per_node=8 -m ltx_distillation.train_distillation \
        --config_path configs/ltx2_bidirectional_dmd.yaml
"""

import argparse
import json
import math
import os
import time
from typing import Optional, Tuple, Dict, List
import gc
import logging

import torch
import torch.distributed as dist
import wandb
from omegaconf import OmegaConf

from ltx_distillation.models import LTX2DMD, LTX23DMD, CausalLTX23DMD
from ltx_distillation.datasets import TextDataset, ODERegressionLMDBDataset
from ltx_distillation.util import (
    launch_distributed_job,
    set_seed,
    init_logging_folder,
    fsdp_wrap,
    fsdp_state_dict,
    barrier,
    cycle,
)

# new
from functools import partial
import numpy as np
import random
from ltx_distillation.utils.dataset_image_video_audio import ImageVideoAudioDataset, get_random_mask
from ltx_distillation.data.ltx_unified_dataset import build_ltx_unified_dataset
from ltx_distillation.utils.bucket_sampler import (AspectRatioBatchImageVideoSampler, 
                                  ASPECT_RATIO_512, 
                                  RandomSampler, 
                                  get_closest_ratio,)
from torchvision import transforms
# from ltx_distillation.data.dataset_audio_visual_mulres_chinesewav2vec_multask_mulresf import AudioVisualDataset, get_random_mask
# from ltx_distillation.data.bucket_sampler import (ASPECT_RATIO_512, 
#                                      ASPECT_RATIO_RANDOM_CROP_512, 
#                                      AspectRatioBatchSingletaskVideoSampler, 
#                                      AspectRatioBatchMultitaskVideoSampler,
#                                      get_closest_ratio)
from torch.utils.data import RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from diffusers.optimization import get_scheduler
from tqdm import tqdm
from einops import rearrange

from ltx_core.types import Audio
from ltx_pipelines.utils.constants import DEFAULT_NEGATIVE_PROMPT
from ltx_pipelines.utils.types import PipelineComponents
from ltx_core.types import AudioLatentShape, LatentState, VideoLatentShape, VideoPixelShape
from ltx_core.components.patchifiers import get_pixel_coords

from ltx_pipelines.utils.media_io import encode_video as write_video

def collate_fn(examples, config=None):
    # Get frame num
    video_sample_n_frames = examples[0]["video_sample_n_frames"]

    # Create new output
    new_examples                 = {}
    new_examples["pixel_values"] = []
    new_examples["text"]         = []
    new_examples["file_path"]    = []
    new_examples["video_sample_n_frames"] = video_sample_n_frames
    new_examples["mask_pixel_values"] = []
    new_examples["mask"] = []
    new_examples["clip_pixel_values"] = []
    new_examples["audio_data"] = []


    # Get downsample ratio in image and videos
    pixel_value     = examples[0]["pixel_values"]
    f, h, w, c      = np.shape(pixel_value)
    random_downsample_ratio = 1
    batch_video_length = 100000

    aspect_ratio_sample_size = {key : [x / 512 * config.video_sample_size / random_downsample_ratio for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}

    closest_size, closest_ratio = get_closest_ratio(h, w, ratios=aspect_ratio_sample_size)
    closest_size = [int(x / 32) * 32 for x in closest_size]

    for example in examples:
        
        pixel_values = torch.from_numpy(example["pixel_values"]).permute(0, 3, 1, 2).contiguous()
        pixel_values = pixel_values / 255.
        batch_video_length = int(min(batch_video_length, len(pixel_values)))

        # Get adapt hw for resize
        closest_size = list(map(lambda x: int(x), closest_size))
        if closest_size[0] / h > closest_size[1] / w:
            resize_size = closest_size[0], int(w * closest_size[0] / h)
        else:
            resize_size = int(h * closest_size[1] / w), closest_size[1]
        
        transform = transforms.Compose([
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BILINEAR),  # Image.BICUBIC
            transforms.CenterCrop(closest_size),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ])
        new_examples["audio_data"].append(example["audio_data"])
        new_examples["pixel_values"].append(transform(pixel_values))
        new_examples["text"].append(example["text"])
        new_examples["file_path"].append(example.get("file_path", ""))

        mask = get_random_mask(new_examples["pixel_values"][-1].size(), image_start_only=True)
        mask_pixel_values = new_examples["pixel_values"][-1] * (1 - mask) 
        new_examples["mask_pixel_values"].append(mask_pixel_values)
        new_examples["mask"].append(mask)
        clip_pixel_values = new_examples["pixel_values"][-1][0].permute(1, 2, 0).contiguous()
        clip_pixel_values = (clip_pixel_values * 0.5 + 0.5) * 255
        new_examples["clip_pixel_values"].append(clip_pixel_values)

    # Limit the number of frames to the same
    new_examples["pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["pixel_values"]])
    new_examples["mask_pixel_values"] = torch.stack([example[:batch_video_length] for example in new_examples["mask_pixel_values"]])
    new_examples["mask"] = torch.stack([example[:batch_video_length] for example in new_examples["mask"]])
    new_examples["clip_pixel_values"] = torch.stack([example for example in new_examples["clip_pixel_values"]])
    min_length = min(example.shape[-1] for example in new_examples["audio_data"])
    new_examples["audio_data"] = torch.stack([example[:, :min_length] for example in new_examples["audio_data"]])

    return new_examples

class LTXDMDTrainer:
    """
    DMD Distillation Trainer for LTX-2.

    Handles:
    - Distributed training with FSDP
    - Alternating generator and critic training
    - Checkpointing and logging
    """

    def __init__(self, config, accelerator):
        self.config = config
        self.accelerator = accelerator
        self.step = 0
        self.step_generator = 0
        self.step_critic = 0

        global_rank = accelerator.process_index
        self.global_rank = global_rank

        world_size = accelerator.num_processes
        self.world_size = world_size
        env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
        if env_local_rank != -1:
            local_rank = env_local_rank
        self.local_rank = local_rank
        if self.local_rank == 0:
            print(f"num of all gpus: {world_size}")

        self.torch_rng = torch.Generator().manual_seed(config.seed + local_rank)

        self.dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            self.dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            self.dtype = torch.bfloat16
        self.device = accelerator.device
        self.is_main_process = accelerator.is_main_process

        self.causal = config.causal
        # self.disable_wandb = config.disable_wandb
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

        # Set seed
        if config.seed == 0:
            random_seed = torch.randint(0, 10000000, (1,), device=self.device)
            if world_size > 1:
                dist.broadcast(random_seed, src=0)
            config.seed = random_seed.item()

        set_seed(config.seed + self.global_rank)

        self.output_path = config.logdir
        self.wandb_folder = None

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

        barrier()

        # Dataloader
        self._init_dataloader(config)

        # Initialize DMD module
        if config.distribution_loss == "bidirectional_dmd_ltx23":
            self.model = LTX23DMD(config, device=self.device)
        elif config.distribution_loss == "causal_dmd_ltx23":
            self.model = CausalLTX23DMD(config, device=self.device, accelerator=self.accelerator)

        self.model.init_models()

        if getattr(config, "use_8bit_optimizer", False):
            import bitsandbytes as bnb
            adamw_cls = bnb.optim.AdamW8bit
        else:
            adamw_cls = torch.optim.AdamW

        self.generator_optimizer = adamw_cls(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.generator_lr if hasattr(config, "generator_lr") else config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        self.critic_optimizer = adamw_cls(
            [param for param in self.model.fake_score.parameters()
             if param.requires_grad],
            lr=config.critic_lr if hasattr(config, "critic_lr") else config.lr,
            betas=(config.beta1_critic, config.beta2_critic),
            weight_decay=config.weight_decay
        )

        # Learning rate schedulers
        # self.lr_scheduler_generator = self._create_lr_scheduler(self.generator_optimizer)
        # self.lr_scheduler_critic = self._create_lr_scheduler(self.critic_optimizer)
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

        # Benchmark prompts (for periodic inference visualization)
        self._init_benchmark_prompts()

        self.step = 0
        self.max_grad_norm = getattr(config, "max_grad_norm", 1.0)
        self.log_iters = int(getattr(config, "log_iters", 0))
        self.layerwise_grad_log_interval = max(
            1, int(getattr(config, "layerwise_grad_log_interval", config.log_iters))
        )
        self.previous_time = None
        self.accumulation_steps = config.accumulation_steps

        # Method-B: sequence-level KV cache state
        # seq_steps_per_update: how many segments form one sequence before a param update
        self.seq_steps_per_update = int(getattr(config, "seq_steps_per_update", 1))
        # seq_segment_prompt: optional fixed prompt used for segments beyond the first.
        # When None, each segment uses the batch's own text prompt unchanged.
        self.seq_segment_prompt: Optional[str] = getattr(config, "seq_segment_prompt", None) or None

        # multi_state_prompt_file: optional JSONL mapping file_path -> multi_state_prompts list.
        # When set, segments beyond the first use per-sample prompts from this file,
        # falling back to seq_segment_prompt (then batch text) when a key is missing.
        multi_state_prompt_file = getattr(config, "multi_state_prompt_file", None)
        self._multi_state_prompts: Dict[str, List[str]] = {}
        if multi_state_prompt_file:
            with open(multi_state_prompt_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    entry = json.loads(line)
                    fp = entry.get("file_path")
                    prompts = entry.get("multi_state_prompts")
                    if fp and prompts:
                        self._multi_state_prompts[fp] = prompts
            if self.is_main_process:
                print(f"[multi_state_prompt] Loaded {len(self._multi_state_prompts)} entries from {multi_state_prompt_file}")
        # Separate seq states for generator and critic to avoid KV cache shape mismatches
        # when the two paths operate on different batches (different resolutions/lengths).
        self._seq_state: Optional[Dict] = None          # generator path
        self._seq_state_critic: Optional[Dict] = None   # critic path

        # Per-segment visualization accumulator: holds decoded video/audio frames until
        # is_last_segment, then concatenates and writes a single multi-segment mp4.
        self._vis_accum: Dict[str, List] = {
            "gen_video": [], "gen_audio": [],
            "gen_noisy_video": [], "gen_noisy_audio": [],
            "pred_real_video": [], "pred_real_audio": [],
            "pred_fake_video": [], "pred_fake_audio": [],
            "sigma": None, "gen_sigma": None,
        }

        # Resume from a causal DMD checkpoint (full state: generator + critic + step)
        # resume_ckpt = getattr(config, "resume_checkpoint", None)
        # if resume_ckpt:
        #     if self.is_main_process:
        #         print(f"[Resume] Loading causal DMD checkpoint from {resume_ckpt}")
        #     ckpt = torch.load(resume_ckpt, map_location="cpu")
        #     self.model.generator.load_state_dict(ckpt["generator"])
        #     self.model.fake_score.load_state_dict(ckpt["critic"])
        #     self.step = ckpt.get("step", 0)
        #     if self.is_main_process:
        #         print(f"[Resume] Resumed at step {self.step}")
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
                generator_ckpt_combine_path = os.path.join(self.output_path, os.path.join(path, "model.pt"))
                generator_ckpt_path = os.path.join(self.output_path, os.path.join(path, "model_gen.pt")) 
                critic_ckpt_path = os.path.join(self.output_path, os.path.join(path, "model_critic.pt"))
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

        # maybe faster
        print("Accelerate prepare start.")
        t1 = time.time()
        self.model.generator = self.model.generator.to(self.device)
        self.model.generator = accelerator.prepare(self.model.generator)

        self.model.real_score = self.model.real_score.to(self.device)
        self.model.real_score = accelerator.prepare(self.model.real_score)

        self.model.fake_score = self.model.fake_score.to(self.device)
        self.model.fake_score = accelerator.prepare(self.model.fake_score)

        self.model.text_encoder = self.model.text_encoder.to(self.device)
        self.model.text_encoder = accelerator.prepare(self.model.text_encoder)

        self.generator_optimizer, self.critic_optimizer, self.dataloader, self.lr_scheduler_generator, self.lr_scheduler_critic = accelerator.prepare(
            self.generator_optimizer, self.critic_optimizer, self.dataloader, self.lr_scheduler_generator, self.lr_scheduler_critic
        )
        print(f"Accelerate prepare done. Time cost: {time.time() - t1:.4f}s.")


    # def _create_lr_scheduler(self, optimizer):
    #     """Create learning rate scheduler based on config.

    #     IMPORTANT: The scheduler is NOT stepped per-optimizer-call. Instead,
    #     both generator and critic schedulers are stepped once per global
    #     training step (in the training loop), so they stay synchronized
    #     even though the generator only trains every dfake_gen_update_ratio steps.

    #     Supported scheduler_type values:
    #     - None / "constant": No scheduling (constant LR)
    #     - "cosine_warmup": Linear warmup then cosine decay to min_lr
    #     """
    #     scheduler_type = getattr(self.config, "scheduler_type", None)
    #     if scheduler_type is None or scheduler_type == "constant":
    #         return None

    #     warmup_steps = getattr(self.config, "warmup_steps", 1000)
    #     max_steps = getattr(self.config, "max_steps", 30000)
    #     min_lr = getattr(self.config, "min_lr", 1e-7)
    #     base_lr = optimizer.param_groups[0]["lr"]

    #     if scheduler_type == "cosine_warmup":
    #         def lr_lambda(step):
    #             if step < warmup_steps:
    #                 return step / max(1, warmup_steps)
    #             else:
    #                 progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    #                 progress = min(progress, 1.0)
    #                 cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    #                 return max(min_lr / base_lr, cosine_decay)

    #         return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    #     else:
    #         raise ValueError(f"Unknown scheduler_type: {scheduler_type}")

    # def _validate_preinstalled_bidirectional_delegate(self) -> None:
    #     """Fail early if causal benchmark fallback would need lazy delegate construction."""
    #     if not getattr(self.model, "generator_use_causal_wrapper", False):
    #         return

    #     has_delegate = getattr(self.model.generator, "has_bidirectional_delegate", None)
    #     if callable(has_delegate) and has_delegate():
    #         return

    #     raise RuntimeError(
    #         "Causal Stage-3 generator is missing a pre-installed bidirectional delegate before FSDP "
    #         "wrapping. Install it during model init (for example from "
    #         "bootstrap_bidirectional_ckpt_path / generator_ckpt) instead of relying on lazy "
    #         "delegate construction at benchmark time."
    #     )


    def _init_dataloader(self, config):

        # Step 3: Initialize the dataloader
        dataset_type = getattr(config, "dataset_type", "legacy")
        if dataset_type == "unified":
            train_dataset = build_ltx_unified_dataset(config)
            sampler_metadata = train_dataset.dataset
            print("[DATASET] using unified dataset adapter")
        else:
            train_dataset = ImageVideoAudioDataset(
                config.train_data_meta,
                video_sample_size=config.video_sample_size,
                enable_bucket=config.enable_bucket,
                enable_inpaint=True,
            )
            sampler_metadata = train_dataset.dataset
        aspect_ratio_sample_size = {key : [x / 512 * config.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
        batch_sampler_generator = torch.Generator().manual_seed(config.seed)
        batch_sampler = AspectRatioBatchImageVideoSampler(
            sampler=RandomSampler(train_dataset, generator=batch_sampler_generator), dataset=sampler_metadata, 
            batch_size=config.batch_size, drop_last=True,
            aspect_ratios=aspect_ratio_sample_size,
        )
        self.dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_sampler=batch_sampler,
            collate_fn=partial(collate_fn, config=config),
            persistent_workers=True if config.dataloader_num_workers != 0 else False,
            num_workers=config.dataloader_num_workers,
        )
        print("###### batch size:", config.batch_size, " num clips of train dataset:", len(train_dataset), " ######")
        num_update_steps_per_epoch = math.ceil(len(train_dataset) / config.accumulation_steps)
        self.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch
        self.num_training_frames = config.video_num_training_latents
        self.frame_rate = config.frame_rate

    #     train_dataset = AudioVisualDataset(
    #         video_sample_size=config.video_sample_size,
    #         video_sample_stride=config.video_sample_stride,
    #         video_sample_n_frames=config.video_sample_n_frames,
    #         video_repeat=1,
    #         image_sample_size=config.video_sample_size,
    #         enable_bucket=True, enable_inpaint=True,
    #         dataset_paths_txt=config.dataset_paths_txt
    #     )

    #     aspect_ratio_sample_size = {key : [x / 512 * config.video_sample_size for x in ASPECT_RATIO_512[key]] for key in ASPECT_RATIO_512.keys()}
    #     # batch_sampler = AspectRatioBatchSingletaskVideoSampler(
    #     #     sampler_ai2v=RandomSampler(train_dataset.ai2v_dataset, generator=torch_rng), 
    #     #     ai2v_dataset=train_dataset.ai2v_dataset, 
    #     #     batch_size=config.batch_size, drop_last=True,
    #     #     aspect_ratios=aspect_ratio_sample_size,
    #     # )
    #     batch_sampler_generator = torch.Generator().manual_seed(config.seed)
    #     batch_sampler = AspectRatioBatchMultitaskVideoSampler(
    #         sampler_ai2v=RandomSampler(train_dataset.ai2v_dataset, generator=batch_sampler_generator), 
    #         sampler_i2v=RandomSampler(train_dataset.i2v_dataset, generator=batch_sampler_generator), 
    #         ai2v_dataset=train_dataset.ai2v_dataset, 
    #         i2v_dataset=train_dataset.i2v_dataset,
    #         batch_size=config.batch_size, drop_last=True,
    #         aspect_ratios=aspect_ratio_sample_size,
    #         ai2v_ratio=0.8,
    #     )

    #     def worker_init_fn(_seed):
    #         _seed = _seed * 256
    #         def _worker_init_fn(worker_id):
    #             print(f"worker_init_fn with {_seed + worker_id}")
    #             np.random.seed(_seed + worker_id)
    #             random.seed(_seed + worker_id)
    #         return _worker_init_fn

    #     collate_fn_configed = partial(collate_fn, args=config)

    #     self.dataloader = torch.utils.data.DataLoader(
    #         train_dataset,
    #         batch_sampler=batch_sampler,
    #         collate_fn=collate_fn_configed,
    #         persistent_workers=True if config.dataloader_num_workers != 0 else False,
    #         num_workers=config.dataloader_num_workers,
    #         # shuffle=True,
    #         worker_init_fn=worker_init_fn(config.seed + self.local_rank)
    #     )

    #     num_update_steps_per_epoch = math.ceil(len(train_dataset) / config.accumulation_steps)
    #     self.max_train_steps = config.num_train_epochs * num_update_steps_per_epoch


    def _init_benchmark_prompts(self):
        """
        Load fixed benchmark prompts from the training prompt file.

        Reads the first ``benchmark_num_prompts`` lines from ``config.data_path``
        so that every benchmark run uses exactly the same prompts for comparison.

        **All ranks** load the prompts because FSDP-wrapped models require all
        ranks to participate in forward passes during benchmark inference.
        """
        self.benchmark_enabled = False
        # config = self.config
        # self.benchmark_enabled = getattr(config, "benchmark_enabled", True)
        # self.benchmark_iters = int(getattr(config, "benchmark_iters", config.log_iters))
        # self.benchmark_seed = getattr(config, "benchmark_seed", 12345)
        # self.benchmark_num_prompts = getattr(config, "benchmark_num_prompts", 2)
        # self.benchmark_video_fps = getattr(config, "benchmark_video_fps", 24)
        # self.benchmark_audio_sample_rate = getattr(config, "benchmark_audio_sample_rate", 24000)
        # self.benchmark_mode = str(getattr(config, "benchmark_mode", "bidirectional")).lower()
        # if self.benchmark_mode not in {"bidirectional", "causal"}:
        #     if self.is_main_process:
        #         print(f"[Benchmark] Invalid benchmark_mode={self.benchmark_mode}, falling back to bidirectional.")
        #     self.benchmark_mode = "bidirectional"
        # self.benchmark_num_frame_per_block = int(getattr(config, "benchmark_num_frame_per_block", getattr(config, "num_frame_per_block", 3)))
        # self.benchmark_use_kv_cache = bool(getattr(config, "benchmark_use_kv_cache", False))
        # self.benchmark_clear_cuda_cache_per_round = bool(getattr(config, "benchmark_clear_cuda_cache_per_round", True))
        # self.benchmark_prompts = []

        # if self.benchmark_iters <= 0:
        #     self.benchmark_enabled = False
        #     if self.is_main_process:
        #         print("[Benchmark] Disabled because benchmark_iters <= 0.")

        # if self.benchmark_mode == "causal" and self.benchmark_use_kv_cache:
        #     if self.is_main_process:
        #         print(
        #             "[Benchmark] benchmark_use_kv_cache=true requested, but the current "
        #             "causal wrapper does not expose a stable KV-cache runtime API. "
        #             "Falling back to prefix-rerun autoregressive benchmark mode."
        #         )
        #     self.benchmark_use_kv_cache = False

        # if not self.benchmark_enabled:
        #     return

        # try:
        #     # When backward_simulation=false, data_path is an LMDB directory.
        #     # Use benchmark_prompt_file if specified, otherwise fall back to data_path.
        #     data_path = getattr(config, "benchmark_prompt_file", None) or config.data_path
        #     with open(data_path, "r", encoding="utf-8") as f:
        #         all_prompts = [line.strip() for line in f if line.strip()]
        #     self.benchmark_prompts = all_prompts[: self.benchmark_num_prompts]
        #     if self.is_main_process:
        #         print(f"[Benchmark] Loaded {len(self.benchmark_prompts)} prompts from {data_path}")
        #         print(f"[Benchmark] mode={self.benchmark_mode}, kv_cache={self.benchmark_use_kv_cache}, frames_per_block={self.benchmark_num_frame_per_block}")
        #         for i, p in enumerate(self.benchmark_prompts):
        #             print(f"  [{i}] {p[:80]}{'...' if len(p) > 80 else ''}")
        # except Exception as e:
        #     if self.is_main_process:
        #         print(f"[Benchmark] Failed to load prompts: {e}")
        #     self.benchmark_enabled = False

    def _vae_to_device(self):
        """Move VAEs to GPU for decoding (visualization / benchmark)."""
        if self.model.video_vae is not None:
            self.model.video_vae = self.model.video_vae.to(device=self.device)
        if self.model.audio_vae is not None:
            self.model.audio_vae = self.model.audio_vae.to(device=self.device)

    def _vae_to_cpu(self):
        """Offload VAEs back to CPU to free GPU memory."""
        if self.model.video_vae is not None:
            self.model.video_vae = self.model.video_vae.to(device="cpu")
        if self.model.audio_vae is not None:
            self.model.audio_vae = self.model.audio_vae.to(device="cpu")
        torch.cuda.empty_cache()

    # def save(self):
    #     """Save checkpoint."""
    #     print("Gathering distributed model states...")

    #     generator_state_dict = fsdp_state_dict(self.model.generator)
    #     critic_state_dict = fsdp_state_dict(self.model.fake_score)

    #     state_dict = {
    #         "generator": generator_state_dict,
    #         "critic": critic_state_dict,
    #         "step": self.step,
    #     }

    #     if self.is_main_process:
    #         checkpoint_dir = os.path.join(
    #             self.output_path,
    #             f"checkpoint_{self.step:06d}"
    #         )
    #         os.makedirs(checkpoint_dir, exist_ok=True)

    #         save_path = os.path.join(checkpoint_dir, "model.pt")
    #         torch.save(state_dict, save_path)
    #         print(f"Checkpoint saved to {save_path}")

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

        # if self.config.ema_start_step < self.step:
        #     state_dict_ema = {
        #         "generator_ema": self.generator_ema.state_dict(),
        #     }
        #     # if self.is_main_process:
        #     torch.save(state_dict_ema, os.path.join(self.output_path,
        #             f"checkpoint_model_{self.step:06d}", f"model_ema_rank_{self.global_rank}.pt"))

        print("Model saved to", os.path.join(self.output_path,
                f"checkpoint_model_{self.step:06d}", "model_xxx.pt"))
        self.accelerator.wait_for_everyone()


    @staticmethod
    def _to_scalar(value):
        """Convert tensor-like values to Python scalars for WandB logging."""
        if torch.is_tensor(value):
            if value.numel() == 1:
                return value.item()
            return value.detach().float().mean().item()
        return value

    def _compute_layerwise_grad_norms(self, module, prefix):
        """
        Compute per-layer gradient L2 norm for monitoring.

        Aggregation strategy:
        - For transformer blocks, log at block granularity: blocks.{idx}
        - For others, log at up-to-2-level module granularity.
        """
        layer_sq_norm = {}
        fsdp_prefix = "_fsdp_wrapped_module."

        for name, param in module.named_parameters():
            if param.grad is None or not param.requires_grad:
                continue

            normalized_name = name[len(fsdp_prefix):] if name.startswith(fsdp_prefix) else name
            parts = normalized_name.split(".")
            if len(parts) >= 3 and parts[1] == "blocks" and parts[2].isdigit():
                layer_key = f"blocks.{parts[2]}"
            elif len(parts) >= 2:
                layer_key = f"{parts[0]}.{parts[1]}"
            else:
                layer_key = parts[0]

            grad_sq = param.grad.detach().float().pow(2).sum().item()
            layer_sq_norm[layer_key] = layer_sq_norm.get(layer_key, 0.0) + grad_sq

        return {
            f"train/{prefix}_grad_norm/{k}": math.sqrt(v) for k, v in layer_sq_norm.items()
        }

    def adapt_batch(self, batch, prev_video_output: Optional[torch.Tensor] = None,
                    prompt_override: Optional[str] = None):
        """Encode one batch of raw video/audio into LatentState dicts.

        When prev_video_output is given (shape [B, SeqLen, C]), the last
        token_per_frame tokens of the previous segment are used as the
        conditioning frame of this segment instead of the batch's own first frame.
        This implements Method-B's cross-segment conditioning.

        When prompt_override is given, it replaces every sample's text prompt in
        the batch (useful for giving subsequent segments a fixed prompt such as a
        continuation description).
        """
        B, F, C, H, W = batch["pixel_values"].shape
        num_frames = F
        width = W
        height = H
        # encode audio stream
        audio_data = Audio(batch["audio_data"].to(self.device, self.dtype), sampling_rate=16000)
        with torch.no_grad():
            audio_clean_latent = self.model.audio_vae.encode(audio_data.to(device=self.device))
        audio_clean_latent = rearrange(audio_clean_latent, "b c t f -> b t (c f)")
        
        # encode video stream
        with torch.no_grad():
            video_clean_latent = self.model.video_vae.encode(
                rearrange(batch["pixel_values"], "b f c h w -> b c f h w").to(device=self.device, dtype=self.dtype))
        torch.cuda.empty_cache()
        
        B, C, F, H, W = video_clean_latent.shape
        # latent_num_frames = F
        token_per_frame = H * W
        video_clean_latent = rearrange(video_clean_latent, "b c t h w -> b (t h w) c")

        # only use the first frame, no regression loss
        latent_num_frames = self.num_training_frames
        num_frames = (latent_num_frames - 1) * 8 + 1
        seq_len = latent_num_frames * token_per_frame

        if prev_video_output is not None:
            # Use the last frame of the previous segment as the conditioning frame.
            prev_last_frame = prev_video_output[:, -token_per_frame:].detach()
            video_clean_latent = torch.cat([
                prev_last_frame,
                torch.randn([video_clean_latent.shape[0], seq_len - token_per_frame, video_clean_latent.shape[2]],
                            dtype=self.dtype, device=self.device),
            ], dim=1)
        else:
            video_clean_latent = torch.cat([video_clean_latent[:, :token_per_frame],
                                            torch.randn([video_clean_latent.shape[0], seq_len - token_per_frame, video_clean_latent.shape[2]], dtype=self.dtype, device=self.device)], dim=1)

        # clip audio to match video length
        video_pixel_shape = VideoPixelShape(batch=video_clean_latent.shape[0], frames=num_frames, width=width, height=height, fps=self.frame_rate)
        audio_latent_shape = AudioLatentShape.from_video_pixel_shape(video_pixel_shape)
        audio_clean_latent = audio_clean_latent[:, :audio_latent_shape.frames]

        # only use the first frame, no regression loss
        audio_clean_latent = torch.randn([audio_clean_latent.shape[0], audio_latent_shape.frames, audio_clean_latent.shape[2]], dtype=self.dtype, device=self.device)

        # encode text prompt
        v_context_p_list = []
        a_context_p_list = []
        v_context_n_list = []
        a_context_n_list = []
        prompts = (
            prompt_override if isinstance(prompt_override, list)
            else [prompt_override] * len(batch["text"]) if prompt_override is not None
            else batch["text"]
        )
        with torch.no_grad():
            for prompt in prompts:
                ctx_p, ctx_n =self.model.text_encoder(
                    [prompt, DEFAULT_NEGATIVE_PROMPT],
                    device=self.device,
                )
                v_context_p, a_context_p = ctx_p.video_encoding, ctx_p.audio_encoding
                v_context_n, a_context_n = ctx_n.video_encoding, ctx_n.audio_encoding
                v_context_p_list.append(v_context_p)
                a_context_p_list.append(a_context_p)
                v_context_n_list.append(v_context_n)
                a_context_n_list.append(a_context_n)
        v_context_p = torch.cat(v_context_p_list, dim=0)
        a_context_p = torch.cat(a_context_p_list, dim=0)
        v_context_n = torch.cat(v_context_n_list, dim=0)
        a_context_n = torch.cat(a_context_n_list, dim=0)
        
        # noisy video and audio
        noisy_video_latent = torch.randn(
            video_clean_latent.shape,
            dtype=self.dtype,
            generator=self.torch_rng,
        ).to(device=self.device)
        noisy_audio_latent = torch.randn(
            audio_clean_latent.shape,
            dtype=self.dtype,
            generator=self.torch_rng,
        ).to(device=self.device)

        # denoise_mask
        video_denoise_mask = torch.ones(
            video_clean_latent.shape[:2]+(1,),
            device=self.device,
            dtype=torch.float32,
        )
        video_denoise_mask[:, :token_per_frame] = 0  # only the first frame is clean
        audio_denoise_mask = torch.ones(
            audio_clean_latent.shape[:2]+(1,),
            device=self.device,
            dtype=torch.float32,
        )
        
        # video_position
        components = PipelineComponents(dtype=self.dtype, device=self.device)
        video_latent_shape = VideoLatentShape.from_pixel_shape(
            shape=video_pixel_shape,
            latent_channels=components.video_latent_channels,
            scale_factors=components.video_scale_factors,
        )
        video_patchifier = components.video_patchifier
        video_latent_coords = video_patchifier.get_patch_grid_bounds(
            output_shape=video_latent_shape,
            device=self.device,
        )
        video_positions = get_pixel_coords(
            latent_coords=video_latent_coords,
            scale_factors=components.video_scale_factors,
            causal_fix=True,
        ).float()
        video_positions[:, 0, ...] = video_positions[:, 0, ...] / self.frame_rate
        
        # audio_position
        audio_patchifier = components.audio_patchifier
        audio_latent_coords = audio_patchifier.get_patch_grid_bounds(
            output_shape=audio_latent_shape,
            device=self.device,
        )
        audio_positions = audio_latent_coords
        
        # build video and audio state
        with torch.no_grad():
            initial_video_latent_state = LatentState(
                latent=noisy_video_latent.detach(),
                denoise_mask=video_denoise_mask.detach(),
                positions=video_positions.detach(),
                clean_latent=video_clean_latent.detach(),
                attention_mask=None,
            )
            initial_audio_latent_state = LatentState(
                latent=noisy_audio_latent.detach(),
                denoise_mask=audio_denoise_mask.detach(),
                positions=audio_positions.detach(),
                clean_latent=audio_clean_latent.detach(),
                attention_mask=None,
            )
            
            # build conditional dict
            conditional_dict = {
                "v_context": v_context_p.detach(),
                "a_context": a_context_p.detach(),
            }
            unconditional_dict = {
                "v_context": v_context_n.detach(),
                "a_context": a_context_n.detach(),
            }
        
        new_batch = {
            "conditional_dict": conditional_dict,
            "unconditional_dict": unconditional_dict,
            "initial_video_latent_state": initial_video_latent_state,
            "initial_audio_latent_state": initial_audio_latent_state,
            "video_latent_num_frames": latent_num_frames,
            "video_num_frames": num_frames,
            "width": width,
            "height": height,
            "video_latent_shape": video_latent_shape,
            "audio_latent_shape": audio_latent_shape,
        }

        return new_batch

    def fwdbwd_one_step(self, batch, train_generator):
        """Execute one training step without dataloader.

        Method-B: KV-cache-persistent multi-segment training.
        For the first (seq_steps_per_update-1) segments of each sequence we run
        with compute_grad=False and accumulate no gradients.  Only on the last
        segment do we enable grad and call backward().
        """

        if train_generator:
            self.step_generator += 1
        else:
            self.step_critic += 1

        self.model.current_step = self.step
        config = self.config
        LOG_LAYERWISE_GRAD = False
        if self.step % 20 == 0:
            torch.cuda.empty_cache()

        # Determine where we are in the current sequence.
        _state = self._seq_state if train_generator else self._seq_state_critic
        if _state is None:
            seg_idx = 0
        else:
            seg_idx = _state["segment_idx"]

        # Previous outputs used as prefix for this segment.
        prev_video_output = _state["video_output"] if _state is not None else None
        persistent_kv = _state["kv_cache_list"] if _state is not None else None
        segment_video_offset = _state["video_offset"] if _state is not None else 0
        prev_video_seqlen_frame = _state["video_seqlen_frame"] if _state is not None else None
        shared_exit_step = _state["exit_step"] if _state is not None else None

        # Multi-segment sequences must use the same (H, W) throughout, but bucket
        # sampling can produce a different resolution for each new batch drawn from
        # the dataloader. Between two consecutive generator calls (separated by
        # dfake_gen_update_ratio critic steps) the dataloader has advanced and the
        # new batch may belong to a different bucket.  Pin the batch to the one used
        # at seg_idx==0 so that (H, W) is always consistent across all segments of a
        # sequence.  This prevents needs_reset from firing mid-sequence, which was
        # causing seg_idx to be reset to 0 every time and is_last_segment to never
        # become True (blocking generator updates and visualization entirely).
        if seg_idx > 0 and _state is not None and "seg0_batch" in _state:
            batch = _state["seg0_batch"]

        # Bucket sampling can change (H, W) across batches. The persistent KV cache
        # and prev_video_output encode a specific spatial layout (token-per-frame);
        # reusing them at a different resolution writes new tokens past the buffer
        # end (and aliases spatial coords to wrong positions). When the resolution
        # changes mid-sequence, abandon prior state and treat this batch as the
        # start of a fresh sequence.
        #
        # The reset decision MUST be synchronized across ranks: each rank sees its
        # own batch shape, but the no-cache vs persistent-cache forward paths drive
        # different FSDP collective sequences. A per-rank decision causes NCCL
        # deadlock. all_reduce(max) so any single rank's shape change resets all.
        cur_pixel_shape = tuple(batch["pixel_values"].shape[-2:])
        prev_pixel_shape = _state.get("pixel_shape") if _state is not None else None
        needs_reset_local = int(prev_pixel_shape is not None and prev_pixel_shape != cur_pixel_shape)
        if dist.is_available() and dist.is_initialized():
            t = torch.tensor(needs_reset_local, device=self.device, dtype=torch.int32)
            dist.all_reduce(t, op=dist.ReduceOp.MAX)
            needs_reset = bool(t.item())
        else:
            needs_reset = bool(needs_reset_local)
        if needs_reset:
            prev_video_output = None
            persistent_kv = None
            segment_video_offset = 0
            prev_video_seqlen_frame = None
            shared_exit_step = None
            seg_idx = 0
            if train_generator:
                self._seq_state = None
            else:
                self._seq_state_critic = None

        is_last_segment = (seg_idx == self.seq_steps_per_update - 1)

        # adapt batch (uses prev_video_output as conditioning frame if given)
        # For segments beyond the first, look up per-sample multi_state_prompts if available,
        # then fall back to seq_segment_prompt, then batch text.
        if seg_idx > 0 and self._multi_state_prompts:
            file_paths = batch.get("file_path", [])
            # Build per-sample prompt list aligned to the batch
            segment_prompts: List[Optional[str]] = []
            for fp in file_paths:
                prompts = self._multi_state_prompts.get(fp)
                if prompts and seg_idx - 1 < len(prompts):
                    segment_prompts.append(prompts[seg_idx - 1])
                else:
                    segment_prompts.append(self.seq_segment_prompt)
            # If all entries are identical (or list is empty) pass a single scalar;
            # otherwise pass the per-sample list to adapt_batch.
            if not segment_prompts:
                segment_prompt: Optional[object] = self.seq_segment_prompt
            elif len(set(segment_prompts)) == 1:
                segment_prompt = segment_prompts[0]
            else:
                segment_prompt = segment_prompts
        elif seg_idx > 0:
            segment_prompt = self.seq_segment_prompt
        else:
            segment_prompt = None
        new_batch = self.adapt_batch(batch, prev_video_output=prev_video_output,
                                     prompt_override=segment_prompt)

        conditional_dict = new_batch["conditional_dict"]
        unconditional_dict = new_batch["unconditional_dict"]
        video_state = new_batch["initial_video_latent_state"]
        audio_state = new_batch["initial_audio_latent_state"]
        video_latent_num_frames = new_batch["video_latent_num_frames"]
        width = new_batch["width"]
        height = new_batch["height"]
        video_num_frames = new_batch["video_num_frames"]

        # Train generator
        if train_generator:

            # Replayed back-prop (RELIC §4.4.2): pass loss_scale; the model does
            # per-block .backward() internally and accumulates gradients. The
            # returned loss is a detached scalar for logging only — do NOT call
            # .backward() on it again.
            generator_loss_scale = 1.0 / (self.accumulation_steps * self.seq_steps_per_update)
            generator_loss, generator_log_dict, updated_kv = self.model.generator_loss_segment(
                video_state=video_state,
                audio_state=audio_state,
                video_latent_num_frames=video_latent_num_frames,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                persistent_kv_cache_list=persistent_kv,
                segment_video_offset=segment_video_offset,
                prev_video_seqlen_frame=prev_video_seqlen_frame,
                loss_scale=generator_loss_scale,
                shared_exit_step=shared_exit_step,
            )

            # Extract the generated video for use as the next segment's conditioning frame.
            gen_video_output = generator_log_dict.get("dmdtrain_clean_latent_video", None)
            gen_audio_output = generator_log_dict.get("dmdtrain_clean_latent_audio", None)

            cur_video_seqlen_frame = video_state.latent.shape[1] // video_latent_num_frames

            # Update sequence state (carries kv cache and prefix frame to the next segment).
            self._seq_state = {
                "kv_cache_list": updated_kv,
                "video_output": gen_video_output.detach() if gen_video_output is not None else prev_video_output,
                "audio_output": gen_audio_output.detach() if gen_audio_output is not None else None,
                "segment_idx": (seg_idx + 1) % self.seq_steps_per_update,
                "video_offset": segment_video_offset + video_latent_num_frames,
                "video_seqlen_frame": cur_video_seqlen_frame,
                "pixel_shape": cur_pixel_shape,
                "seg0_batch": batch if seg_idx == 0 else _state.get("seg0_batch"),
                "exit_step": generator_log_dict.get("exit_step", shared_exit_step),
            }
            if is_last_segment:
                self._seq_state = None

            update_params_flag_generator = False
            update_params_flag_critic = False

            # generator_loss_segment performs per-block backward internally
            # (RELIC replayed back-prop) and already applied generator_loss_scale.
            # `generator_loss` is a detached scalar for logging only.
            scaled_loss = None

            # --- Per-segment visualization decode (runs every segment) ---
            # Decide at seg_idx==0 whether this sequence will be visualized, so
            # all segments use the same VISUALIZE flag despite step_generator advancing.
            if seg_idx == 0:
                # step_generator is already incremented before this call, so at seg_idx==0
                # it's odd. The last segment (seg_idx==1) will see the even value.
                # We want to visualize when the sequence completes a param update
                # (every accumulation_steps generator calls), so check if this sequence
                # will land on an even step_generator at seg_idx==0.
                next_gen_step = self.step_generator + (self.config.seq_steps_per_update - 1)
                self._vis_accum["visualize"] = (next_gen_step % (2 * self.accumulation_steps) == 0)
                for k in ("gen_video", "gen_audio", "gen_noisy_video", "gen_noisy_audio",
                          "pred_real_video", "pred_real_audio", "pred_fake_video", "pred_fake_audio"):
                    self._vis_accum[k] = []
                self._vis_accum["sigma"] = None
                self._vis_accum["gen_sigma"] = None

            if self._vis_accum["visualize"] and self.is_main_process:
                torch.cuda.empty_cache()
                os.makedirs(os.path.join(self.config.logdir, "sample_video"), exist_ok=True)
                self._vis_accum["gen_sigma"] = generator_log_dict["dmdtrain_generator_sigma"]
                self._vis_accum["sigma"] = generator_log_dict["dmdtrain_noise_sigma"]
                with torch.no_grad():
                    gen_video = generator_log_dict["dmdtrain_clean_latent_video"][:1].to(device=self.device, dtype=self.dtype)
                    gen_audio = generator_log_dict["dmdtrain_clean_latent_audio"][:1].to(device=self.device, dtype=self.dtype)
                    self._vis_accum["gen_video"].append(
                        self.model.video_vae.decode_to_visualize(gen_video,
                            num_frames=video_num_frames, width=width, height=height, frame_rate=self.frame_rate))
                    self._vis_accum["gen_audio"].append(
                        self.model.audio_vae.decode_to_visualize(gen_audio,
                            num_frames=video_num_frames, width=width, height=height, frame_rate=self.frame_rate))

                    gen_video_noisy = generator_log_dict["dmdtrain_noisy_latent_video"][:1].to(device=self.device, dtype=self.dtype)
                    gen_audio_noisy = generator_log_dict["dmdtrain_noisy_latent_audio"][:1].to(device=self.device, dtype=self.dtype)
                    self._vis_accum["gen_noisy_video"].append(
                        self.model.video_vae.decode_to_visualize(gen_video_noisy,
                            num_frames=video_num_frames, width=width, height=height, frame_rate=self.frame_rate))
                    self._vis_accum["gen_noisy_audio"].append(
                        self.model.audio_vae.decode_to_visualize(gen_audio_noisy,
                            num_frames=video_num_frames, width=width, height=height, frame_rate=self.frame_rate))

                    pred_real_video = generator_log_dict["dmdtrain_pred_real_video"][:1].to(device=self.device, dtype=self.dtype)
                    pred_real_audio = generator_log_dict["dmdtrain_pred_real_audio"][:1].to(device=self.device, dtype=self.dtype)
                    self._vis_accum["pred_real_video"].append(
                        self.model.video_vae.decode_to_visualize(pred_real_video,
                            num_frames=video_num_frames, width=width, height=height, frame_rate=self.frame_rate))
                    self._vis_accum["pred_real_audio"].append(
                        self.model.audio_vae.decode_to_visualize(pred_real_audio,
                            num_frames=video_num_frames, width=width, height=height, frame_rate=self.frame_rate))

                    pred_fake_video = generator_log_dict["dmdtrain_pred_fake_video"][:1].to(device=self.device, dtype=self.dtype)
                    pred_fake_audio = generator_log_dict["dmdtrain_pred_fake_audio"][:1].to(device=self.device, dtype=self.dtype)
                    self._vis_accum["pred_fake_video"].append(
                        self.model.video_vae.decode_to_visualize(pred_fake_video,
                            num_frames=video_num_frames, width=width, height=height, frame_rate=self.frame_rate))
                    self._vis_accum["pred_fake_audio"].append(
                        self.model.audio_vae.decode_to_visualize(pred_fake_audio,
                            num_frames=video_num_frames, width=width, height=height, frame_rate=self.frame_rate))

            if is_last_segment:
                generator_layerwise_grad_dict = (
                    self._compute_layerwise_grad_norms(self.model.generator, "generator")
                    if LOG_LAYERWISE_GRAD else {}
                )
                if hasattr(self.model.generator, 'clip_grad_norm_'):
                    generator_grad_norm = self.model.generator.clip_grad_norm_(self.max_grad_norm)
                else:
                    generator_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.generator.parameters(), self.max_grad_norm
                    )

                generator_log_dict.update({
                    "generator_loss": generator_loss.detach().item(),
                    "generator_grad_norm": generator_grad_norm.detach().item(),
                })

                # Write concatenated multi-segment videos after the last segment.
                if self._vis_accum["visualize"] and self.is_main_process:
                    def _concat_audio(audio_list):
                        waveforms = [a.waveform for a in audio_list]
                        return Audio(waveform=torch.cat(waveforms, dim=-1), sampling_rate=audio_list[0].sampling_rate)

                    gen_sigma_str = f"{self._vis_accum['gen_sigma']:.2f}"
                    sigma_str = f"{self._vis_accum['sigma']:.2f}"
                    save_dir = os.path.join(self.config.logdir, "sample_video")

                    write_video(
                        video=torch.cat(self._vis_accum["gen_video"], dim=0),
                        audio=_concat_audio(self._vis_accum["gen_audio"]),
                        fps=self.frame_rate,
                        output_path=os.path.join(save_dir, f"pred_gen_{self.step}_{gen_sigma_str}.mp4"),
                        video_chunks_number=1)
                    write_video(
                        video=torch.cat(self._vis_accum["gen_noisy_video"], dim=0),
                        audio=_concat_audio(self._vis_accum["gen_noisy_audio"]),
                        fps=self.frame_rate,
                        output_path=os.path.join(save_dir, f"pred_gen_noisy_{self.step}_{sigma_str}.mp4"),
                        video_chunks_number=1)
                    write_video(
                        video=torch.cat(self._vis_accum["pred_real_video"], dim=0),
                        audio=_concat_audio(self._vis_accum["pred_real_audio"]),
                        fps=self.frame_rate,
                        output_path=os.path.join(save_dir, f"pred_real_{self.step}_{sigma_str}.mp4"),
                        video_chunks_number=1)
                    write_video(
                        video=torch.cat(self._vis_accum["pred_fake_video"], dim=0),
                        audio=_concat_audio(self._vis_accum["pred_fake_audio"]),
                        fps=self.frame_rate,
                        output_path=os.path.join(save_dir, f"pred_fake_{self.step}_{sigma_str}.mp4"),
                        video_chunks_number=1)

                if self.step_generator % self.accumulation_steps == 0:
                    update_params_flag_generator = True

                del generator_grad_norm
            else:
                generator_log_dict.setdefault("generator_loss", generator_loss.detach().item())

            del scaled_loss
            torch.cuda.empty_cache()

            return generator_log_dict, update_params_flag_generator, update_params_flag_critic

        else:
            # Train critic
            critic_loss, critic_log_dict, updated_kv = self.model.critic_loss_segment(
                video_state=video_state,
                audio_state=audio_state,
                video_latent_num_frames=video_latent_num_frames,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
                persistent_kv_cache_list=persistent_kv,
                segment_video_offset=segment_video_offset,
                prev_video_seqlen_frame=prev_video_seqlen_frame,
                shared_exit_step=shared_exit_step,
            )

            cur_video_seqlen_frame = video_state.latent.shape[1] // video_latent_num_frames

            # Update sequence state for critic path.
            curr_video = (self._seq_state_critic["video_output"]
                          if self._seq_state_critic is not None else prev_video_output)
            self._seq_state_critic = {
                "kv_cache_list": updated_kv,
                "video_output": curr_video,
                "audio_output": None,
                "segment_idx": (seg_idx + 1) % self.seq_steps_per_update,
                "video_offset": segment_video_offset + video_latent_num_frames,
                "video_seqlen_frame": cur_video_seqlen_frame,
                "pixel_shape": cur_pixel_shape,
                "seg0_batch": batch if seg_idx == 0 else _state.get("seg0_batch"),
                "exit_step": critic_log_dict.get("exit_step", shared_exit_step),
            }
            if is_last_segment:
                self._seq_state_critic = None

            update_params_flag_generator = False
            update_params_flag_critic = False

            # Every segment contributes gradients; divide by total steps in the sequence.
            scaled_critic_loss = critic_loss / (self.accumulation_steps * self.seq_steps_per_update)
            scaled_critic_loss.backward()

            if is_last_segment:
                critic_layerwise_grad_dict = (
                    self._compute_layerwise_grad_norms(self.model.fake_score, "critic")
                    if LOG_LAYERWISE_GRAD else {}
                )
                if hasattr(self.model.fake_score, 'clip_grad_norm_'):
                    critic_grad_norm = self.model.fake_score.clip_grad_norm_(self.max_grad_norm)
                else:
                    critic_grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.model.fake_score.parameters(), self.max_grad_norm
                    )

                critic_log_dict.update({"critic_loss": critic_loss.item(),
                                        "critic_grad_norm": critic_grad_norm.item()})

                del critic_grad_norm
                torch.cuda.empty_cache()

                if self.step_critic % self.accumulation_steps == 0:
                    update_params_flag_critic = True
            else:
                critic_log_dict.setdefault("critic_loss", critic_loss.item())
                critic_log_dict.setdefault("critic_grad_norm", 0.0)

            del scaled_critic_loss, critic_loss
            torch.cuda.empty_cache()

            return critic_log_dict, update_params_flag_generator, update_params_flag_critic

    def train(self):
        """Main training loop."""

        # Set all models to eval mode first (disables dropout/batchnorm),
        # then re-enable train mode for generator and fake_score so that
        # gradient checkpointing remains active during their gradient-enabled
        # forward passes. This is critical for the 19B model's memory footprint.
        # The real_score (teacher) stays in eval mode since it's frozen.
        #
        # For backward simulation's @torch.no_grad() forward passes, the
        # generator is temporarily switched to eval() inside
        # _consistency_backward_simulation() to avoid FSDP+checkpoint conflicts.
        self.model.eval()
        self.model.generator.train()
        self.model.fake_score.train()

        # set up training loop
        start_step = self.step
        pbar = tqdm(
            range(0, self.max_train_steps),
            initial=start_step,
            desc="Steps",
            # Only show the progress bar once on each machine.
            disable=not self.accelerator.is_local_main_process,
        )
        train_gen_loss = 0.0
        train_critic_loss = 0.0
        local_dataloader_iterator = iter(self.dataloader)
        can_save = True

        # training loop
        while True:

            TRAIN_GENERATOR = self.step % self.config.dfake_gen_update_ratio == 0
            did_update_generator = False

            if TRAIN_GENERATOR:
                # data loading
                try:
                    batch = next(local_dataloader_iterator)
                except StopIteration:
                    # 当数据取完时，重新设置迭代器
                    print("数据迭代完，开始新的循环...")
                    local_dataloader_iterator = iter(self.dataloader)  # 重新创建迭代器
                    batch = next(local_dataloader_iterator)

                # forward and backward pass
                generator_log_dict, update_params_flag_generator, update_params_flag_critic = self.fwdbwd_one_step(batch, True)
                did_update_generator = update_params_flag_generator

                if update_params_flag_generator:
                    self.generator_optimizer.step()
                    self.generator_optimizer.zero_grad() # set_to_none=True
                    self.lr_scheduler_generator.step()
                    can_save = True
                    if self.is_main_process:
                        # Update progress bar
                        pbar.update(1)
                        pbar.write(f"generator loss: {generator_log_dict['generator_loss']:.4f}, generator video loss: {generator_log_dict['video_dmd_loss'].cpu().item():.4f}, generator audio loss: {generator_log_dict['audio_dmd_loss'].cpu().item():.4f}, critic_loss: -1")

            try:
                batch = next(local_dataloader_iterator)
            except StopIteration:
                # 当数据取完时，重新设置迭代器
                print("数据迭代完，开始新的循环...")
                local_dataloader_iterator = iter(self.dataloader)  # 重新创建迭代器
                batch = next(local_dataloader_iterator)
            critic_log_dict, update_params_flag_generator, update_params_flag_critic = self.fwdbwd_one_step(batch, False)

            if update_params_flag_critic:
                self.critic_optimizer.step()
                self.critic_optimizer.zero_grad() # set_to_none=True
                self.lr_scheduler_critic.step()
                can_save = True
                if self.is_main_process:
                    pbar.update(1)
                    pbar.write(f"generator loss: -1, critic_loss: {critic_log_dict['critic_loss']:.4f}")

            # Save checkpoint
            # Logging
            if self.step % self.config.gc_interval == 0:
                if self.local_rank == 0:
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

                if self.writer is not None:
                    if TRAIN_GENERATOR and did_update_generator:
                        self.writer.add_scalar("generator_loss", generator_log_dict["generator_loss"], self.step)
                        self.writer.add_scalar("generator_video_loss", generator_log_dict["video_dmd_loss"].cpu().item(), self.step)
                        self.writer.add_scalar("generator_audio_loss", generator_log_dict["audio_dmd_loss"].cpu().item(), self.step)
                        self.writer.add_scalar("generator_grad_norm", generator_log_dict["generator_grad_norm"], self.step)

                    self.writer.add_scalar("critic_loss", critic_log_dict["critic_loss"], self.step)
                    self.writer.add_scalar("critic_grad_norm", critic_log_dict["critic_grad_norm"], self.step)

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

        self.accelerator.end_training()
        pbar.close()

    @torch.no_grad()
    def _run_benchmark_and_log(self):
        """
        Run 4-step inference on fixed benchmark prompts, distributing work
        across all ranks for maximum parallelism.

        **All ranks** must call this method because the generator and text
        encoder are FSDP-wrapped and require collective communication.

        Flow (per round, one prompt per rank):
        1. ALL ranks: encode 1 prompt each (FSDP collective, batch_size=1)
        2. ALL ranks: run inference pipeline (FSDP collective, batch_size=1)
        3. ALL ranks: decode video/audio with local VAE, save mp4 to shared FS
        4. Rank 0: collect all saved files, log to WandB

        This distributes N prompts across W ranks in ceil(N/W) rounds,
        reducing per-rank memory vs the old single-rank-decodes-all approach.

        RNG is forked per prompt for reproducibility without affecting training.
        """
        from ltx_distillation.inference.bidirectional_pipeline import (
            BidirectionalAVInferencePipeline,
        )
        from ltx_distillation.inference.causal_pipeline import (
            CausalAVInferencePipeline,
        )

        config = self.config

        # Free training intermediate memory before benchmark
        torch.cuda.empty_cache()

        num_prompts = len(self.benchmark_prompts)
        num_rounds = math.ceil(num_prompts / self.world_size)

        if self.is_main_process:
            print(
                f"[Benchmark] Step {self.step}: generating {num_prompts} samples "
                f"({self.benchmark_mode} mode) across {self.world_size} ranks "
                f"({num_rounds} round(s))..."
            )

        step_dir = os.path.join(
            self.output_path, "benchmark", f"step_{self.step:07d}"
        )
        os.makedirs(step_dir, exist_ok=True)

        video_shape_single, audio_shape_single = compute_latent_shapes(
            num_frames=config.video_sample_n_frames,
            video_height=config.video_height,
            video_width=config.video_width,
            batch_size=1,
        )

        # Keep Stage 3 benchmark aligned with the Stage-2 ODE benchmark:
        # temporarily switch the FSDP-wrapped generator to eval() under no_grad,
        # then restore the previous mode afterwards.
        was_training = self.model.generator.training
        self.model.generator.eval()
        try:
            if self.benchmark_mode == "causal":
                pipeline = CausalAVInferencePipeline(
                    generator=self.model.generator,
                    add_noise_fn=self.model.add_noise,
                    denoising_sigmas=self.model.denoising_sigmas,
                    num_frame_per_block=self.benchmark_num_frame_per_block,
                    use_kv_cache=self.benchmark_use_kv_cache,
                    clear_cuda_cache_per_round=self.benchmark_clear_cuda_cache_per_round,
                )
            else:
                pipeline = BidirectionalAVInferencePipeline(
                    generator=self.model.generator,
                    add_noise_fn=self.model.add_noise,
                    denoising_sigmas=self.model.denoising_sigmas,
                )

            self._vae_to_device()

            # Timing: wall-clock for full benchmark, and per-video generation time
            benchmark_wall_start = time.perf_counter()
            my_total_generate_seconds = 0.0

            for round_idx in range(num_rounds):
                prompt_idx = round_idx * self.world_size + self.global_rank
                has_real_prompt = prompt_idx < num_prompts

                if has_real_prompt:
                    my_prompt = [self.benchmark_prompts[prompt_idx]]
                else:
                    my_prompt = [self.benchmark_prompts[0]]

                with torch.no_grad():
                    conditional_dict = self.model.text_encoder(text_prompts=my_prompt)

                prompt_seed = self.benchmark_seed + prompt_idx
                with torch.random.fork_rng(devices=[self.device]):
                    torch.manual_seed(prompt_seed)
                    torch.cuda.manual_seed(prompt_seed)

                    gen_start = time.perf_counter()
                    video_latent, audio_latent = pipeline.generate(
                        video_shape=tuple(video_shape_single),
                        audio_shape=tuple(audio_shape_single),
                        conditional_dict=conditional_dict,
                    )
                    gen_elapsed = time.perf_counter() - gen_start
                    my_total_generate_seconds += gen_elapsed

                if has_real_prompt:
                    self._decode_and_save_sample(
                        video_latent=video_latent,
                        audio_latent=audio_latent,
                        prompt_idx=prompt_idx,
                        step_dir=step_dir,
                    )

                del video_latent, audio_latent, conditional_dict
                if self.benchmark_clear_cuda_cache_per_round:
                    torch.cuda.empty_cache()

                barrier()
        finally:
            if was_training:
                self.model.generator.train()

        benchmark_wall_elapsed = time.perf_counter() - benchmark_wall_start

        # Gather total generation time from all ranks (each rank sums its own generate times)
        total_generate_tensor = torch.tensor(
            [my_total_generate_seconds], device=self.device, dtype=torch.float64
        )
        dist.all_reduce(total_generate_tensor, op=dist.ReduceOp.SUM)
        total_generate_seconds = total_generate_tensor.item()

        self._vae_to_cpu()

        barrier()

        # ---- Rank 0: log all samples to WandB and print benchmark timing ----
        if self.is_main_process:
            time_per_video_wall = benchmark_wall_elapsed / max(1, num_prompts)
            time_per_video_generate = total_generate_seconds / max(1, num_prompts)

            benchmark_wandb_dict = {}
            prompt_rows = []

            for idx in range(num_prompts):
                sample_path = os.path.join(step_dir, f"sample_{idx}.mp4")
                if os.path.exists(sample_path):
                    benchmark_wandb_dict[f"benchmark/sample_{idx}"] = wandb.Video(
                        sample_path, fps=self.benchmark_video_fps, format="mp4"
                    )
                    prompt_rows.append(
                        [idx, self.benchmark_prompts[idx], sample_path]
                    )

            if prompt_rows:
                table = wandb.Table(
                    columns=["index", "prompt", "local_path"],
                    data=prompt_rows,
                )
                benchmark_wandb_dict["benchmark/prompt_table"] = table

            if benchmark_wandb_dict:
                wandb.log(benchmark_wandb_dict, step=self.step)

            # One line: timing + save path (flush so it always appears in logs)
            print(
                f"[Benchmark] Step {self.step}: {num_prompts} video(s) | "
                f"wall {benchmark_wall_elapsed:.2f}s ({time_per_video_wall:.2f}s/video) | "
                f"generate {total_generate_seconds:.2f}s ({time_per_video_generate:.2f}s/video) | "
                f"saved to {step_dir}",
                flush=True,
            )

        barrier()

    def _decode_and_save_sample(
        self,
        video_latent: torch.Tensor,
        audio_latent: torch.Tensor,
        prompt_idx: int,
        step_dir: str,
    ):
        """
        Decode one (video, audio) latent pair and save as mp4 with audio.

        Called by every rank that owns a real benchmark prompt.  VAEs must
        already be on GPU (via ``_vae_to_device``) before calling this.
        """
        # Decode video → pixel  [1, C, F, H, W]  →  [0, 1]
        video_pixel = self.model.video_vae.decode_to_pixel(video_latent)

        # Decode audio → waveform  [1, 1, samples]
        audio_waveform = None
        try:
            audio_waveform = self.model.audio_vae.decode_to_waveform(audio_latent)
        except Exception as e:
            print(
                f"[Benchmark][Rank {self.global_rank}] Audio decode failed "
                f"for prompt {prompt_idx}: {e}"
            )

        # Prepare video tensor: -> uint8 [F, H, W, C]
        vid = video_pixel[0]  # [C, F, H, W]
        if vid.shape[0] == 3:
            vid = vid.permute(1, 0, 2, 3)  # -> [F, C, H, W]
        vid = vid.permute(0, 2, 3, 1)  # -> [F, H, W, C]
        vid = (vid.clamp(0, 1) * 255).cpu().to(torch.uint8)

        sample_path = os.path.join(step_dir, f"sample_{prompt_idx}.mp4")

        # Try writing mp4 with embedded audio track
        written_with_audio = False
        if audio_waveform is not None:
            try:
                wav_float = audio_waveform[0].cpu().float()  # [1, samples]
                from torchvision.io import write_video

                write_video(
                    sample_path,
                    vid,
                    fps=self.benchmark_video_fps,
                    audio_array=wav_float,
                    audio_fps=self.benchmark_audio_sample_rate,
                    audio_codec="aac",
                )
                written_with_audio = True
            except Exception as e:
                print(
                    f"[Benchmark][Rank {self.global_rank}] write_video with "
                    f"audio failed for prompt {prompt_idx}: {e}"
                )

        # Fallback: silent video + separate wav
        if not written_with_audio:
            try:
                from torchvision.io import write_video

                write_video(sample_path, vid, fps=self.benchmark_video_fps)
            except Exception as e:
                print(
                    f"[Benchmark][Rank {self.global_rank}] write_video (silent) "
                    f"failed for prompt {prompt_idx}: {e}"
                )
                return

            if audio_waveform is not None:
                try:
                    import torchaudio

                    wav = audio_waveform[0].cpu().float()
                    wav_path = os.path.join(
                        step_dir, f"sample_{prompt_idx}.wav"
                    )
                    torchaudio.save(
                        wav_path, wav, self.benchmark_audio_sample_rate
                    )
                except Exception as e:
                    print(
                        f"[Benchmark][Rank {self.global_rank}] torchaudio.save "
                        f"failed for prompt {prompt_idx}: {e}"
                    )

        # Free decoded tensors
        del video_pixel, audio_waveform
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    trainer = LTXDMDTrainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
