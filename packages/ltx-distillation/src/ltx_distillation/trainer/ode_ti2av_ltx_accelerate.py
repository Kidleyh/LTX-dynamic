import gc
import logging
from ltx_distillation.utils.dataset import ODERegressionStateDictDataset, cycle, ltx_collate_fn
from ltx_distillation.models.ode_regression_ltx import ODERegressionLTX23
from collections import defaultdict
from ltx_distillation.util import (
    set_seed,
)
import torch.distributed as dist
from omegaconf import OmegaConf
import torch
import wandb
import time
import os

# from ltx_distillation.utils.distributed import barrier, fsdp_wrap, fsdp_state_dict, launch_distributed_job
from accelerate import Accelerator
# from torchvision.io import write_video
from ltx_pipelines.utils.media_io import encode_video as write_video
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from einops import rearrange

class ODELTX23TrainerAccelerate:
    def __init__(self, config):
        self.config = config
        self.step = 0

        # Step 1: Initialize the distributed training environment (rank, seed, dtype, logging etc.)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        self.accelerator = Accelerator(
            mixed_precision='bf16' if config.mixed_precision else 'no',
            # gradient_accumulation_steps=config.accumulation_steps
        )
        self.device = self.accelerator.device
        self.is_main_process = self.accelerator.is_main_process
        self.world_size = self.accelerator.num_processes

        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        # self.device = torch.device("cuda", torch.cuda.current_device())
        # self.is_main_process = global_rank == 0
        self.disable_wandb = config.disable_wandb
        self.disable_tensorboard = getattr(config, 'disable_tensorboard', False)

        # use a random seed for the training
        if config.seed == 0:
            import random
            config.seed = random.randint(0, 10000000)

        set_seed(config.seed + self.accelerator.process_index + 2)

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

        self.output_path = config.logdir

        # Step 2: Initialize the model and optimizer

        assert config.distribution_loss == "ode", "Only ODE loss is supported for ODE training"
        self.model = ODERegressionLTX23(config, device=self.device).to(self.dtype)
        
        if getattr(config, "generator_ckpt", False):
            print(f"Loading pretrained generator from {config.generator_ckpt}")
            # 加载完整状态字典到 CPU
            loaded_state_dict = torch.load(config.generator_ckpt, map_location="cpu")
            generator_state_dict = loaded_state_dict["generator"]
            self.model.generator.load_state_dict(generator_state_dict, strict=True)
            print(f"Model loaded from {config.generator_ckpt}")
            # 解析 step 号
            self.step = int(config.generator_ckpt.split("/")[-2].split("_")[-1])
            print(f"Setting self.step to {self.step}")
        else:
            self.step = 0

        if not config.no_visualize or config.load_raw_video:
            self.model.audio_vae = self.model.audio_vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)
            self.model.video_vae = self.model.video_vae.to(
                device=self.device, dtype=torch.bfloat16 if config.mixed_precision else torch.float32)

        self.generator_optimizer = torch.optim.AdamW(
            [param for param in self.model.generator.parameters()
             if param.requires_grad],
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            weight_decay=config.weight_decay
        )

        # Prepare model, optimizer, and dataloader with Accelerator
        self.model.generator, self.generator_optimizer = self.accelerator.prepare(
            self.model.generator, self.generator_optimizer
        )

        # Step 3: Initialize the dataloader
        ode_data_dir = config.ode_data_dir
        ode_datasets_path = [ode_data_dir]
        dataset = ODERegressionStateDictDataset(ode_datasets_path)
        # sampler = torch.utils.data.distributed.DistributedSampler(
        #     dataset, shuffle=True, drop_last=True)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True, drop_last=True, num_workers=config.dataloader_num_workers, collate_fn=ltx_collate_fn)
        total_batch_size = getattr(config, "total_batch_size", None)
        if total_batch_size is not None:
            assert total_batch_size == config.batch_size * self.world_size, "Gradient accumulation is not supported for ODE training"
        self.dataloader = cycle(dataloader)
        self.dataloader = self.accelerator.prepare(self.dataloader)

        self.max_grad_norm = config.max_grad_norm
        self.previous_time = None
        self.accumulation_steps = config.accumulation_steps


    def save(self):
        print("Start gathering distributed model states...")
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            StateDictType,
            FullStateDictConfig,
        )
        unwrapped_model = self.accelerator.unwrap_model(self.model.generator)
        save_policy = FullStateDictConfig(
            offload_to_cpu=True,  # 将状态字典卸载到 CPU，减少 GPU 内存使用
            rank0_only=True       # 只在 rank 0 进程上收集完整状态
        )
        with FSDP.state_dict_type(unwrapped_model, StateDictType.FULL_STATE_DICT, save_policy):
            generator_state_dict = unwrapped_model.state_dict()
        # unwrapped_model = self.accelerator.unwrap_model(self.model.generator)
        # generator_state_dict = unwrapped_model.state_dict()
        state_dict = {
            "generator": generator_state_dict
        }

        if self.is_main_process:
            os.makedirs(os.path.join(self.output_path,
                        f"checkpoint_model_{self.step:06d}"), exist_ok=True)
            torch.save(state_dict, os.path.join(self.output_path,
                       f"checkpoint_model_{self.step:06d}", "model.pt"))
            print("Model saved to", os.path.join(self.output_path,
                  f"checkpoint_model_{self.step:06d}", "model.pt"))

    def train_one_step(self):
        VISUALIZE = True
        accumulation_steps = getattr(self, "accumulation_steps", 1)

        try:
            batch = next(self.local_dataloader_iterator)
        except StopIteration:
            print("数据迭代完，开始新的循环...")
            self.local_dataloader_iterator = iter(self.dataloader)  # 重新创建迭代器
            batch = next(self.local_dataloader_iterator)

        # Step 3: Train the generator
        with self.accelerator.autocast(): 
            generator_loss, log_dict = self.model.generator_loss(batch)

        unnormalized_loss = log_dict["unnormalized_loss"]
        timestep = log_dict["timestep"]

        torch.cuda.empty_cache()

        # Use Accelerator for gradient accumulation
        # with self.accelerator.accumulate(self.model.generator):
            # generator_loss = generator_loss / accumulation_steps
            # self.accelerator.backward(generator_loss.to(torch.float32))
        generator_loss = generator_loss / accumulation_steps
        generator_loss.backward()
        update_params_flag = False
        if self.step % accumulation_steps == 0:
            generator_grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.generator.parameters(), self.max_grad_norm
            )
            update_params_flag = True
            self.generator_optimizer.step()
            self.generator_optimizer.zero_grad()
        else:
            generator_grad_norm = torch.tensor(0.0, device=self.device)  # dummy value if not stepped

        # Step 4: Visualization
        if VISUALIZE and not self.config.no_visualize and self.is_main_process and self.step % 10 == 0:
            # Visualize the output, and ground truth
            output_video = log_dict["output_video"].to(self.device).to(self.dtype)
            ground_truth_video = log_dict["target_video"].to(self.device).to(self.dtype)
            output_audio = log_dict["output_audio"].to(self.device).to(self.dtype)
            ground_truth_audio = log_dict["target_audio"].to(self.device).to(self.dtype)
            
            output_video_dir = os.path.join(self.config.logdir, "ode_train_videos/")
            os.makedirs(output_video_dir, exist_ok=True)

            output_video_path = os.path.join(output_video_dir, f"output_{self.step}.mp4")
            ground_truth_video_path = os.path.join(output_video_dir, f"ground_truth_{self.step}.mp4")
            
            with torch.no_grad():
                video_num_frames = 169
                width = 768
                height = 512
                frame_rate = 24
                decoded_video = self.model.video_vae.decode_to_visualize(output_video, \
                                                num_frames=video_num_frames, width=width, height=height, frame_rate=frame_rate)
                decoded_audio = self.model.audio_vae.decode_to_visualize(output_audio, \
                                                num_frames=video_num_frames, width=width, height=height, frame_rate=frame_rate)           
                write_video(video=decoded_video, audio=decoded_audio, fps=frame_rate, output_path=output_video_path, video_chunks_number=1)    
                
                decoded_video = self.model.video_vae.decode_to_visualize(ground_truth_video, \
                                                num_frames=video_num_frames, width=width, height=height, frame_rate=frame_rate)
                decoded_audio = self.model.audio_vae.decode_to_visualize(ground_truth_audio, \
                                                num_frames=video_num_frames, width=width, height=height, frame_rate=frame_rate)           
                write_video(video=decoded_video, audio=decoded_audio, fps=frame_rate, output_path=ground_truth_video_path, video_chunks_number=1)    

                # self.model.vae.forward(output_video, output_audio, output_video_path, num_frames=121, width=768, height=512, frame_rate=24)
                # self.model.vae.forward(ground_truth_video, ground_truth_audio, ground_truth_video_path, num_frames=121, width=768, height=512, frame_rate=24)

        # Step 5: Logging
        if self.is_main_process and self.writer is not None:
            self.writer.add_scalar('loss/generator_loss', generator_loss.item() * accumulation_steps, self.step)
            self.writer.add_scalar('gradient/generator_grad_norm', generator_grad_norm.item(), self.step)

            if log_dict and False:
                for key, value in log_dict.items():
                    if isinstance(value, (int, float, torch.Tensor)):
                        if isinstance(value, torch.Tensor):
                            value = value.item()
                        self.writer.add_scalar(f'metrics/{key}', value, self.step)

        if self.step % self.config.gc_interval == 0:
            if dist.get_rank() == 0:
                logging.info("DistGarbageCollector: Running GC.")
            gc.collect()

        return generator_loss.detach().item(), update_params_flag

    def train(self):

        self.model.generator.model.train()

        if self.is_main_process:
            pbar = tqdm(total=self.config.num_train_timestep, desc="Training", 
                    unit="iter", ncols=120, 
                    initial=self.step,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        self.step = self.step + 1
        can_save = True

        self.local_dataloader_iterator = iter(self.dataloader)

        while True:
            loss_generated, update_params_flag = self.train_one_step()
            if (not self.config.no_save) and self.step % self.config.log_iters == 0 and can_save:
                self.save()
                torch.cuda.empty_cache()
                can_save = False

            self.accelerator.wait_for_everyone()
            if self.is_main_process:
                current_time = time.time()
                if self.previous_time is None:
                    self.previous_time = current_time
                else:
                    if self.writer is not None:
                        self.writer.add_scalar("timing/per_iteration_time", current_time - self.previous_time, self.step)
                    self.previous_time = current_time
                # Update progress bar
                if update_params_flag:
                    print(f"loss_generated:{loss_generated:.4f}")
                    pbar.update(1)
                    pbar.set_postfix({"loss":f"{loss_generated:.4f}"})
                    pbar.write(f"loss:{loss_generated:.4f}")            
            if update_params_flag:
                can_save = True
            self.step += 1
