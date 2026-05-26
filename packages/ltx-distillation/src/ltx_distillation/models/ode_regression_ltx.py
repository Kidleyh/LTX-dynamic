import torch.nn.functional as F
from typing import Tuple, Dict
import torch

from ltx_distillation.models.base import BaseModel
from ltx_distillation.models.ltx_wrapper import LTX2DiffusionWrapper, create_ltx2_wrapper, create_causal_ltx2_wrapper
from ltx_distillation.models.text_encoder_wrapper import GemmaTextEncoderWrapper, create_text_encoder_wrapper
from ltx_distillation.models.vae_wrapper import VideoVAEWrapper, AudioVAEWrapper, create_vae_wrappers
# from ltx_distillation.loss import get_denoising_loss
from torch.distributions import Categorical

from omegaconf import OmegaConf
import math

from ltx_core.model.transformer import Modality
from ltx_core.loader.registry import StateDictRegistry
import torch.distributed as dist

class ODERegressionLTX23(BaseModel):
    def __init__(self, args, device):
        """
        Initialize the ODERegression module.
        This class is self-contained and compute generator losses
        in the forward pass given precomputed ode solution pairs.
        This class supports the ode regression loss for both causal and bidirectional models.
        See Sec 4.3 of CausVid https://arxiv.org/abs/2412.07772 for details
        """
        self.args = args
        self.device = device
        self.dtype = torch.bfloat16 if args.mixed_precision else torch.float32

        super().__init__(args, device)

        # self._initialize_models()

        # Step 1: Initialize all models

        if getattr(args, "generator_ckpt", False):
            print(f"Loading pretrained generator from {args.generator_ckpt}")
            state_dict = torch.load(args.generator_ckpt, map_location="cpu")[
                'generator']
            self.generator.load_state_dict(
                state_dict, strict=True
            )

        self.num_frame_per_block = getattr(args, "num_frame_per_block", 1)

        if self.num_frame_per_block > 1:
            self.generator.model.num_frame_per_block = self.num_frame_per_block

        self.independent_first_frame = getattr(args, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        self.timestep_shift = getattr(args, "timestep_shift", 1.0)

        

    # def _initiali_zemodels(self, args, device):
    #     self.generator = DiffusionLTXWrapper(device=device, **getattr(args, "model_kwargs", {}), is_causal=True, max_token_size=getattr(args, "max_token_size", -1))
    #     self.generator.model.requires_grad_(True)
    #     # self.generator.model.to(dtype=torch.bfloat16) # for memory saving's sake
    #     self.generator.model.train()

    #     ##NOTE: debug only: unfreeze the first transformer block and self-attn layers
    #     for param in self.generator.model.parameters():
    #         param.requires_grad = True
    #     # for i in range(30):
    #     #     for param in self.generator.model.transformer_blocks[i].parameters():
    #     #         param.requires_grad = True

    #     total_params = sum(p.numel() for p in self.generator.model.parameters())
    #     trainable_params = sum(p.numel() for p in self.generator.model.parameters() if p.requires_grad)

    #     total_params_b = total_params / 1e9
    #     trainable_params_b = trainable_params / 1e9
    #     trainable_ratio = trainable_params / total_params

    #     print(f"\n✅ Total parameters in the model: {total_params_b:.3f}B")
    #     print(f"✅ Trainable parameters: {trainable_params_b:.3f}B")
    #     print(f"✅ Percentage trainable: {trainable_ratio:.2%}\n")


    #     # self.text_encoder = WanTextEncoder()
    #     # self.text_encoder.requires_grad_(False)

    #     # add LTX VAE wrapper
    #     self.vae = LTXVAEWrapper(device=device)
    #     self.vae.requires_grad_(False)

    def _initialize_models(self, args, device):
        """
        Initialize all models from checkpoints.

        This method should be called BEFORE FSDP wrapping in distributed training.
        Models must exist before they can be wrapped with FSDP.
        """
        # args = self.args

        def _init_log(message: str) -> None:
            if not dist.is_initialized() or dist.get_rank() == 0:
                print(f"[DMDInit] {message}", flush=True)

        # Get video dimensions from config
        # video_height = getattr(args, "video_height", 512) # [NOTE] deprecated
        # video_width = getattr(args, "video_width", 768) # [NOTE] deprecated

        # Create diffusion wrappers per model (CausVid-style hybrid setup):
        # generator can be causal while real/fake remain bidirectional.
        if isinstance(device, int):
            target_device = f"cuda:{device}"
        else:
            target_device = str(device)

        def _load_checkpoint_state_dict(checkpoint_path: str) -> dict:
            if checkpoint_path in checkpoint_state_cache:
                return checkpoint_state_cache[checkpoint_path]
            if checkpoint_path.endswith(".safetensors"):
                from safetensors.torch import load_file
                loaded = load_file(checkpoint_path)
                checkpoint_state_cache[checkpoint_path] = loaded
                return loaded

            loaded = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(loaded, dict) and "generator" in loaded:
                loaded = loaded["generator"]
            elif isinstance(loaded, dict) and "model" in loaded:
                loaded = loaded["model"]
            elif isinstance(loaded, dict) and "state_dict" in loaded:
                loaded = loaded["state_dict"]
            checkpoint_state_cache[checkpoint_path] = loaded
            return loaded

        def _remap_state_dict_keys(state_dict: dict) -> dict:
            if not state_dict:
                return state_dict

            non_transformer_prefixes = (
                "vae.", "audio_vae.", "vocoder.",
                "model.vae.", "model.audio_vae.", "model.vocoder.",
            )
            remapped_non_transformer_prefixes = (
                "model.audio_embeddings_connector.",
                "model.video_embeddings_connector.",
            )

            sample_keys = list(state_dict.keys())[:20]
            has_diffusion_model = any(k.startswith("model.diffusion_model.") for k in sample_keys)
            if not has_diffusion_model:
                has_diffusion_model = any(k.startswith("model.diffusion_model.") for k in state_dict)

            if has_diffusion_model:
                remapped = {}
                for k, v in state_dict.items():
                    if not k.startswith("model.diffusion_model."):
                        continue
                    new_key = "model." + k[len("model.diffusion_model."):]
                    if any(new_key.startswith(p) for p in remapped_non_transformer_prefixes):
                        continue
                    remapped[new_key] = v
                return remapped

            first_key = next(iter(state_dict))
            if first_key.startswith("model.velocity_model."):
                return {
                    "model." + k[len("model.velocity_model."):]: v
                    for k, v in state_dict.items()
                    if k.startswith("model.velocity_model.")
                }
            if first_key.startswith("model."):
                return {
                    k: v for k, v in state_dict.items()
                    if not any(k.startswith(p) for p in non_transformer_prefixes)
                }
            return {
                "model." + k: v
                for k, v in state_dict.items()
                if not any(k.startswith(p) for p in non_transformer_prefixes)
            }

        def _build_wrapper(use_causal: bool):

            if use_causal:
                return create_causal_ltx2_wrapper(
                    checkpoint_path=args.checkpoint_path,
                    gemma_path=args.gemma_path,
                    device=torch.device("cpu"),
                    dtype=self.dtype,
                    # video_height=None,
                    # video_width=None,
                    use_flex_attention=args.use_flex_attention,
                    registry=shared_registry,
                )

            _init_log("build bidirectional wrapper start")
            return create_ltx2_wrapper(
                checkpoint_path=args.checkpoint_path,
                gemma_path=args.gemma_path,
                device=torch.device("cpu"),
                dtype=self.dtype,
                # video_height=video_height,
                # video_width=video_width,
                registry=shared_registry,
            )

        checkpoint_state_cache: Dict[str, dict] = {}
        shared_registry = StateDictRegistry()
        _init_log("generator wrapper init start")
        self.generator = _build_wrapper(True)
        _init_log("generator wrapper init done")

        _init_log("text encoder init start")
        self.text_encoder = create_text_encoder_wrapper(
            checkpoint_path=args.checkpoint_path,
            gemma_path=args.gemma_path,
            device=torch.device("cpu"),
            dtype=self.dtype,
            load_in_8bit=False,
            registry=shared_registry,
        )
        _init_log("text encoder init done")

        _init_log("vae init start")
        self.video_vae, self.audio_vae = create_vae_wrappers(
            checkpoint_path=args.checkpoint_path,
            device=device,
            dtype=self.dtype,
            registry=shared_registry,
        )
        _init_log("vae init done")

        # Set gradients

        # enable all grads
        self.generator.set_module_grad(args.generator_grad)

        # [DEBUG] enable part of grads
        # self.generator.set_module_grad({'model': False})
        # # enable 2 layers' grad
        # for i in range(5):
        #     for param in self.generator.model.velocity_model.transformer_blocks[i].parameters():
        #         param.requires_grad = True
        

        self.text_encoder.requires_grad_(False)
        self.video_vae.requires_grad_(False)
        self.audio_vae.requires_grad_(False)

        # Enable gradient checkpointing if requested
        if args.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Checkpoint loading with priority:
        #   resume_checkpoint > generator_ckpt > stage1_ckpt_path
        stage1_ckpt = getattr(args, "stage1_ckpt_path", None)
        stage1_strict = getattr(args, "stage1_ckpt_strict", False)
        generator_ckpt = getattr(args, "generator_ckpt", None)
        generator_ckpt_strict = getattr(args, "generator_ckpt_strict", False)

        if generator_ckpt:
            print(f"Loading pretrained generator from {generator_ckpt}")
            ckpt = torch.load(generator_ckpt, map_location="cpu")
            gen_sd = ckpt.get("generator", ckpt)
            if self.generator_use_causal_wrapper:
                gen_sd = _remap_state_dict_keys(gen_sd)
            missing_g, unexpected_g = self.generator.load_state_dict(gen_sd, strict=generator_ckpt_strict)
            real_missing_g = [k for k in missing_g if "mask_builder" not in k]
            if real_missing_g:
                print(f"  [generator] missing keys ({len(real_missing_g)}): {real_missing_g[:10]}...")
            if unexpected_g:
                print(f"  [generator] unexpected keys ({len(unexpected_g)}): {unexpected_g[:10]}...")

            sink_key = None
            for k in gen_sd:
                if "audio_sink_tokens" in k:
                    sink_key = k
                    break
            if sink_key is not None:
                for pname, param in self.generator.named_parameters():
                    if "audio_sink_tokens" in pname:
                        assert param.shape == gen_sd[sink_key].shape, (
                            f"[Stage3] Sink token shape mismatch in generator: "
                            f"model={param.shape} vs ckpt={gen_sd[sink_key].shape}"
                        )
                        break
            print("[Stage3] Generator checkpoint load complete")

        elif stage1_ckpt:
            print(f"[Stage2] Loading Stage 1 checkpoint from {stage1_ckpt}")
            ckpt = torch.load(stage1_ckpt, map_location="cpu")

            gen_sd = ckpt.get("generator", ckpt)
            # Stage 3 configs may point stage1_ckpt_path at either:
            # 1. a causal/ODE checkpoint already keyed as model.*
            # 2. a bidirectional DMD checkpoint keyed as model.velocity_model.*
            # The causal generator expects model.* keys, so remap before load.
            if self.generator_use_causal_wrapper:
                gen_sd = _remap_state_dict_keys(gen_sd)
            missing_g, unexpected_g = self.generator.load_state_dict(gen_sd, strict=stage1_strict)
            real_missing_g = [k for k in missing_g if "mask_builder" not in k]
            if real_missing_g:
                print(f"  [generator] missing keys ({len(real_missing_g)}): {real_missing_g[:10]}...")
            if unexpected_g:
                print(f"  [generator] unexpected keys ({len(unexpected_g)}): {unexpected_g[:10]}...")

            # CausVid-style hybrid setup: only load Stage1 ckpt into fake_score
            # when fake_score itself is causal.
            if self.fake_score_use_causal_wrapper:
                missing_f, unexpected_f = self.fake_score.load_state_dict(gen_sd, strict=stage1_strict)
                real_missing_f = [k for k in missing_f if "mask_builder" not in k]
                if real_missing_f:
                    print(f"  [fake_score] missing keys ({len(real_missing_f)}): {real_missing_f[:10]}...")
                if unexpected_f:
                    print(f"  [fake_score] unexpected keys ({len(unexpected_f)}): {unexpected_f[:10]}...")
            else:
                print("[Stage2] fake_score is bidirectional, skip Stage1 causal ckpt load for fake_score")

            # Validate sink token shape consistency
            sink_key = None
            for k in gen_sd:
                if "audio_sink_tokens" in k:
                    sink_key = k
                    break
            if sink_key is not None:
                models_to_check = [("generator", self.generator)]
                if self.fake_score_use_causal_wrapper:
                    models_to_check.append(("fake_score", self.fake_score))
                for name, model in models_to_check:
                    for pname, param in model.named_parameters():
                        if "audio_sink_tokens" in pname:
                            assert param.shape == gen_sd[sink_key].shape, (
                                f"[Stage2] Sink token shape mismatch in {name}: "
                                f"model={param.shape} vs ckpt={gen_sd[sink_key].shape}"
                            )
                            break
            print("[Stage2] Stage1 checkpoint load complete")    

    def _process_timestep(self, timestep, generator_task='causal_video'):
        """
        Pre-process the randomly generated timestep based on the generator's task type.
        Input:
            - timestep: [batch_size, num_frame] tensor containing the randomly generated timestep.

        Output Behavior:
            - image: check that the second dimension (num_frame) is 1.
            - bidirectional_video: broadcast the timestep to be the same for all frames.
            - causal_video: broadcast the timestep to be the same for all frames **in a block**.
        """
        if generator_task == "image":
            assert timestep.shape[1] == 1
            return timestep
        elif generator_task == "bidirectional_video":
            for index in range(timestep.shape[0]):
                timestep[index] = timestep[index, 0]
            return timestep
        elif generator_task == "causal_video":
            # make the noise level the same within every motion block
            timestep = timestep.reshape(
                timestep.shape[0], -1, self.num_frame_per_block)
            timestep[:, :, 1:] = timestep[:, :, 0:1]
            timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep
        else:
            raise NotImplementedError()

    @torch.no_grad()
    def _prepare_generator_input(self, audio_video_input_data: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories,
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - audio_video_input_data: a dictionary containing the audio and video ODE sampling trajectories.
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep [batch_size].
        """
        batch_size, num_denoising_steps, seq_len, dim = audio_video_input_data["video_ode_latents"].shape
        assert num_denoising_steps - 1 == len(self.denoising_step_list), "The number of denoising steps in the input data does not match the expected number."
        # Step 1: Randomly choose a timestep for each frame
        # index = torch.randint(0, len(self.denoising_step_list), [
        #     batch_size, num_frames], device=self.device, dtype=torch.long)
        alpha = 0.6
        idx = torch.arange(len(self.denoising_step_list), dtype=torch.float32)
        probs = torch.exp(-alpha * idx)    # exp(-alpha * k)
        probs = probs / probs.sum()        # 归一化为概率
        # step = 2时: probs == [0.69, 0.31] step = 4时: probs == [0.5741, 0.2579, 0.1159, 0.0521]
        index = Categorical(probs).sample((batch_size,)).to(torch.long).to(self.device)
        sigma = self.denoising_step_list[index].to(self.device)
        
        video_noisy_input = Modality(
            latent=audio_video_input_data["video_ode_latents"][torch.arange(batch_size), index],
            sigma=sigma,
            timesteps=audio_video_input_data["video_denoise_mask"][torch.arange(batch_size), index] * sigma,
            positions=audio_video_input_data["video_positions"][torch.arange(batch_size), index],
            context=audio_video_input_data["video_positive_context"][:, 0],
            enabled=True,
            context_mask=None,
            attention_mask=None,
        )
        
        audio_noisy_input = Modality(
            latent=audio_video_input_data["audio_ode_latents"][torch.arange(batch_size), index],
            sigma=sigma,
            timesteps=audio_video_input_data["audio_denoise_mask"][torch.arange(batch_size), index] * sigma,
            positions=audio_video_input_data["audio_positions"][torch.arange(batch_size), index],
            context=audio_video_input_data["audio_positive_context"][:, 0],
            enabled=True,
            context_mask=None,
            attention_mask=None,
        )
        
        return video_noisy_input, audio_noisy_input

    def generator_loss(self, data_batch: dict) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noisy latents and compute the ODE regression loss.
        Input:
            - ode_latent: a tensor containing the ODE latents [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
            They are ordered from most noisy to clean latents.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - loss: a scalar tensor representing the generator loss.
            - log_dict: a dictionary containing additional information for loss timestep breakdown.
        """
        # Step 1: Run generator on noisy latents
        for key in data_batch:
            data_batch[key] = data_batch[key].to(device=self.device, dtype=self.dtype)

        video_target_latent = data_batch["video_ode_latents"][:, -1]
        audio_target_latent = data_batch["audio_ode_latents"][:, -1]
        video_noisy_input, audio_noisy_input = self._prepare_generator_input(
            audio_video_input_data=data_batch)

        pred_x0_v, pred_x0_a = self.generator(
            video=video_noisy_input, audio=audio_noisy_input, perturbations=None
        )
        video_denoise_mask = data_batch["video_denoise_mask"][:, 0]
        audio_denoise_mask = data_batch["audio_denoise_mask"][:, 0]
        video_clean_latent = data_batch["video_clean_latents"][:, 0]
        audio_clean_latent = data_batch["audio_clean_latents"][:, 0]
        pred_x0_v = (pred_x0_v * video_denoise_mask + video_clean_latent * (1 - video_denoise_mask))
        pred_x0_a = (pred_x0_a * audio_denoise_mask + audio_clean_latent * (1 - audio_denoise_mask))
        
        # Step 2: Compute the regression loss
        loss_v = F.mse_loss(
            pred_x0_v, video_target_latent, reduction="mean")
        loss_a = F.mse_loss(
            pred_x0_a, audio_target_latent, reduction="mean")
        loss = loss_v + loss_a

        unnormalized_loss_v = F.mse_loss(pred_x0_v, video_target_latent, reduction='none').mean(dim=[1, 2]).detach()
        unnormalized_loss_a = F.mse_loss(pred_x0_a, audio_target_latent, reduction='none').mean(dim=[1, 2]).detach()

        log_dict = {
            "unnormalized_loss": unnormalized_loss_v + unnormalized_loss_a,
            "timestep": video_noisy_input.sigma.float().detach(),
            "input_video": video_noisy_input.latent.detach(),
            "output_video": pred_x0_v.detach(),
            "target_video": video_target_latent.detach(),
            "input_audio": audio_noisy_input.latent.detach(),
            "output_audio": pred_x0_a.detach(),
            "target_audio": audio_target_latent.detach(),
        }

        return loss, log_dict
