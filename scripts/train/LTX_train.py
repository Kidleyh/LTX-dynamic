# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

# import torch
# torch.autograd.set_detect_anomaly(True)

import argparse
import os
from omegaconf import OmegaConf
import wandb

from ltx_distillation.trainer import (LTXDMDTrainer,
                                      ODELTX23TrainerAccelerate)

from accelerate import (Accelerator, 
                        FullyShardedDataParallelPlugin, 
                        InitProcessGroupKwargs)
from accelerate.logging import get_logger
from accelerate.state import AcceleratorState
from accelerate.utils import ProjectConfiguration, set_seed
from datetime import timedelta

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    parser.add_argument("--no_save", action="store_true")
    parser.add_argument("--no_visualize", action="store_true")
    parser.add_argument("--logdir", type=str, default="wan_experiments_test", help="Path to the directory to save logs")
    parser.add_argument("--report_to", type=str, default="tensorboard", help="Where to report logs")

    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    config.no_save = args.no_save
    config.no_visualize = args.no_visualize

    # get the filename of config_path
    config_name = os.path.basename(args.config_path).split(".")[0]
    config.config_name = config_name
    config.logdir = args.logdir
    config.report_to = args.report_to
    os.makedirs(args.logdir, exist_ok=True)

    pg_kwargs = InitProcessGroupKwargs(timeout=timedelta(minutes=30))  # 90 分钟超时

    if config.trainer == "dmd_ltx23":
        accelerator_project_config = ProjectConfiguration(project_dir=args.logdir, logging_dir=os.path.join(args.logdir, "logs"))
        accelerator = Accelerator(
            gradient_accumulation_steps=1,
            mixed_precision=config.mixed_precision_type,
            log_with=config.report_to,
            project_config=accelerator_project_config,
            kwargs_handlers=[pg_kwargs],
        )
        trainer = LTXDMDTrainer(config, accelerator)
    elif config.trainer == "ode_regression_ltx23":
        trainer = ODELTX23TrainerAccelerate(config)

    os.system(f"cp {args.config_path} {args.logdir}")
    trainer.train()



if __name__ == "__main__":
    main()
