#!/bin/bash
export PYTHONPATH=$(pwd)

export GPUS_PER_NODE=8
export NCCL_IB_HCA="mlx5_0,mlx5_10,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8,mlx5_9"
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_ALGO=RING
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

BASE_LOG_DIR="./logs"
TODAY=$(date +'%Y%m%d')
LOG_DIR="$BASE_LOG_DIR/$TODAY"

mkdir -p "$LOG_DIR"
until (
    TS=$(date +'%Y%m%d_%H%M%S')
    LOGFILE="$LOG_DIR/log_train_0_$TS.log"
    echo "[$(date +'%F %T')] 启动单机正式训练，日志: $LOGFILE"
    nohup accelerate launch \
        --num_processes $GPUS_PER_NODE \
        --num_machines 1 \
        --mixed_precision bf16 \
        --dynamo_backend no \
        --use_fsdp \
        --fsdp_auto_wrap_policy SIZE_BASED_WRAP \
        --fsdp_min_num_params 1000000 \
        --fsdp_sharding_strategy FULL_SHARD \
        --fsdp_state_dict_type SHARDED_STATE_DICT \
        --fsdp_backward_prefetch BACKWARD_PRE \
        --fsdp_cpu_ram_efficient_loading true \
        --fsdp_sync_module_states true \
        --fsdp_offload_params true \
        scripts/train/LTX_train.py \
        --config_path packages/ltx-distillation/configs/causal_dmd/ltx23_causal_dmd.yaml \
        --logdir ltx_experiments/0520_1node_bs40_causaldmd > $LOGFILE 2>&1 &
    PID=$!
    echo "[$(date +'%F %T')] 启动训练脚本，PID=$PID"
    wait $PID
); do
    EXIT_CODE=$?
    echo "[$(date +'%F %T')] 训练脚本意外退出，退出码 $EXIT_CODE，准备重启..." >&2
    sleep 5
done

echo "[$(date +'%F %T')] 训练脚本已正常退出，结束循环。"
