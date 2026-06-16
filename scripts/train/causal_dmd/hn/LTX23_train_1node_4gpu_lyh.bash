#!/bin/bash
export PYTHONPATH=$(pwd)

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export GPUS_PER_NODE=4
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

TS=$(date +'%Y%m%d_%H%M%S')
LOGFILE="$LOG_DIR/log_train_1node_4gpu_$TS.log"
echo "[$(date +'%F %T')] Start single-node 4-GPU training, log: $LOGFILE"

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
  --fsdp_offload_params false \
  scripts/train/LTX_train.py \
  --config_path packages/ltx-distillation/configs/causal_dmd/ltx23_causal_dmd_lyh.yaml \
  --logdir ltx_experiments/ltx23_causal_dmd_1node_4gpu_$(date +'%m%d_%H%M') > $LOGFILE 2>&1 &

PID=$!
echo "[$(date +'%F %T')] PID=$PID"
wait $PID
echo "[$(date +'%F %T')] 训练结束，退出码: $?"
