#!/bin/bash
# 单机8卡调试脚本 — 关闭 fsdp_offload_params，低分辨率配置，用于快速验证训练逻辑
export PYTHONPATH=$(pwd)

export GPUS_PER_NODE=${GPUS_PER_NODE:-8}
export NCCL_IB_HCA="mlx5_0,mlx5_10,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8,mlx5_9"
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ALGO=RING
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:Upcasted low precision parameters:UserWarning,ignore:Using a slow image processor:UserWarning}"

BASE_LOG_DIR="./logs"
TODAY=$(date +'%Y%m%d')
LOG_DIR="$BASE_LOG_DIR/$TODAY"

mkdir -p "$LOG_DIR"

TS=$(date +'%Y%m%d_%H%M%S')
LOGFILE="$LOG_DIR/log_debug_$TS.log"

echo "[$(date +'%F %T')] 启动调试训练，PID将写入日志: $LOGFILE"

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
    --config_path packages/ltx-distillation/configs/causal_dmd/ltx23_causal_dmd_debug.yaml \
    --logdir ltx_experiments/debug_$(date +'%m%d_%H%M') > "$LOGFILE" 2>&1 &

PID=$!
echo "[$(date +'%F %T')] 启动训练脚本，PID=$PID，日志: $LOGFILE"
wait $PID
echo "[$(date +'%F %T')] 调试训练结束，退出码: $?"
