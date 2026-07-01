#!/bin/bash
set -Eeuo pipefail

export PYTHONPATH=$(pwd)

# Single-node 7-GPU LoRA-DMD launch. Override CUDA_VISIBLE_DEVICES/GPUS_PER_NODE
# when running on a different local GPU layout.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6}"
export GPUS_PER_NODE=${GPUS_PER_NODE:-7}
export NNODES=1
export NODE_RANK=0
export MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
export MASTER_PORT=${MASTER_PORT:-12345}
export WORLD_SIZE=$((GPUS_PER_NODE * NNODES))

export NCCL_IB_HCA=${NCCL_IB_HCA:-"mlx5_0,mlx5_10,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8,mlx5_9"}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-0}
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ALGO=${NCCL_ALGO:-RING}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export PYTHONWARNINGS="${PYTHONWARNINGS:-ignore:Upcasted low precision parameters:UserWarning,ignore:Using a slow image processor:UserWarning}"

BASE_LOG_DIR=${BASE_LOG_DIR:-./logs}
TODAY=$(date +'%Y%m%d')
LOG_DIR="$BASE_LOG_DIR/$TODAY"
mkdir -p "$LOG_DIR"

TS=$(date +'%Y%m%d_%H%M%S')
LOGFILE="$LOG_DIR/log_train_1node_7gpu_bidirectional_lora_384x576_rank0_${TS}.log"
CONFIG_PATH=${CONFIG_PATH:-packages/ltx-distillation/configs/bidirectional_dmd/ltx23_bidirectional_dmd_lora_lyh_384x576_121f_normalopt_seq1_bs1_1node_7gpu.yaml}
RUN_NAME=${RUN_NAME:-ltx23_bidirectional_dmd_lora_1node_7gpu_384x576_121f_normalopt_seq1_bs1_cpuoffload_8step_log200_$(date +'%m%d_%H%M')}

echo "[$(date +'%F %T')] Start 1-node 7-GPU LTX23 bidirectional LoRA-DMD training"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  NODE_RANK=$NODE_RANK NNODES=$NNODES GPUS_PER_NODE=$GPUS_PER_NODE WORLD_SIZE=$WORLD_SIZE"
echo "  MASTER_ADDR=$MASTER_ADDR MASTER_PORT=$MASTER_PORT"
echo "  CONFIG_PATH=$CONFIG_PATH"
echo "  RUN_NAME=$RUN_NAME"
echo "  LOGFILE=$LOGFILE"

nohup accelerate launch \
  --num_processes "$WORLD_SIZE" \
  --num_machines "$NNODES" \
  --machine_rank "$NODE_RANK" \
  --main_process_ip "$MASTER_ADDR" \
  --main_process_port "$MASTER_PORT" \
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
  --config_path "$CONFIG_PATH" \
  --logdir "ltx_experiments/$RUN_NAME" > "$LOGFILE" 2>&1 &

PID=$!
echo "[$(date +'%F %T')] PID=$PID"
wait $PID
EXIT_CODE=$?
echo "[$(date +'%F %T')] Training finished, exit code: $EXIT_CODE"
exit $EXIT_CODE
