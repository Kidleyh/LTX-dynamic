#!/bin/bash
export PYTHONPATH=$(pwd)

# 设置单POD的GPU数量
export GPUS_PER_NODE=8
export WORLD_SIZE=2
export MASTER_ADDR=$GEMINI_HOST_IP_taskrole1_0
export MASTER_PORT=12345
export RANK=$GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX

# 设置分布式训练DDP所需的环境变量
export MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
export MASTER_PORT=${MASTER_PORT:-'12345'}
export NNODES=${WORLD_SIZE:-'1'}
export NODE_RANK=${RANK:-'0'}
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

export NCCL_IB_HCA="mlx5_0,mlx5_10,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8,mlx5_9"
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=INFO
export NCCL_ALGO=RING

# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# nohup torchrun $DISTRIBUTED_ARGS Wan_train.py --config_path configs/self_forcing_df_ai2v.yaml --logdir wan_experiments/0820_4nodes_bs128_tf_ffmask > log_train_$GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX.log 2>&1 &

# 捕获 SIGINT/SIGTERM
# trap "echo '收到中断信号，退出重启循环'; exit 0" SIGINT SIGTERM

BASE_LOG_DIR="./logs"
TODAY=$(date +'%Y%m%d')
LOG_DIR="$BASE_LOG_DIR/$TODAY"

mkdir -p "$LOG_DIR"
until (
    TS=$(date +'%Y%m%d_%H%M%S')
    LOGFILE="$LOG_DIR/log_train_${GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX}_$TS.log"
    # 启动子脚本为后台进程，并记录其 PID
    # nohup accelerate launch --num_processes $WORLD_SIZE --num_machines $NNODES --machine_rank $GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX --main_process_ip $GEMINI_HOST_IP_taskrole1_0 --main_process_port $MASTER_PORT --mixed_precision bf16 --use_fsdp --fsdp_auto_wrap_policy SIZE_BASED_WRAP --fsdp_min_num_params 100000000 --fsdp_sharding_strategy HYBRID_SHARD --fsdp_state_dict_type SHARDED_STATE_DICT --fsdp_backward_prefetch BACKWARD_PRE --fsdp_cpu_ram_efficient_loading False --fsdp_offload_params true  Wan_train.py --config_path configs/ode_data_generation_ai2v_gz.yaml --logdir wan_experiments/0912_6nodes_bs96_ode_ffmask > log_train_$GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX.log 2>&1 &
    # nohup accelerate launch --num_processes $WORLD_SIZE --num_machines $NNODES --machine_rank $GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX --main_process_ip $GEMINI_HOST_IP_taskrole1_0 --main_process_port $MASTER_PORT --mixed_precision bf16 --dynamo_backend no --use_fsdp --fsdp_auto_wrap_policy SIZE_BASED_WRAP --fsdp_min_num_params 100000000 --fsdp_sharding_strategy FULL_SHARD --fsdp_state_dict_type SHARDED_STATE_DICT --fsdp_backward_prefetch BACKWARD_PRE --fsdp_cpu_ram_efficient_loading true --fsdp_sync_module_states true --fsdp_offload_params false Wan_train.py --config_path configs/self_forcing_dmd_ai2v_causvid_gz.yaml --logdir wan_experiments/0919_6nodes_bs96_dmd_ffmask_16fps_fullx2 > $LOGFILE 2>&1 &
    nohup accelerate launch --num_processes $WORLD_SIZE --num_machines $NNODES --machine_rank $GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX --main_process_ip $GEMINI_HOST_IP_taskrole1_0 --main_process_port $MASTER_PORT --mixed_precision bf16 --dynamo_backend no --use_fsdp --fsdp_auto_wrap_policy SIZE_BASED_WRAP --fsdp_min_num_params 1000000 --fsdp_sharding_strategy FULL_SHARD --fsdp_state_dict_type SHARDED_STATE_DICT --fsdp_backward_prefetch BACKWARD_PRE --fsdp_cpu_ram_efficient_loading true --fsdp_sync_module_states true --fsdp_offload_params false scripts/train/LTX_train.py --config_path /gemini/platform/public/aigc/human_guozz2/code/hys/LTX-2/packages/ltx-distillation/configs/bidirectional_dmd/ltx23_bidirectional_dmd.yaml --logdir ltx_experiments/0415_ltx_node_bs21_bidmd_smalllr > $LOGFILE 2>&1 &
    PID=$!
    echo "[`date +'%F %T'`] 启动训练脚本，PID=$PID"
    # 等待该后台进程结束，$? 即为退出码
    wait $PID
); do
    EXIT_CODE=$?
    echo "[`date +'%F %T'`] 训练脚本意外退出，退出码 $EXIT_CODE，准备重启..." >&2
    sleep 5  # 防止快速重启风暴
done

echo "[`date +'%F %T'`] 训练脚本已正常退出，结束循环。"