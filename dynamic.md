# `dynamic` 分支说明

本分支在 `init commit` 基础上持续开发，聚焦 **Causal DMD 多段流式训练**。

---

## 主要特性

### 多段 KV Cache 持久化训练（Method-B）
- 训练时跨 segment 复用 KV Cache，每段生成结果的最后一帧作为下一段的 conditioning frame，实现真实的自回归流式训练。
- `seq_steps_per_update` 控制每次参数更新前推进的 segment 数（当前 debug config 设为 2）。
- 两段之间 RoPE 位置连续：seg1 的 `segment_video_offset` 对应 seg0 的帧数，position tensor 在全局范围 tile 展开。
- Teacher（real_score）在 seg1+ 额外施加 TI2V RoPE time shift（第 0 帧 time=0，其余 time+=10s），对齐蒸馏目标分布。

### RELIC 逐 Block Replayed Backward
- generator 每段调用 `generator_loss_segment`，内部先 `no_grad` 完整 rollout 并保存每个 block 的 replay state，再逐 block 重新 forward + 立即 backward，峰值显存仅需一个 block 的计算图。
- 两段各自 backward，梯度在 optimizer step 前自然累积；`loss_scale = 1 / (accumulation_steps × seq_steps_per_update)` 保证等效全序列均值。

### 多状态 Prompt（multi_state_prompts）
- 从 JSONL 文件按视频 `file_path` 索引每段对应的描述文本，seg0 用原始 batch prompt，seg1+ 用对应状态的 prompt，支持描述内容随时间真实变化的视频。
- 训练数据已过滤为仅含多状态 prompt 的样本。

### Debug 单机配置
- `packages/ltx-distillation/configs/causal_dmd/ltx23_causal_dmd_debug.yaml`：256×384、49帧/段（2秒）、单机8卡。
- 启动脚本：`scripts/train/causal_dmd/hn/LTX23_train_1node_debug.bash`。
- 使用 `bitsandbytes` 8-bit AdamW（`use_8bit_optimizer: true`），双 optimizer 可省约 50GB/卡显存。
- Visualization 在每次参数更新后解码所有 segment 并拼接为完整多段视频写入 `<logdir>/sample_video/`。

---

## 多机多卡配置

正式训练使用 5 节点 × 8 卡 = 40 卡 H100，启动脚本：`scripts/train/causal_dmd/hn/LTX23_train_5nodes.bash`。

**与单机 debug 配置的主要差异：**

| 参数 | 单机 debug | 5 节点正式 |
|------|-----------|-----------|
| 节点数 | 1 | 5 |
| 总卡数 | 8 | 40 |
| 分辨率 | 256×384 | 512×768 |
| 每段帧数 | 49（7 latent frames） | 169（22 latent frames）|
| 每段时长 | ~2s | ~7s |
| `total_batch_size` | 8 | 64（`accumulation_steps=2`）|
| `self_forcing_max_generated_blocks` | 2 | 5 |
| `log_iters`（checkpoint 频率） | 100 | 500 |
| config | `ltx23_causal_dmd_debug.yaml` | `ltx23_causal_dmd.yaml` |

**分布式启动方式：**

多机通过 Gemini 平台环境变量自动感知拓扑（`GEMINI_HOST_IP_taskrole1_0` 作为 master addr，`GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX` 作为 node rank），使用 `accelerate launch` + FSDP FULL_SHARD，各节点独立执行同一脚本。脚本内置 `until` 自动重启循环，训练意外中断后 5s 自动重试。

**NCCL 配置：**

```bash
NCCL_IB_HCA="mlx5_0,mlx5_10,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_8,mlx5_9"
NCCL_SOCKET_IFNAME=eth0
NCCL_ALGO=RING
```

---

## 提交记录（自 init commit 起）

| Commit | 说明 |
|--------|------|
| `a272f7a` | debug config: seq_steps_per_update=2、8bit optimizer、单机调试脚本 |
| `66605ed` | seg1+ teacher 施加 TI2V RoPE time shift；修复 KV cache PE pinning |
| `e282121` | 跨 segment 同步 exit_step；训练数据过滤为 multi_state_prompts 样本 |
| `66861cc` | 修复 visualization 触发时机与参数更新节奏对齐 |
| `cf7fd3d` | 修复多段训练中跨 segment batch 分辨率不一致问题（pin seg0 batch） |
| `ee89ea5` | 修复 generator 日志作用域：用 did_update_generator flag 避免 stale 引用 |
| `040d749` / `9695976` | visualization 改为每段 decode 并拼接，输出完整多段 mp4 |
| `a87e5fb` | 从 JSONL 加载 per-sample multi_state_prompts |
| `1d01431` | 修复 KV cache 持久化跨段训练的各类 bug |
| `2bbb5ee` | 实现 Method-B：KV Cache 持久化跨段多段训练 |
| `e251cf3` | 初始 causal DMD 训练支持 |
