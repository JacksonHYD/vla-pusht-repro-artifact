#!/usr/bin/env bash
# DP 复现实验：照抄官方 lerobot/diffusion_pusht 的架构。约 13.6 小时。
# 输出 outputs/train/dp_repro/   日志 dp_repro.log
#
# 怎么跑（在 WSL 里）：bash /mnt/f/vla/train_dp_repro.sh
#
# ============ 和上次"照抄官方配方"的区别 ============
# 上次只抄了 batch/steps/horizon，架构还是 v0.6.1 的默认值。这轮把四项默认值也改成官方的：
#   use_group_norm                     True    （默认 False）
#   pretrained_backbone_weights        null    （默认 ImageNet；GroupNorm 强制要求从零）
#   crop_shape                         [84,84] （默认不裁）
#   use_separate_rgb_encoder_per_camera False  （默认 True）
# 再加官方原配方：batch 64 / 200k 步 / horizon 16 / n_action_steps 8 / seed 100000。
#
# ============ 实测（200 步冒烟）============
# 吞吐 4.07 步/秒 = 194 样本/秒 -> 200k 步约 13.6 小时
# 峰值显存 7675 MiB / 8188 —— 余量约 500 MiB，跑之前关掉别的 GPU 程序。
#
# ============ 存档 ============
# save_freq=40000 -> 40k/80k/120k/160k/200k 五个点，可以中途评测决定要不要跑完。
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

LOG=/mnt/f/vla/dp_repro.log
OUT=/mnt/f/vla/outputs/train/dp_repro

echo "=== DP 官方架构复现 / batch 64 / 200k 步 / 约 13.6 小时 ==="
echo "峰值显存 7675 MiB，余量小 —— 别开别的 GPU 程序。"
echo "对照：官方 65.4%；我们上次（默认架构）33%"
echo

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train \
  --dataset.repo_id=lerobot/pusht \
  --policy.type=diffusion \
  --policy.horizon=16 \
  --policy.n_action_steps=8 \
  --policy.use_group_norm=true \
  --policy.pretrained_backbone_weights=null \
  --policy.crop_shape='[84,84]' \
  --policy.use_separate_rgb_encoder_per_camera=false \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=64 \
  --steps=200000 \
  --seed=100000 \
  --save_freq=40000 \
  --output_dir="$OUT" \
  --wandb.enable=false \
  2>&1 | tee "$LOG"

echo
echo "=== 训练结束。最后几行： ==="
tr '\r' '\n' < "$LOG" | grep -a 'step:' | tail -3
echo
echo "=== 下一步：评测（按官方协议，batch 22 / 500 局） ==="
echo "bash /mnt/f/vla/eval_dp_repro.sh"