#!/usr/bin/env bash
# ACT 续训：从 30k 续到 100k。约 2 小时。
# 输出 outputs/train/act_pusht_100k/   日志 act_100k.log
#
# ============ 怎么跑 ============
# 必须在 WSL 里跑（PowerShell 里的 bash 是 Git Bash，没有 conda.sh）：
#   bash /mnt/f/vla/train_act_100k.sh
# 或从 PowerShell（整体包在引号里，否则 MSYS 会把 /mnt/... 改写成 C:/Program Files/Git/...）：
#   wsl -d Ubuntu-22.04 -e bash -lc "bash /mnt/f/vla/train_act_100k.sh"
#
# ============ 两个必须守住的点 ============
# 1. 终端要一直开着。WSL2 在最后一个客户端断开后约 60 秒会销毁整个 VM，nohup 挡不住。
# 2. 进度条会一直刷，别以为卡死了。约 2 小时。
#
# ============ 为什么要续训 ============
# 30k 步的 ACT 在 pusht 上只有 6%（mine_na20）。日志显示 loss 到 30k 还在稳定下降
# （10k 时 0.336 -> 30k 时 0.189），没有平台期 —— 判断是训练不够，不是别的问题。
# 30k 只花了 38 分钟，训到 100k 约 2 小时。
#
# ============ 跑起来先看这一行 ============
# 日志里第一条 step 的 loss：
#   * 0.19 左右  -> 权重加载成功，是在 30k 的基础上继续
#   * 6.x 左右   -> 从头训了，续训没生效，立刻停掉重来
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

LOG=/mnt/f/vla/act_100k.log
OUT=/mnt/f/vla/outputs/train/act_pusht_100k
CKPT=/mnt/f/vla/outputs/train/act_pusht_30k/checkpoints/030000

echo "=== ACT 续训：30k -> 100k ==="
echo "从     $CKPT"
echo "输出   $OUT"
echo "日志   $LOG"
echo "预计   约 2 小时（30k 步用了 38 分钟）"
echo

lerobot-train \
  --config_path="$CKPT/pretrained_model/train_config.json" \
  --resume=true \
  --output_dir="$OUT" \
  --steps=100000 \
  --wandb.enable=false \
  2>&1 | tee "$LOG"

echo
echo "=== 训练结束。最后几行 loss： ==="
tr '\r' '\n' < "$LOG" | grep -a 'step:' | tail -3
echo
echo "=== 下一步：评测 ==="
echo "outputs/train/act_pusht_100k/checkpoints/100000/pretrained_model"