#!/usr/bin/env bash
# 用新（确定性）设置重测 600 步那一轮，与新基线 base300_det 配对比较。
# 输出 outputs/eval/base600_det/   日志 base600_det.log   约 90 分钟
#
# ============ 怎么跑 ============
# 必须在 WSL 里跑。已经在 WSL 终端里的话：
#   bash /mnt/f/vla/eval_base600_det.sh
# 从 PowerShell 起用（整体包在引号里，否则 MSYS 会把 /mnt/... 改写成 C:/Program Files/Git/...）：
#   wsl -d Ubuntu-22.04 -e bash -lc "bash /mnt/f/vla/eval_base600_det.sh"
#
# ============ 两个必须守住的点 ============
# 1. 终端要一直开着。WSL2 在最后一个客户端断开后约 60 秒会销毁整个 VM，nohup 挡不住。
# 2. 进度条会一直刷，别以为卡死了。
#
# ============ 为什么重测 ============
# 老的 76%（outputs/eval/dp100k_na8_600）是在 cudnn.benchmark=True 的不可复现设置下跑的。
# 同一天测得：那个设置下同配置同 seed 跑两遍，ep0 的 sum_reward 能差 3 倍（197 vs 64）。
# 所以 76% 不能再用。这一轮配置与 base300_det 完全一致，只有 episode_length 不同，
# 两边同 seed，可以直接逐局配对比。
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

LOG=/mnt/f/vla/base600_det.log
OUT=/mnt/f/vla/outputs/eval/base600_det

echo "=== episode_length=600 / n_action_steps=8 / 50 局 / seed=1000 ==="
echo "输出目录 $OUT"
echo "日志     $LOG"
echo "预计 90 分钟。下面是实时进度。"
echo

lerobot-eval \
  --policy.path=/mnt/f/vla/outputs/train/diffusion_pusht_30k/checkpoints/100000/pretrained_model \
  --env.type=pusht \
  --env.episode_length=600 \
  --eval.n_episodes=50 \
  --eval.batch_size=1 \
  --policy.n_action_steps=8 \
  --seed=1000 \
  --output_dir="$OUT" \
  2>&1 | tee "$LOG"

echo
echo "=== 评测结束。汇总： ==="
python /mnt/f/vla/eval_stat.py "$OUT"
echo
echo "=== 与新基线逐局配对比较 ==="
python /mnt/f/vla/check_paired_nesting.py \
  /mnt/f/vla/outputs/eval/base300_det \
  /mnt/f/vla/outputs/eval/base600_det