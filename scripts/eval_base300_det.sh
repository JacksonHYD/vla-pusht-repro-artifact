#!/usr/bin/env bash
# 新基线：确定性设置下的 300 步 / 50 局。约 45-60 分钟。
# 输出 outputs/eval/base300_det/   日志 base300_det.log
#
# ============ 怎么跑 ============
# 必须在 WSL 里跑。已经在 WSL 终端里的话：
#   bash /mnt/f/vla/eval_base300_det.sh
# 从 PowerShell 起用（整体包在引号里，否则 MSYS 会把 /mnt/... 改写成 C:/Program Files/Git/...）：
#   wsl -d Ubuntu-22.04 -e bash -lc "bash /mnt/f/vla/eval_base300_det.sh"
#
# ============ 两个必须守住的点 ============
# 1. 终端要一直开着。WSL2 在最后一个客户端断开后约 60 秒会销毁整个 VM，nohup 挡不住。
# 2. 进度条会一直刷，别以为卡死了。
#
# ============ 为什么要重跑 ============
# 2026-10-20 查明：老的设置（cudnn.benchmark=True）下，同一配置同 seed 跑两遍结果不同
# （ep0 sum_reward 197.30 vs 63.91，ep2 成功/失败都翻盘）。已修成确定性设置，
# 修完两遍逐位一致。但这也意味着老的 44% 无法复现 —— 它是三个"同样正确"的实现之一。
# 所以基准要重测，之后所有实验都跟这个新数比。
#
# 配置和老的 dp100k_na8_50 完全一致，只多了一个 --seed=1000（显式写出来）。
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

LOG=/mnt/f/vla/base300_det.log
OUT=/mnt/f/vla/outputs/eval/base300_det

echo "=== 新基线：episode_length=300 / n_action_steps=8 / 50 局 / seed=1000 ==="
echo "输出目录 $OUT"
echo "日志     $LOG"
echo "预计 45-60 分钟（确定性算法比原来的略慢）。下面是实时进度。"
echo

lerobot-eval \
  --policy.path=/mnt/f/vla/outputs/train/diffusion_pusht_30k/checkpoints/100000/pretrained_model \
  --env.type=pusht \
  --env.episode_length=300 \
  --eval.n_episodes=50 \
  --eval.batch_size=1 \
  --policy.n_action_steps=8 \
  --seed=1000 \
  --output_dir="$OUT" \
  2>&1 | tee "$LOG"

echo
echo "=== 评测结束。汇总： ==="
python /mnt/f/vla/eval_stat.py "$OUT"