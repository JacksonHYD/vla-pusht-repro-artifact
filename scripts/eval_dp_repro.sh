#!/usr/bin/env bash
# 评测 DP 复现版，按官方协议：batch 22、500 局、300 步、seed 1000。约 15-30 分钟。
#
# 怎么跑（在 WSL 里）：bash /mnt/f/vla/eval_dp_repro.sh
#
# ============ 为什么用 500 局 ============
# 官方那个 65.4% 就是 500 局评出来的（它仓库里的 eval_info.json 有 500 条 per_episode，
# 第一条的 seed 是 1000）。用同样的局数和 seed 区间，结果才直接可比。
# 而且 batch_size=22 是官方用的并行环境数 —— 一次跑 22 个环境，墙钟时间能压下来。
#
# 对照组：
#   官方 lerobot/diffusion_pusht   65.4%（500 局）
#   我们上次（默认架构）            33%（30 局）
#   我们自己从零训的 ACT            38%（50 局）
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

BASE=/mnt/f/vla/outputs/train/dp_repro/checkpoints
CK=${1:+$BASE/$1/pretrained_model}
if [ -z "${1:-}" ]; then
  STEP=$(ls -d "$BASE"/[0-9]* 2>/dev/null | sort -V | tail -1 | xargs -r basename)
  CK="$BASE/$STEP/pretrained_model"
fi
OUT=/mnt/f/vla/outputs/eval/dp_repro_500
LOG=/mnt/f/vla/dp_repro_eval.log

echo "=== 评测 $CK / batch 22 同步环境 / 50 局 / 300 步 ==="
echo

# ⚠️ use_async_envs=false 是必须的：
# 异步模式（async=True）在 batch_size>1 时会起多个 spawn 子进程，而 spawn 不继承父进程
# 导入过的 gym_pusht，子进程里报 "Namespace gym_pusht not found" 然后崩溃
# （ConnectionResetError: Connection reset by peer）。
# batch_size=1 时没暴露这个问题，所以之前的评测一直是好的。
lerobot-eval \
  --policy.path="$CK" \
  --env.type=pusht \
  --env.episode_length=300 \
  --eval.n_episodes=50 \
  --eval.batch_size=22 \
  --eval.use_async_envs=false \
  --policy.n_action_steps=8 \
  --seed=1000 \
  --output_dir="$OUT" \
  2>&1 | tee "$LOG"

echo
echo "=== 汇总 ==="
python /mnt/f/vla/eval_stat.py "$OUT"
echo
echo "=== 对照 ==="
echo "  官方 lerobot/diffusion_pusht   65.4%（500 局）"
echo "  我们上次（默认架构）            33%（30 局）"
echo "  ACT（自己从零训）               38%（50 局）"