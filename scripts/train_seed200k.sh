#!/usr/bin/env bash
# 2x2 的第二个训练种子：只补关键的两格（baseline 与 crop_only）。
#
# ============ 为什么是这两格 ============
# 整个 2x2 目前只有 seed=100000 一个训练种子。核心主张是"crop 是主因子（+22pp）」，
# 而它由 baseline(33.0%) 与 crop_only(55.0%) 这一对支撑。这两格换种子能复现，主张才站得住。
# （GN 单独那一格已证明无效，不值得再花 7 小时。）
#
# ============ 预算 ============
# 每格 120k 步 / batch 64 / 约 7 小时 20 分，两格约 15 小时。
# 与原来的四格逐字一致，只有 --seed 不同（200000），所以结果可直接对比。
#
# ============ 开跑前必须做 ============
#   1. 关掉 Wallpaper Engine 和微信，nvidia-smi 的"可用"要到 7500 MiB 以上
#      （dp_repro 峰值是 7675 MiB，现在只有 6020，不关必 OOM）
#   2. 终端别关。WSL2 在最后一个客户端断开约 60 秒后销毁整个 VM，nohup 挡不住
#   3. 笔记本别合盖 / 别睡眠
#
# ============ 怎么跑 ============
#   bash /mnt/f/vla/train_seed200k.sh
#
# 断了可以续：--config_path=<ckpt>/pretrained_model/train_config.json --resume=true
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

COMMON=(
  --dataset.repo_id=lerobot/pusht
  --policy.type=diffusion
  --policy.horizon=16
  --policy.n_action_steps=8
  --policy.device=cuda
  --policy.push_to_hub=false
  --batch_size=64
  --steps=120000
  --seed=200000
  --save_freq=60000
  --wandb.enable=false
)

run_arm () {
  local NAME="$1"; shift
  local OUT="/mnt/f/vla/outputs/train/${NAME}"
  local LOG="/mnt/f/vla/${NAME}.log"
  echo
  echo "=================================================================="
  echo "  ${NAME}"
  echo "  输出 ${OUT}"
  echo "  日志 ${LOG}"
  echo "=================================================================="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train \
    "${COMMON[@]}" "$@" --output_dir="$OUT" 2>&1 | tee "$LOG"
}

# 全默认（GN关 + crop关 + ImageNet 骨干）
run_arm dp_ab_baseline_s200k

# 只开 crop（GN关 + crop开 + ImageNet 骨干）
run_arm dp_ab_crop_only_s200k --policy.crop_shape='[84,84]'

echo
echo "=== 两格训练完成。下一步评测（每格 200 局，约 20 分钟）： ==="
cat <<'HINT'
for tag in baseline crop_only; do
  lerobot-eval --policy.path=/mnt/f/vla/outputs/train/dp_ab_${tag}_s200k/checkpoints/120000/pretrained_model \
    --env.type=pusht --env.episode_length=300 --eval.n_episodes=200 \
    --eval.batch_size=22 --eval.use_async_envs=false \
    --policy.n_action_steps=8 --seed=1000 \
    --output_dir=/mnt/f/vla/outputs/eval/dp_ab_${tag}_s200k_200
done
HINT