#!/usr/bin/env bash
# 2x2 析因消融：把 33% -> 64% 的 31 个百分点拆成两个主效应。
#
# ============ 设计 ============
# 两个因子（第三、四项已验证无效，见下）：
#   A = use_group_norm=true + pretrained_backbone_weights=null   （这两项代码强制绑定）
#   B = crop_shape=[84,84]
#
# 四个格子，全部对齐 seed=100000 / 120k 步 / batch 64：
#   (A off, B off)  <- 本脚本 arm1
#   (A on,  B off)  <- 本脚本 arm2
#   (A off, B on )  <- 本脚本 arm3
#   (A on,  B on )  <- 已有：outputs/eval/dp_repro_curve/ck120000 = 62%
#
# ============ 为什么三项要一致 ============
# 1. 训练种子：dp_official(33%) 用的是 seed=1000，dp_repro(64%) 用的是 seed=100000。
#    骨干从零初始化时，种子直接决定初始权重，所以必须在四个格子里统一。
# 2. 步数：dp_repro_curve 显示 120k=62%、200k=64%，差 2pp 在噪声内。
#    四个格子都用 120k，省 40% 时间且可比。
# 3. use_separate_rgb_encoder_per_camera 保持默认(true)不动：
#    modeling_diffusion.py:197-206 显示单相机时两个分支结构、参数量、RNG 消耗都相同，
#    已验证等价。所以它在四格里都取默认值，不构成变量。
#
# ============ 怎么跑 ============
# 在 WSL 里：bash /mnt/f/vla/train_ablation_2x2.sh
# 从 PowerShell：wsl -d Ubuntu-22.04 -e bash -lc "bash /mnt/f/vla/train_ablation_2x2.sh"
# 约 25 小时（三次 × 约 8.2 小时）。终端要一直开着，WSL2 断连会销毁 VM。
#
# ============ 跑之前先写预测 ============
# 在 dp_ablation_预测.md 里写：你预测 A 贡献多少 pp、B 贡献多少 pp、有没有交互。
# 没有预测的记录，跑完只是在读数字。
#
# ============ 存档 ============
# save_freq=60000 -> 每格 60k/120k 两个点，共约 6 GB。三格 18 GB。
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
  --seed=100000
  --save_freq=60000
  --wandb.enable=false
)

run_arm () {
  local NAME="$1"; shift
  local OUT="/mnt/f/vla/outputs/train/${NAME}"
  local LOG="/mnt/f/vla/${NAME}.log"
  echo
  echo "=================================================================="
  echo "  ${NAME}   ($*)"
  echo "  输出 ${OUT}"
  echo "=================================================================="
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True lerobot-train \
    "${COMMON[@]}" "$@" --output_dir="$OUT" 2>&1 | tee "$LOG"
}

# arm1: A off, B off —— 全默认
run_arm dp_ab_baseline

# arm2: A on, B off —— 只开 GroupNorm + 从零骨干
run_arm dp_ab_gn_only \
  --policy.use_group_norm=true \
  --policy.pretrained_backbone_weights=null

# arm3: A off, B on —— 只开 crop
run_arm dp_ab_crop_only \
  --policy.crop_shape='[84,84]'

echo
echo "=== 全部训练完成。下一步每格评测 50 局（约 5 分钟/格）： ==="
echo "  bash /mnt/f/vla/eval_ablation_2x2.sh"