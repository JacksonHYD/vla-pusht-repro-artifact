#!/usr/bin/env bash
# 去掉 agent_pos 重训 ACT：只让模型看画面。约 2 小时。
# 输出 outputs/train/act_nostate_100k/   日志 act_nostate.log
#
# 怎么跑（在 WSL 里）：bash /mnt/f/vla/train_act_nostate.sh
#
# ============ 为什么做这个 ============
# 两个独立测试都指向同一件事：ACT 的输出几乎完全由 agent_pos 决定，画面里的方块几乎不参与。
#   * 方块在图上平移 53 个单位，输出只动约 2 个单位（斜率 0.03）
#   * 换图测试：t_BA=0.10、t_AB=0.83，签名就是"只听推杆坐标"
# 所以把 agent_pos 拿掉，模型就被迫只能从画面里找方块。
#
# 数据集 pusht_nostate 是 lerobot/pusht 的副本，用
#   lerobot-edit-dataset --operation.type remove_feature --operation.feature_names "['observation.state']"
# 造出来的。
#
# ============ 和老训练的唯一差别 ============
# 老训练用 --config_path 指定了写死的 input_features（含 observation.state）；
# 这里直接从数据集推导特征，所以模型没有状态输入这一路。
# 其余超参（chunk_size=100、lr=1e-5、batch=8、resnet18）都走 ACT 默认值，和老的相同。
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

LOG=/mnt/f/vla/act_nostate.log
OUT=/mnt/f/vla/outputs/train/act_nostate_100k

echo "=== ACT 无 agent_pos：100k 步 ==="
echo "数据 pusht_nostate（只有 observation.image + action）"
echo "输出 $OUT"
echo "预计 约 2 小时"
echo

lerobot-train \
  --dataset.repo_id=pusht_nostate \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --steps=100000 \
  --output_dir="$OUT" \
  --wandb.enable=false \
  2>&1 | tee "$LOG"

echo
echo "=== 训练结束。最后几行： ==="
tr '\r' '\n' < "$LOG" | grep -a 'step:' | tail -3