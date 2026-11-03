#!/usr/bin/env bash
# 评三个消融 arm 的 60k checkpoint，看"交互是从训练早期就有，还是后期才出现"。
#
# 协议与 120k 那次完全一致：50 局 / batch 22 同步 / episode_length 300 / na 8 / seed=1000。
# 这样 60k 和 120k 的结果才能直接比。
#
# 怎么跑（WSL 里）：bash /mnt/f/vla/eval_ablation_60k.sh
# 三次 × 50 局 × 约 6 秒 ≈ 15 分钟。
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

STEP=060000
T=/mnt/f/vla/outputs/train

run_one () {
  local NAME="$1" CK="$2"
  local OUT="/mnt/f/vla/outputs/eval/${NAME}_60k"
  local LOG="/mnt/f/vla/${NAME}_60k_eval.log"
  if [ ! -d "$CK" ]; then echo "!! 跳过 $NAME：找不到 $CK"; return; fi
  echo
  echo "=== ${NAME} @ ${STEP} ==="
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
    > "$LOG" 2>&1 || { echo "  失败，见 $LOG"; tail -5 "$LOG"; return; }
  python /mnt/f/vla/eval_stat.py "$OUT" | head -2 | sed 's/^/  /'
}

run_one dp_ab_baseline  "$T/dp_ab_baseline/checkpoints/$STEP/pretrained_model"
run_one dp_ab_gn_only   "$T/dp_ab_gn_only/checkpoints/$STEP/pretrained_model"
run_one dp_ab_crop_only "$T/dp_ab_crop_only/checkpoints/$STEP/pretrained_model"

echo
echo "=== 60k vs 120k 对照 ==="
python3 - <<'PY'
import json, os
rows = []
for arm in ["dp_ab_baseline", "dp_ab_gn_only", "dp_ab_crop_only"]:
    line = [arm]
    for sfx in ["_60k", ""]:
        p = "/mnt/f/vla/outputs/eval/%s%s/eval_info.json" % (arm, sfx)
        if os.path.exists(p):
            o = json.load(open(p))["overall"]
            line.append("%5.1f%%" % o["pc_success"])
        else:
            line.append("  --  ")
    rows.append(line)
print("%-18s %8s %8s" % ("arm", "60k", "120k"))
for r in rows:
    print("%-18s %8s %8s" % tuple(r))
PY