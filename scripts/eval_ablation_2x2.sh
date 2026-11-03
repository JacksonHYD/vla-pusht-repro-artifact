#!/usr/bin/env bash
# 2x2 消融的评测：四个格子用完全相同的协议评一遍。
#
# ============ 协议 ============
# 50 局 / batch 22 同步环境 / episode_length 300 / n_action_steps 8 / seed=1000。
# 四格必须逐字一致，否则比出来的不是主效应。
#
# 为什么是 50 局而不是 500：主效应预计在 10-30 个百分点量级，
# 50 局在 p≈0.5 附近的分辨力约 ±14pp（见 噪声地板表.md 表 D2）。
# 如果某一对差得接近 14pp，再补到 200 局。
#
# 怎么跑（WSL 里）：bash /mnt/f/vla/eval_ablation_2x2.sh
# 四次 × 50 局，按 6 秒/局算约 20 分钟。
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

STEP=120000

run_one () {
  local NAME="$1" CK="$2"
  local OUT="/mnt/f/vla/outputs/eval/${NAME}"
  local LOG="/mnt/f/vla/${NAME}_eval.log"
  if [ ! -d "$CK" ]; then
    echo "!! 跳过 ${NAME}：找不到 $CK"
    return
  fi
  echo
  echo "=== ${NAME} ==="
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

T=/mnt/f/vla/outputs/train
run_one dp_ab_baseline   "$T/dp_ab_baseline/checkpoints/$STEP/pretrained_model"    # A off, B off
run_one dp_ab_gn_only    "$T/dp_ab_gn_only/checkpoints/$STEP/pretrained_model"     # A on,  B off
run_one dp_ab_crop_only  "$T/dp_ab_crop_only/checkpoints/$STEP/pretrained_model"   # A off, B on
run_one dp_ab_gn_crop    "$T/dp_repro/checkpoints/$STEP/pretrained_model"          # A on,  B on（复用已有权重）

echo
echo "=== 四格汇总 ==="
python3 - <<'PY'
import json, glob, os
rows = []
for p in sorted(glob.glob("/mnt/f/vla/outputs/eval/dp_ab_*/eval_info.json")):
    d = json.load(open(p)); o = d["overall"]
    rows.append((os.path.basename(os.path.dirname(p)), o["n_episodes"],
                 o["pc_success"], o.get("avg_max_reward", 0)))
print("%-18s %4s %8s %8s" % ("arm", "n", "succ%", "max_r"))
for r in rows:
    print("%-18s %4d %7.1f%% %8.3f" % r)
PY