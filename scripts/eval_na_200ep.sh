#!/usr/bin/env bash
# 把两处 n_action_steps 扫描补到 200 局，让全篇口径统一（2x2 已经是 200 局）。
#
# ============ 为什么 ============
# 论文里 §4.1–4.4 全是 200 局，只有 §4.5（na 扫描）还是 50 局。审稿人一眼看得出。
# 50 局在 p≈0.25 附近的分辨力是 ±12pp，而 na 曲线的相邻档差往往只有几个点。
#
# ============ 协议 ============
# 200 局 / 300 步 / seed=1000 / batch 22 同步 —— 与 2x2 逐字一致。
# act100k_na10_200 已经存在（同协议），脚本会跳过它。
#
# ============ 怎么跑 ============
#   bash /mnt/f/vla/eval_na_200ep.sh
# ACT 8 档 + SmolVLA 4 档，约 100 分钟。
#
# ============ 如果 SmolVLA 在 batch 22 下 OOM ============
# 把下面 SMOLVLA_BATCH 改成 8，并在论文里注明 SmolVLA 用更小批（会略微改变动作噪声的
# 随机流，但不影响 na 各档之间的相对比较——四档用的是同一个 batch）。
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

N_EP=200
ACT_CK=/mnt/f/vla/outputs/train/act_pusht_100k/checkpoints/100000/pretrained_model
SMOL_CK=/mnt/f/vla/outputs/train/smolvla_expert/checkpoints/016000/pretrained_model
ACT_BATCH=22
SMOLVLA_BATCH=22

run_one () {
  local TAG="$1" CK="$2" NA="$3" BS="$4"
  local OUT="/mnt/f/vla/outputs/eval/${TAG}"
  local LOG="/mnt/f/vla/${TAG}_eval.log"
  if [ -f "$OUT/eval_info.json" ]; then
    printf "  %-28s 已存在，跳过\n" "$TAG"; return
  fi
  if [ ! -d "$CK" ]; then echo "  !! 找不到 $CK"; return; fi
  lerobot-eval \
    --policy.path="$CK" \
    --env.type=pusht \
    --env.episode_length=300 \
    --eval.n_episodes="$N_EP" \
    --eval.batch_size="$BS" \
    --eval.use_async_envs=false \
    --policy.n_action_steps="$NA" \
    --seed=1000 \
    --output_dir="$OUT" \
    > "$LOG" 2>&1 || { echo "  !! $TAG 失败，见 $LOG"; tail -3 "$LOG"; return; }
  python /mnt/f/vla/eval_stat.py "$OUT" | sed -n '2p' | sed "s/^/  $(printf '%-28s' $TAG)/"
}

echo "=== ACT（9 档，其中 na=10 复用已有结果）==="
for NA in 1 2 4 6 8 10 20 50 100; do
  run_one "act100k_na${NA}_200" "$ACT_CK" "$NA" "$ACT_BATCH"
done

echo
echo "=== SmolVLA（4 档）==="
for NA in 50 20 10 5; do
  run_one "smolvla_expert_na${NA}_200" "$SMOL_CK" "$NA" "$SMOLVLA_BATCH"
done

echo
echo "=== 汇总 ==="
python3 - <<'PY'
import json, os, math
def get(tag):
    p = "/mnt/f/vla/outputs/eval/%s/eval_info.json" % tag
    if not os.path.exists(p): return None
    o = json.load(open(p))["overall"]
    return o["n_episodes"], o["pc_success"], o.get("avg_max_reward", 0)

print("ACT  n_action_steps 扫描（200 局）")
print("%-8s %6s %9s %9s" % ("na","n","succ%","max_r"))
for na in (1,2,4,6,8,10,20,50,100):
    r = get("act100k_na%d_200" % na)
    print("%-8d %6s %8s %9s" % (na, r[0], "%.1f%%"%r[1], "%.3f"%r[2]) if r else
          ("%-8d  -- " % na))
print()
print("SmolVLA  n_action_steps 扫描（200 局）")
print("%-8s %6s %9s %9s" % ("na","n","succ%","max_r"))
for na in (50,20,10,5):
    r = get("smolvla_expert_na%d_200" % na)
    print("%-8d %6s %8s %9s" % (na, r[0], "%.1f%%"%r[1], "%.3f"%r[2]) if r else
          ("%-8d  -- " % na))
PY