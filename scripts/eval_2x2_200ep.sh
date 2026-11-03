#!/usr/bin/env bash
# 把 2x2 的四格各评到 200 局，判断"crop_only 从 64% 掉到 50%"是真的还是 50 局的噪声。
#
# ============ 为什么是 200 局 ============
# 50 局在 p≈0.5 附近的分辨力只有 ±14pp。而当前争议的差值恰好是 14pp（64% vs 50%），
# 所以 50 局注定分不出来。要分辨这个量级，每组约需 200 局。
#
# ============ 协议（与之前逐字一致，否则不可比）============
# 300 步回合 / na=8 / seed=1000 / batch 22 同步环境。
# 输出到 *_200 目录，不会覆盖已有的 50 局结果。
#
# ============ 怎么跑 ============
#   bash /mnt/f/vla/eval_2x2_200ep.sh          # 只评 120k 四格，约 80 分钟
#   bash /mnt/f/vla/eval_2x2_200ep.sh 60k      # 连 60k 三格一起评，约 140 分钟
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

N_EP=200
T=/mnt/f/vla/outputs/train
INCLUDE_60K="${1:-no}"

run_one () {
  local TAG="$1" CK="$2"
  local OUT="/mnt/f/vla/outputs/eval/${TAG}"
  local LOG="/mnt/f/vla/${TAG}_eval.log"
  if [ ! -d "$CK" ]; then echo "!! 跳过 $TAG：找不到 $CK"; return; fi
  echo
  echo "=== ${TAG}  (${N_EP} 局) ==="
  lerobot-eval \
    --policy.path="$CK" \
    --env.type=pusht \
    --env.episode_length=300 \
    --eval.n_episodes="$N_EP" \
    --eval.batch_size=22 \
    --eval.use_async_envs=false \
    --policy.n_action_steps=8 \
    --seed=1000 \
    --output_dir="$OUT" \
    > "$LOG" 2>&1 || { echo "  失败，见 $LOG"; tail -5 "$LOG"; return; }
  python /mnt/f/vla/eval_stat.py "$OUT" | sed -n '2p' | sed 's/^/  /'
}

# ---- 120k 四格 ----
run_one dp_ab_baseline_200   "$T/dp_ab_baseline/checkpoints/120000/pretrained_model"   # GN关 crop关
run_one dp_ab_gn_only_200    "$T/dp_ab_gn_only/checkpoints/120000/pretrained_model"    # GN开 crop关
run_one dp_ab_crop_only_200  "$T/dp_ab_crop_only/checkpoints/120000/pretrained_model"  # GN关 crop开
run_one dp_ab_gn_crop_200    "$T/dp_repro/checkpoints/120000/pretrained_model"         # GN开 crop开

if [ "$INCLUDE_60K" = "60k" ]; then
  run_one dp_ab_baseline_60k_200   "$T/dp_ab_baseline/checkpoints/060000/pretrained_model"
  run_one dp_ab_gn_only_60k_200    "$T/dp_ab_gn_only/checkpoints/060000/pretrained_model"
  run_one dp_ab_crop_only_60k_200  "$T/dp_ab_crop_only/checkpoints/060000/pretrained_model"
fi

echo
echo "=== 汇总（200 局）==="
python3 - <<'PY'
import json, os, math

def load(tag):
    p = "/mnt/f/vla/outputs/eval/%s/eval_info.json" % tag
    if not os.path.exists(p): return None
    o = json.load(open(p))["overall"]
    return o["n_episodes"], o["pc_success"], o.get("avg_max_reward", 0)

def wilson(k, n, z=1.96):
    p = k/n; d = 1+z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0,c-h), min(1,c+h)

cells = {("off","off"):"dp_ab_baseline_200", ("on","off"):"dp_ab_gn_only_200",
         ("off","on"):"dp_ab_crop_only_200", ("on","on"):"dp_ab_gn_crop_200"}
res = {}
print("%-22s %5s %8s %18s" % ("cell","n","succ%","95% CI"))
for (a,b), tag in cells.items():
    r = load(tag)
    res[(a,b)] = r
    if r:
        n, s, mr = r; k = round(n*s/100)
        lo, hi = wilson(k, n)
        print("%-22s %5d %7.1f%% [%5.1f, %5.1f]" % ("GN=%s crop=%s"%(a,b), n, s, 100*lo, 100*hi))
    else:
        print("%-22s   缺失" % ("GN=%s crop=%s"%(a,b)))

if all(res.values()):
    oo, no, on, nn = (res[k][1] for k in [("off","off"),("on","off"),("off","on"),("on","on")])
    print()
    print("2x2（成功率 %%）        crop off    crop on")
    print("  GN off                %6.1f      %6.1f" % (oo, on))
    print("  GN on                 %6.1f      %6.1f" % (no, nn))
    print()
    print("主效应 A（GN+从零骨干）: %+.1f pp" % (((no-oo)+(nn-on))/2))
    print("主效应 B（crop）       : %+.1f pp" % (((on-oo)+(nn-no))/2))
    print("交互项                 : %+.1f pp" % ((nn-no)-(on-oo)))
PY