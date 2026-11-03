#!/usr/bin/env bash
# 补齐"噪声地板"的最后一块：跨种子块的实测离散。
# 同一 checkpoint、同一配置（na=10、300 步、50 局），只换种子块。
# 10 个块 × 50 局，ACT 约 1.4 秒/局 ⇒ 约 12 分钟。
#
# ============ 怎么跑 ============
# 必须在 WSL 里跑（PowerShell 里的 bash 是 Git Bash，没有 conda.sh）：
#   bash /mnt/f/vla/seedblocks_act.sh
# 或从 PowerShell（整体包在引号里，否则 MSYS 会把 /mnt/... 改写成 C:/Program Files/Git/...）：
#   wsl -d Ubuntu-22.04 -e bash -lc "bash /mnt/f/vla/seedblocks_act.sh"
#
# 跑完之后：python3 /mnt/f/vla/noise_floor.py   ← 表 H 会自动出现
#
# ============ 为什么要换种子块 ============
# 你的确定性补丁让"同种子重跑"逐位相同（act100k_na10 与 act100k_na10_videos 已验证）。
# 于是同种子重跑**不提供**任何关于评测不确定性的信息——它只会给你同一个数。
# 真正的重复只能是"换一批初始状态"，也就是换 --seed。
#
# ============ 为什么是这 10 个种子 ============
# seed=1000 已跑过（act100k_na10），seed=2000..2199 已跑过（cmp_state_200）。
# 从 3000 起每 1000 一个块，保证和已有数据不重叠。
#
# ============ 终端要一直开着 ============
# WSL2 在最后一个客户端断开后约 60 秒会销毁整个 VM，nohup 挡不住。整个时长别关。
set -euo pipefail

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lerobot

POLICY=/mnt/f/vla/outputs/train/act_pusht_100k/checkpoints/100000/pretrained_model
NA=10
N_EP=50

for SEED in 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000; do
  OUT=/mnt/f/vla/outputs/eval/seedblock_${SEED}
  LOG=/mnt/f/vla/seedblock_${SEED}.log
  echo
  echo "=== seed=${SEED} / ${N_EP} 局 / na=${NA} ==="
  if ! lerobot-eval \
      --policy.path="$POLICY" \
      --env.type=pusht \
      --env.episode_length=300 \
      --eval.n_episodes="$N_EP" \
      --eval.batch_size=1 \
      --policy.n_action_steps="$NA" \
      --seed="$SEED" \
      --output_dir="$OUT" \
      > "$LOG" 2>&1; then
    echo "  这一档失败，日志尾巴如下："
    tail -20 "$LOG"
    exit 1
  fi
  python /mnt/f/vla/eval_stat.py "$OUT" | head -2 | sed 's/^/  /'
done

echo
echo "=== 汇总：生成表 H ==="
python3 /mnt/f/vla/noise_floor.py | sed -n '/## 表 H/,$p'