"""300 步 vs 600 步的配对比较。

依据 lerobot_eval.py:801 的 `start_seed=cfg.seed` 和 :531 的
`seeds = range(start_seed + batch_ix*num_envs, ...)`，batch_size=1 时
第 i 局用 seed = 1000+i。两轮都用的默认 --seed 1000，
所以**第 i 局的初始状态在两轮里是同一个** —— 这是配对实验。

注意：配对的是初始状态，不是随机噪声。DP 是随机策略（扩散采样），
set_seed 只在评测开始时调一次，轨迹一分岔噪声序列就错开了。
所以配对能消掉"初始条件"这一项方差，但消不掉策略随机性。
"""
import glob
import json
import os
import subprocess

EVAL = "/mnt/f/vla/outputs/eval"


def load(run):
    d = json.load(open(os.path.join(EVAL, run, "eval_info.json")))
    m = d["per_task"][0]["metrics"]
    return m


a = load("dp100k_na8_50")   # 300 步
b = load("dp100k_na8_600")  # 600 步

mra = a["max_rewards"]; ssa = a["successes"]; sra = a["sum_rewards"]
mrb = b["max_rewards"]; ssb = b["successes"]; srb = b["sum_rewards"]

print(f"300 步: 成功率 {100*sum(ssa)/len(ssa):.1f}%   平均 max_reward {sum(mra)/len(mra):.3f}")
print(f"600 步: 成功率 {100*sum(ssb)/len(ssb):.1f}%   平均 max_reward {sum(mrb)/len(mrb):.3f}")

# ---- 配对翻转表 ----
both = [i for i in range(50) if ssa[i] and ssb[i]]
only600 = [i for i in range(50) if not ssa[i] and ssb[i]]
only300 = [i for i in range(50) if ssa[i] and not ssb[i]]
neither = [i for i in range(50) if not ssa[i] and not ssb[i]]

print("\n=== 配对翻转（seed = 1000+i，初始状态相同）===")
print(f"  两轮都成功            : {len(both):>2}")
print(f"  300 败 → 600 成功     : {len(only600):>2}   ← 加时间救回来的")
print(f"  300 成 → 600 失败     : {len(only300):>2}   ← 反向翻（策略随机性，不是时间造成的）")
print(f"  两轮都失败            : {len(neither):>2}")
print(f"\n  McNemar 意义上的净变化: +{len(only600)} - {len(only300)} = "
      f"{len(only600)-len(only300)} 局")

print("\n=== 600 步之后仍然失败的局 ===")
print(f"{'ep':>4}{'300步 max_r':>13}{'600步 max_r':>13}{'600覆盖':>10}{'sum/max':>9}")
for i in neither:
    cov = mrb[i] * 0.95
    ratio = srb[i] / mrb[i] if mrb[i] > 0 else float("nan")
    print(f"{i:>4}{mra[i]:>13.4f}{mrb[i]:>13.4f}{cov:>10.4f}{ratio:>9.1f}")

if only300:
    print("\n=== 反向翻的局（提示策略随机性有多大）===")
    for i in only300:
        print(f"  ep{i}: 300步 {mra[i]:.4f} → 600步 {mrb[i]:.4f}")

# ---- 用录像量实际回合长度 ----
print("\n=== 前 10 局录像时长（只录前 10 局，写死在 lerobot_eval.py:786）===")


def vid_lens(run):
    """录像时长（秒）。WSL 里没装 ffprobe，用 PyAV 读（LeRobot 自带）。"""
    import av
    out = []
    for i in range(10):
        p = os.path.join(EVAL, run, "videos", "pusht_0", f"eval_episode_{i}.mp4")
        if not os.path.exists(p):
            out.append(None); continue
        with av.open(p) as c:
            n = c.streams.video[0].frames
            if not n:  # 容器没写帧数就退回时长
                n = int(round((c.duration / av.time_base) * 10))
            else:
                n = int(n)
        out.append(n / 10.0)  # fps=10 → 秒
    return out


la = vid_lens("dp100k_na8_50")
lb = vid_lens("dp100k_na8_600")
print(f"{'ep':>4}{'300步(s)':>11}{'=步数':>8}{'600步(s)':>11}{'=步数':>8}   判定(300→600)")
for i in range(10):
    sa = f"{la[i]:.2f}" if la[i] is not None else "n/a"
    sb = f"{lb[i]:.2f}" if lb[i] is not None else "n/a"
    na = f"{la[i]*10:.0f}" if la[i] is not None else "-"
    nb = f"{lb[i]*10:.0f}" if lb[i] is not None else "-"
    print(f"{i:>4}{sa:>11}{na:>8}{sb:>11}{nb:>8}   "
          f"{'成' if ssa[i] else '败'} → {'成' if ssb[i] else '败'}")

print("\n（10 Hz 控制，步数 = 秒 × 10）")
print("示范数据集每集长度 49–246 帧（4.9–24.6 秒，均值 12.4 秒）—— 拿它当'人类多快做完'的参照")