#!/usr/bin/env python3
"""把 outputs/eval/ 下所有评测整理成"噪声地板"表。

用途：路线 A（评测方法论）的第一张表。只读 eval_info.json，不需要 GPU。
怎么跑（WSL 里）：python3 /mnt/f/vla/noise_floor.py

为什么需要这张表：
  一个成功率的数字背后有两种不确定性——
  (1) 抽样噪声：只跑 50 局，换 50 个初始状态数字就会变，这是二项的、可以精确算的；
  (2) 越界的离散：换评测种子后，实际波动是不是比 (1) 预测的更大。
  没有 (1)，任何"提高了 6 个百分点"的说法都无法判断真假。

配对性说明：
  sweep 里同一 --seed 的多次评测共享同一批初始状态，因此是"配对"的。
  配对数据要用 McNemar 精确检验，不能当成两个独立样本做比例检验。
"""
import glob
import json
import math
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# 同一 --seed=1000、50 局、episode_length=300、batch_size=1 —— 成对可比
PAIRED_NA = ["act100k_na1", "act100k_na2", "act100k_na4", "act100k_na6",
             "act100k_na8", "act100k_na10", "act100k_na20",
             "act100k_na50", "act100k_na100"]
PAIRED_TE = ["act100k_te", "act100k_te003", "act100k_te01", "act100k_te03"]

Z_A = 1.959964   # 双侧 95%
Z_B = 0.841621   # 功效 80%


def load_all():
    runs = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "outputs", "eval", "*", "eval_info.json"))):
        name = os.path.basename(os.path.dirname(p))
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        pt = d.get("per_task") or []
        if not pt:
            continue
        m = pt[0]["metrics"]
        succ = [bool(x) for x in (m.get("successes") or [])]
        runs[name] = {"n": len(succ), "k": sum(succ), "succ": succ}
    return runs


def wilson(k, n, z=1.96):
    """成功率的 Wilson 置信区间——n 很小时比正态近似可靠得多。"""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def mcnemar_exact(ref, cmp_):
    """配对两分类的精确 McNemar。返回 (ref赢, cmp赢, 双侧 p)。"""
    only_cmp = sum(1 for r, c in zip(ref, cmp_) if c and not r)
    only_ref = sum(1 for r, c in zip(ref, cmp_) if r and not c)
    n = only_cmp + only_ref
    if n == 0:
        return only_ref, only_cmp, 1.0
    m = min(only_ref, only_cmp)
    tail = sum(math.comb(n, i) for i in range(m + 1)) / (2 ** n)
    return only_ref, only_cmp, min(1.0, 2 * tail)


def n_req(p1, p2):
    """两组独立比较时，检测 p1 vs p2 所需的每组局数（α=0.05 双侧，功效 80%）。"""
    d = abs(p1 - p2)
    if d == 0:
        return float("inf")
    return (Z_A + Z_B) ** 2 * (p1 * (1 - p1) + p2 * (1 - p2)) / (d * d)


def main():
    runs = load_all()
    out = []
    w = out.append

    w("# 噪声地板表（pusht，既有评测结果整理）")
    w("")
    w("全部数字来自 `outputs/eval/*/eval_info.json`。重跑本表：`python3 noise_floor.py`。")
    w("")

    # ---------- 表 A ----------
    w("## 表 A · 每一次评测的精度")
    w("")
    w("`95% 区间` 是 Wilson 区间。**半宽**列的意思是：如果换一批初始状态重跑，")
    w("数字最可能在这个范围内摆动。")
    w("")
    w("| 评测 | 局数 n | 成功 k | 成功率 | 95% 区间 | 半宽 |")
    w("|---|---|---|---|---|---|")
    for name in sorted(runs):
        r = runs[name]
        if r["n"] < 10:
            continue
        lo, hi = wilson(r["k"], r["n"])
        half = (hi - lo) / 2
        w(f"| `{name}` | {r['n']} | {r['k']} | {100*r['k']/r['n']:.1f}% | "
          f"[{100*lo:.1f}, {100*hi:.1f}]% | ±{100*half:.1f}pp |")
    w("")

    # ---------- 表 B ----------
    w("## 表 B · `n_action_steps` 扫描的配对检验")
    w("")
    w("基准取 `act100k_na10`（14/50，最高档）。`只A成` / `只B成` 是配对里")
    w("两边结果不同的局数——**配对检验只看这些局**，这是它比未配对灵敏的地方。")
    w("`p` 是精确 McNemar 双侧 p 值。")
    w("")
    ref = runs.get("act100k_na10", {}).get("succ")
    if ref:
        w("| 对比 | 成功率 | 只基准成 | 只对照成 | p 值 | 在 0.05 下可区分？ |")
        w("|---|---|---|---|---|---|")
        for name in PAIRED_NA:
            if name not in runs or name == "act100k_na10":
                continue
            r = runs[name]
            a, b, p = mcnemar_exact(ref, r["succ"])
            verdict = "**是**" if p < 0.05 else "否"
            w(f"| `{name}` vs na10 | {100*r['k']/r['n']:.1f}% | {a} | {b} | "
              f"{p:.4f} | {verdict} |")
        w("")

    # ---------- 表 C ----------
    w("## 表 C · 时间集成（temporal ensembling）的配对检验")
    w("")
    w("⚠️ 开 ensembling 时 `n_action_steps` 被迫为 1（`configuration_act.py:138` 写死），")
    w("所以 **对 na10 的比较不是同类比较**——那是\"ensembling+na1\"对\"硬切 na10\"。")
    w("公平的同类对照是下面第二张表：同一套 na=1 的框架下，开与不开 ensembling。")
    w("")
    if ref and all(t in runs for t in PAIRED_TE):
        w("### C1 · 对上\"硬切 na=10\"")
        w("")
        w("| 对比 | 成功率 | 只基准成 | 只对照成 | p 值 | 在 0.05 下可区分？ |")
        w("|---|---|---|---|---|---|")
        for name in PAIRED_TE:
            r = runs[name]
            a, b, p = mcnemar_exact(ref, r["succ"])
            verdict = "**是**" if p < 0.05 else "否"
            w(f"| `{name}` vs na10 | {100*r['k']/r['n']:.1f}% | {a} | {b} | "
              f"{p:.4f} | {verdict} |")
        w("")

    base1 = runs.get("act100k_na1", {}).get("succ")
    if base1 and all(t in runs for t in PAIRED_TE):
        w("### C2 · 对上\"关 ensembling、同样 na=1\"")
        w("")
        w("| 对比 | 成功率 | 只 na1 成 | 只对照成 | p 值 | 在 0.05 下可区分？ |")
        w("|---|---|---|---|---|---|")
        for name in PAIRED_TE:
            r = runs[name]
            a, b, p = mcnemar_exact(base1, r["succ"])
            verdict = "**是**" if p < 0.05 else "否"
            w(f"| `{name}` vs na1 | {100*r['k']/r['n']:.1f}% | {a} | {b} | "
              f"{p:.4f} | {verdict} |")
        w("")
        w("⚠️ **这里的 p=1 不是\"证明没差别\"，是\"测不出来\"**：na1 是 0/50，")
        w("两边都贴着地板，不一致局只有 0–1 个，50 局没有任何功效去分辨 2pp。")
        w("在地板/天花板附近，\"不显著\"几乎不携带信息——这是这张表要说明的另一件事。")
        w("")

    # ---------- 表 D ----------
    w("## 表 D · 要分辨 Δ 个百分点需要多少局")
    w("")
    w("两组独立比较，α=0.05 双侧，功效 80%，每组所需局数。")
    w("**配对比较通常远少于这个数**——只要两档在大多数局上结果相同。")
    w("")
    w("| 基线成功率 | 差 5pp | 差 10pp | 差 15pp | 差 20pp |")
    w("|---|---|---|---|---|")
    for p0 in (0.10, 0.20, 0.30, 0.40, 0.50):
        cells = []
        for d in (0.05, 0.10, 0.15, 0.20):
            n = n_req(p0, p0 + d)
            cells.append(f"{math.ceil(n):,}")
        w(f"| {int(p0*100)}% | " + " | ".join(cells) + " |")
    w("")
    w(f"你现在用的 50 局，在 p≈0.30 附近，未配对时只能分辨约 "
      f"**{100*2.8016*math.sqrt(2*0.3*0.7/50):.0f}pp** 以上的差异。")
    w("")
    w("### D2 · 反过来问：给你 n 局，最小能测出多大差异")
    w("")
    w("| 每组局数 | p=20% 时能分辨 | p=30% 时能分辨 | p=50% 时能分辨 |")
    w("|---|---|---|---|")
    for n in (10, 50, 100, 200, 500, 1000):
        cells = []
        for p0 in (0.20, 0.30, 0.50):
            mdd = (Z_A + Z_B) * math.sqrt(2 * p0 * (1 - p0) / n)
            cells.append(f"±{100*mdd:.0f}pp")
        w(f"| {n} | " + " | ".join(cells) + " |")
    w("")
    w("### D3 · 配对比较的经验阈值")
    w("")
    w("McNemar 精确检验只看\"两边结果不同的局\"。若这些局**全部**偏向同一侧，")
    w("达到 p<0.05 所需的不一致局数是：6 局（p=0.031）。")
    w("也就是说：**只要 6 局里 6:0 偏向一边，就已经显著**——这比\"成功率差几个点\"")
    w("可靠得多，而且不需要上千局。")
    w("")

    # ---------- 表 F ----------
    w("## 表 E · 已有多组配对对比：方法在真实数据上的样子")
    w("")
    w("这两组都满足\"同一 `--seed`、同一批初始状态\"，可以直接做配对检验。")
    w("它们对应论文里最常见的两种比较类型：换观测、换任务定义。")
    w("")

    pairs = [
        ("cmp_state_200", "cmp_nostate_200",
         "给不给 agent_pos（同模型，200 局，seed=2000）"),
        ("base300_det", "base600_det",
         "回合上限 300 vs 600（同模型，50 局，seed=1000）"),
    ]
    w("| 比较 | A 成功率 | B 成功率 | 只 A 成 | 只 B 成 | p 值 | 结论 |")
    w("|---|---|---|---|---|---|---|")
    for a_name, b_name, desc in pairs:
        ra, rb = runs.get(a_name), runs.get(b_name)
        if not ra or not rb or len(ra["succ"]) != len(rb["succ"]):
            w(f"| {desc} | — | — | — | — | — | 局数不等，无法配对 |")
            continue
        a, b, p = mcnemar_exact(ra["succ"], rb["succ"])
        if p < 0.05:
            who = "B 更好" if b > a else "A 更好"
            verdict = f"**可区分（{who}）**"
        else:
            verdict = "不可区分"
        w(f"| {desc} | {100*ra['k']/ra['n']:.1f}% | {100*rb['k']/rb['n']:.1f}% | "
          f"{a} | {b} | {p:.4f} | {verdict} |")
    w("")
    w("**为什么 `dp30k_na8_50` vs `dp100k_na8_50` 被排除**：那一对看起来是现成的配对对比，")
    w("但两边都没有显式写 `--seed`（`eval_base300_det.sh:21` 的注释点明了这一点），")
    w("而且跑在 `cudnn.benchmark=True` 下，同配置重跑都会变。")
    w("**配对的前提是\"同一批初始状态\"，这一点验证不了就不能算 p 值。**")
    w("这恰好是本文要主张的纪律：先证书配对条件成立，再报显著性。")
    w("")

    # ---------- 表 E ----------
    w("## 表 F · 确定性验证")
    w("")
    a = runs.get("act100k_na10", {}).get("succ")
    b = runs.get("act100k_na10_videos", {}).get("succ")
    if a and b:
        same = a == b
        w(f"同一配置两次独立评测（`act100k_na10` / `act100k_na10_videos`）："
          f"逐局结果{'**完全相同**' if same else '不同'}。")
        w("")
        if same:
            w("说明确定性补丁生效：这组数据里**没有**可以用作\"独立重复\"的样本，")
            w("所以表 A 的区间只能靠二项理论算，不能靠重跑估。")
    w("")

    # ---------- 表 G ----------
    w("## 表 G · 跨种子块的一致性（唯一的\"真重复\"）")
    w("")
    w("`act100k_na10`（seed=1000，50 局）与 `cmp_state_200`（seed=2000，200 局）")
    w("是**同一配置、同一 checkpoint、不同初始状态**——这才是独立的重复。")
    w("")
    a, b = runs.get("act100k_na10"), runs.get("cmp_state_200")
    if a and b:
        p1, n1 = a["k"] / a["n"], a["n"]
        p2, n2 = b["k"] / b["n"], b["n"]
        se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
        diff = abs(p1 - p2)
        w("| 种子块 | 局数 | 成功率 |")
        w("|---|---|---|")
        w(f"| seed=1000 | {n1} | {100*p1:.1f}% |")
        w(f"| seed=2000 | {n2} | {100*p2:.1f}% |")
        w("")
        w(f"观测差 **{100*diff:.1f}pp**；若两块都是独立二项抽样，"
          f"差的标准误应为 **{100*se:.1f}pp**；观测差 = **{diff/se:.2f} σ**。")
        w("")
        w("两块一致 ⇒ 二项模型**没有**低估评测的不确定性。")
        w("但**两个点估不出方差**。要真的量出\"跨种子离散\"，需要 ≥5 个互不重叠的种子块；")
        w("ACT 每 50 局只要约 72 秒，10 个块约 12 分钟——这是这张表目前最便宜的缺口。")

    # ---------- 表 H ----------
    # 种子块有两族，**不能混用**：同一个 checkpoint 在 CPU 与 GPU 上的浮点结果略有出入，
    # 混在一起量出来的"块间离散"就掺了设备差异，不再是纯种子效应。
    cpu_blocks = sorted(k for k in runs if k.startswith("seedblock_cpu_"))
    gpu_blocks = sorted(k for k in runs
                        if k.startswith("seedblock_") and not k.startswith("seedblock_cpu_"))
    if cpu_blocks:
        series = [(k.replace("seedblock_cpu_", "seed="), runs[k]) for k in cpu_blocks]
        family = "CPU"
    elif len(gpu_blocks) >= 3:
        series = []
        if "act100k_na10" in runs:
            series.append(("seed=1000（GPU）", runs["act100k_na10"]))
        series += [(k.replace("seedblock_", "seed="), runs[k]) for k in gpu_blocks]
        family = "GPU"
    else:
        series = []
        family = None

    if len(series) >= 3:

        ns = [r["n"] for _, r in series]
        ks = [r["k"] for _, r in series]
        m = len(series)
        p_hat = sum(ks) / sum(ns)
        ps = [k / n for (_, _), k, n in zip(series, ks, ns)]
        obs_sd = math.sqrt(sum((p - p_hat) ** 2 for p in ps) / (m - 1))
        pred_sd = math.sqrt(p_hat * (1 - p_hat) / ns[0])
        chi2 = sum((k - n * p_hat) ** 2 for k, n in zip(ks, ns)) / (ns[0] * p_hat * (1 - p_hat))
        df = m - 1
        z = ((chi2 / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
        p_over = 0.5 * math.erfc(z / math.sqrt(2))

        w("## 表 H · 跨种子块的实测离散")
        w("")
        w(f"同一 checkpoint、同一配置、只换初始状态。**这才是\"噪声地板\"的实测值。**")
        w(f"本表全部来自 **{family}** 上的评测；不同设备的块不会混在一起算。")
        w("")
        w("| 种子块 | 局数 | 成功 | 成功率 |")
        w("|---|---|---|---|")
        for (label, _), k, n in zip(series, ks, ns):
            w(f"| {label} | {n} | {k} | {100*k/n:.1f}% |")
        w("")
        w(f"块间实测标准差 **±{100*obs_sd:.1f}pp**；二项模型预测 **±{100*pred_sd:.1f}pp**；"
          f"比值 **{obs_sd/pred_sd:.2f}**。")
        w("")
        w(f"过度离散检验：χ²={chi2:.1f}，df={df}，p={p_over:.3f}。")
        if p_over < 0.05:
            w("")
            w("⇒ **p<0.05：实测离散显著大于二项预测。** 说明换种子带来的波动不止"
              "抽样噪声，评测之间还有别的东西在动。这正是论文要报的那个数。")
        else:
            w("")
            w("⇒ **p≥0.05：没有证据表明离散超出二项预测。** 也就是说，"
              "只要按二项公式算，就能正确估计评测的不确定性——这本身是一个可报的结论。")

    md = "\n".join(out)
    path = os.path.join(ROOT, "噪声地板表.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(md + "\n")
    print(md)
    print(f"\n[已写入] {path}")


if __name__ == "__main__":
    main()