"""逐局对比两轮评测，判断是否完全一致。

用法: python compare_runs.py <目录A> <目录B>

对比 sum_reward / max_reward / success（两边都有 done_steps 时也比它）。
用途：
  1. 可复现性检验 —— 同配置同 seed 跑两遍，应该逐局完全相同；
  2. 副作用检验 —— 改了评测代码但不该影响结果时，用它确认数字没动。
"""
import json
import sys


def load(path):
    with open(path.rstrip("/") + "/eval_info.json") as f:
        return json.load(f)["per_task"][0]["metrics"]


a, b = load(sys.argv[1]), load(sys.argv[2])
n = min(len(a["successes"]), len(b["successes"]))
has_steps = "done_steps" in a and "done_steps" in b
keys = ["sum_rewards", "max_rewards", "successes"] + (["done_steps"] if has_steps else [])

print("A = %s   成功 %d/%d" % (sys.argv[1], sum(a["successes"][:n]), n))
print("B = %s   成功 %d/%d" % (sys.argv[2], sum(b["successes"][:n]), n))
print("对比字段: %s%s\n" % (", ".join(keys), "" if has_steps else "（done_steps 只在一方有，跳过）"))

bad = []
for i in range(n):
    diffs = []
    for k in keys:
        x, y = a[k][i], b[k][i]
        same = abs(x - y) < 1e-9 if k in ("sum_rewards", "max_rewards") else x == y
        if not same:
            diffs.append("%s: %r != %r" % (k, x, y))
    if diffs:
        bad.append(i)
        print("ep%-3d  A sum=%.9f  B sum=%.9f" % (i, a["sum_rewards"][i], b["sum_rewards"][i]))
        for d in diffs:
            print("      " + d)

print()
if not bad:
    print("=> 全部 %d 局逐位一致" % n)
else:
    print("=> %d/%d 局不一致：%s" % (len(bad), n, bad))