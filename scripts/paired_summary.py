"""把两轮配对评测拆成 2×2 表，回答"多给的时间到底救回了多少局"。

用法: python paired_summary.py <300轮目录> <600轮目录>

输出四类局：
    both   两轮都成功
    only600 300 步没成、600 步成了   <- 这些就是"时间不够"救回来的
    only300 300 步成了、600 步没成   <- 配对失败才会出现，应该恒为 0
    neither 两轮都没成

再给出：
    * only600 那批局在 300 轮里的 max_reward —— 看它们"差多远"
    * 两轮 max_reward 的逐局差值分布 —— 注意它会饱和，只作参考
"""
import json
import sys


def load(path):
    with open(path.rstrip("/") + "/eval_info.json") as f:
        return json.load(f)["per_task"][0]["metrics"]


a, b = load(sys.argv[1]), load(sys.argv[2])
sa, sb = a["successes"], b["successes"]
ma, mb = a["max_rewards"], b["max_rewards"]
n = min(len(sa), len(sb))

both = [i for i in range(n) if sa[i] and sb[i]]
only600 = [i for i in range(n) if not sa[i] and sb[i]]
only300 = [i for i in range(n) if sa[i] and not sb[i]]
neither = [i for i in range(n) if not sa[i] and not sb[i]]

print("%s  vs  %s     n=%d\n" % (sys.argv[1], sys.argv[2], n))
print("  两轮都成功        both    %3d  (%.0f%%)" % (len(both), 100.0 * len(both) / n))
print("  只有600步成       only600 %3d  (%.0f%%)   <- 时间救回来的" % (len(only600), 100.0 * len(only600) / n))
print("  只有300步成       only300 %3d  (%.0f%%)   <- 应为 0，否则配对没成立" % (len(only300), 100.0 * len(only300) / n))
print("  两轮都没成        neither %3d  (%.0f%%)" % (len(neither), 100.0 * len(neither) / n))
print()

if only600:
    print("  only600 这些局在 300 轮里的 max_reward（离成功线多远）：")
    for i in only600:
        print("    ep%-3d  300轮 max=%.3f   600轮 max=%.3f" % (i, ma[i], mb[i]))
    print()

if only300:
    print("  ⚠️ only300 非空，配对没成立：")
    for i in only300:
        print("    ep%-3d  300轮 max=%.3f   600轮 max=%.3f" % (i, ma[i], mb[i]))
    print()

diffs = [mb[i] - ma[i] for i in range(n)]
nz = [d for d in diffs if abs(d) > 1e-9]
print("  max_reward 逐局差值：非零 %d/%d 局，最大 %.3f，最小 %.3f" % (
    len(nz), n, max(diffs), min(diffs)))
print("  注意 max_reward 在 1.0 处饱和，两边都成功的局差值恒为 0，所以它信息量有限。")