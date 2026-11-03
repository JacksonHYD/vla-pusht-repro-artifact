import json, sys, statistics
p = sys.argv[1].rstrip("/") + "/eval_info.json"
d = json.load(open(p))
pt = d["per_task"][0]["metrics"]
mr = pt["max_rewards"]; ss = pt["successes"]
n = len(mr); sd = statistics.stdev(mr) if len(mr) > 1 else 0.0
print("%s" % p)
print("  n=%d mean=%.3f std=%.3f sem=%.3f succ=%d/%d  >=0.9:%d  <=0.05:%d" % (
  n, statistics.mean(mr), sd, sd / n ** 0.5,
  sum(1 for x in ss if x), n,
  sum(1 for x in mr if x >= 0.9), sum(1 for x in mr if x <= 0.05)))
print("  sorted: " + " ".join("%.2f" % x for x in sorted(mr)))

# 结束步数（2026-10-20 起 lerobot_eval.py 会写 done_steps）。
# 注意：跑满步数上限也会触发 done，所以"未成功"的那批数值多半就等于上限。
# 真正有信息量的是"成功局用了多少步"。
ds = pt.get("done_steps")
if ds:
  succ_steps = [ds[i] for i in range(n) if ss[i] and ds[i] > 0]
  fail_steps = [ds[i] for i in range(n) if not ss[i] and ds[i] > 0]
  missing = sum(1 for x in ds if x <= 0)
  print("  结束步数: 成功局 %d / 未成功局 %d / 无记录 %d" % (len(succ_steps), len(fail_steps), missing))
  if succ_steps:
    sd2 = statistics.stdev(succ_steps) if len(succ_steps) > 1 else 0.0
    print("    成功局  min=%d median=%d mean=%.1f max=%d std=%.1f" % (
      min(succ_steps), statistics.median(succ_steps), statistics.mean(succ_steps),
      max(succ_steps), sd2))
  if fail_steps:
    print("    未成功局 median=%d max=%d（多数等于步数上限，本来就是“跑满也没成”）" % (
      statistics.median(fail_steps), max(fail_steps)))
  print("    逐局: " + " ".join(str(x) if x > 0 else "-" for x in ds))