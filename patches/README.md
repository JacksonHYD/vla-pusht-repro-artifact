# LeRobot 本地补丁

`lerobot-v0.6.1-local.patch` —— 我们对 vendored LeRobot 源码做的手工修改。
**没有这个补丁，本文的所有数字都复现不出来。**

- 基线提交：`7e241bd630a3719a56157a497ce5d08f244784f1`（`chore(dependencies): update uv.lock (#4284)`）
- 涉及文件：`src/lerobot/policies/act/modeling_act.py`、`src/lerobot/scripts/lerobot_eval.py`
- 规模：15 处 hunk，+83 / −8 行

## 怎么用

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout 7e241bd630a3719a56157a497ce5d08f244784f1
git apply /path/to/lerobot-v0.6.1-local.patch
pip install -e .
```

## 每处改了什么、为什么

### `modeling_act.py`

| 位置 | 改动 | 原因 |
|---|---|---|
| `class ACT` @430 | `device=batch[OBS_STATE].device` → `batch[ACTION].device` | 训练分支里只要 ACTION 在 batch 里就会执行，但**没有机器人状态输入的策略**里 `OBS_STATE` 不存在 → `KeyError` |
| `class ACT` @455 | 按 `OBS_STATE → OBS_IMAGES → ACTION → OBS_ENV_STATE` 顺序取可用张量的 device | 同上，推理分支 |

> 这两处是"把 `agent_pos` 从数据集里删掉"那个消融实验必须的——不改就必崩。

### `lerobot_eval.py`

| 位置 | 改动 | 原因 |
|---|---|---|
| `rollout()` @217 | 开头 `if seeds: set_seed(seeds[0])` | **动作采样用的是全局 torch RNG**。不按批重设的话，噪声流从上一次中断处继续，两个只差 episode_length 的配置会拿到不同的动作噪声，比较就不是配对的 |
| `rollout()` @293 | `LEROBOT_TASK_DESCRIPTION` 环境变量 | pusht 环境既无 `task_description` 也无 `task`，而 SmolVLA 需要一条指令文本 |
| `rollout()` @316 | `LEROBOT_DEBUG_ACTIONS=<N>` 打印前 N 个动作 | 调试用，不影响结果 |
| `eval_main()` @795 | `torch.backends.cudnn.benchmark = False`；`deterministic = True` | **最关键的一处**。`benchmark=True` 让 cuDNN 用实测计时挑卷积算法，逐进程而异 → 浮点累加顺序不同 → 同配置同 seed 重跑也有数值差。实测同配置跑两遍，ep0 的 `sum_reward` 是 197.30 vs 63.91 |
| `eval_main()` @842 | `LEROBOT_EVAL_MAX_VIDEOS` 环境变量 | 默认只录前 10 局视频（写死），要用更多时放开 |
| `eval_policy()` 等多处 | 逐局 `done_steps` 落盘 | 量"完成步数"用；`done` 在跑满上限时也会触发，所以只对成功局有意义 |

## 验证过什么

- 同配置两次独立评测**逐位一致**（`act100k_na10` vs `act100k_na10_videos`，50 局成功率与逐局结果完全相同）
- 单局 `seed=1005` 单独跑，与 50 局批次里的第 5 局完全一致 ⇒ **每局自成一体，不受前面跑过多少局影响**

## 注意

这些是**本地补丁**，不是上游 PR。上游的 `cudnn.deterministic` 相关修复见
`huggingface/lerobot#5132`（未合并），async env 的崩溃见 `#4932`（未合并）。
如果将来重装或 `git pull` LeRobot，这些改动会丢失或冲突。