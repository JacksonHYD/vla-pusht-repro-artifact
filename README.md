# One Config Value, Thirty Points — artifact

Code, raw results and source patches for the paper *"One Config Value, Thirty
Points: Configuration Defaults and the Reproducibility of a Robot Learning
Benchmark"*.

## What is here

| Path | Contents |
|---|---|
| `paper/main.tex` | the manuscript (compile with pdflatex, or upload to Overleaf) |
| `results/eval/` | 65 `eval_info.json` files — per-episode `successes`, `max_rewards`, `done_steps` for every evaluation reported in the paper |
| `patches/` | 15 source changes against LeRobot `7e241bd6`, without which the numbers are not reproducible |
| `scripts/` | training, evaluation and statistics scripts |
| `训练结果汇总.md` | the consolidated result tables |
| `噪声地板表.md` | the evaluation noise-floor analysis |

## Headline numbers

| Setting | Success (200 episodes) |
|---|---|
| Diffusion Policy, library defaults | 33.0 % (66/200) |
| + `crop_shape=[84,84]` only | 55.0 % (110/200) |
| + GroupNorm and random backbone | 63.5 % (127/200) |
| published reference | 65.4 % (500 episodes) |
| same checkpoint, 11 seed blocks of 50 | 14 % – 32 % |

## Reproducing

```bash
git clone https://github.com/huggingface/lerobot.git
cd lerobot && git checkout 7e241bd630a3719a56157a497ce5d08f244784f1
git apply /path/to/patches/lerobot-v0.6.1-local.patch
pip install -e .
```

Then, from the root of this artifact:

```bash
# train one cell of the 2x2 (~7.3 h on one RTX 4060 Laptop)
bash scripts/train_ablation_2x2.sh

# evaluate every cell at 200 episodes (~80 min)
bash scripts/eval_2x2_200ep.sh
```

Every evaluation writes `outputs/eval/<tag>/eval_info.json`; `scripts/eval_stat.py`
prints a one-line summary for a given run.

## Hardware

A single consumer GPU (RTX 4060 Laptop, 8 GiB) was used for every experiment in
the paper. Training one Diffusion Policy cell takes about 7.3 hours; evaluating 200
episodes takes about 20 minutes.

## Note on the patches

Four library behaviours are patched: deterministic convolution (`cudnn.benchmark`
disabled), per-batch RNG seeding, ACT with no robot-state input, and per-episode
`done_steps` recording. `patches/README.md` documents each change and how it was
verified. Two independent evaluations of the same checkpoint are bit-identical
with the patches applied.
