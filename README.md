# What a Success Rate Measures — artifact

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24034427.svg)](https://doi.org/10.5281/zenodo.24034427)

Code, raw results and source patches for the paper *"What a Success Rate Measures:
Configuration, Compute and Sampling in a Robot Learning Benchmark"*.

## What is here

| Path | Contents |
|---|---|
| `paper/` | manuscript source, compiled PDF (10 pages) and figures |
| `results/eval/` | 65 `eval_info.json` files — per-episode `successes`, `max_rewards`, `done_steps` for every evaluation reported in the paper |
| `patches/` | 15 source changes against LeRobot `7e241bd6`, without which the numbers are not reproducible |
| `scripts/` | training, evaluation and statistics scripts |
| `训练结果汇总.md` | the consolidated result tables |
| `噪声地板表.md` | the evaluation noise-floor analysis |

## Headline numbers

Every cell is averaged over three training seeds (seeds 100k / 300k / 400k), each
evaluated on 200 episodes under the same protocol.

| Setting | Success (mean ± seed sd) |
|---|---|
| Diffusion Policy, library defaults | **37.2 ± 3.7 %** |
| GroupNorm and random backbone only | **25.3 ± 2.8 %** |
| + `crop_shape=[84,84]` only | **59.5 ± 5.4 %** |
| + GroupNorm and random backbone | **63.2 ± 6.5 %** |
| published reference (500 episodes) | 65.4 % |

Seed-to-seed spread is 2.8–6.5 pp per cell (pooled 4.8 pp), comparable to the
3.1–3.5 pp binomial standard error of a 200-episode evaluation and an order of
magnitude below the effects. The two configuration fields are not independent:
`crop_shape` is worth 22.3 pp without GroupNorm and 37.8 pp with it, while
GroupNorm with a random backbone costs 11.8 pp alone and gains 3.7 pp once
cropping is on (interaction +15.5 pp, 2.8 seed standard errors).

Reproduce with `train_seed_block.sh` (12 runs, 120k steps each) and
`eval_seed_block.sh` (200 episodes per checkpoint).

## Citation

```bibtex
@software{huang_pusht_repro_2026,
  author    = {Huang, Yandong},
  title     = {PushT reproduction artifact: configuration, compute and sampling
               in a robot learning benchmark},
  year      = {2026},
  publisher = {Zenodo},
  version   = {v1.0.0},
  doi       = {10.5281/zenodo.24034427},
  url       = {https://doi.org/10.5281/zenodo.24034427}
}
```

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
