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

| Setting | Single seed | Three-seed mean |
|---|---|---|
| Diffusion Policy, library defaults | 33.0 % | **37.2 %** |
| GroupNorm and random backbone only | 28.5 % | **25.3 %** |
| + `crop_shape=[84,84]` only | 55.0 % | **59.5 %** |
| + GroupNorm and random backbone | 63.5 % | in progress |
| published reference (500 episodes) | — | 65.4 % |

The three-seed means come from `train_seed_block.sh` / `eval_seed_block.sh`
(120k steps per run, 200 episodes per evaluation). The manuscript currently
reports the single-seed values and will be updated to the seed means once the
remaining two runs finish. Both are 200-episode paired evaluations on the same
initial states.

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
