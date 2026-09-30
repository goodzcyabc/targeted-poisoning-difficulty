# Are Targeted Data Poisoning Attacks as Effective as We Think?

Anonymous code release for our SatML 2027 submission.

Existing evaluations of **targeted data poisoning** report the *average* attack success rate (ASR) over
randomly chosen targets, which hides how much difficulty varies from one target to the next. This repository
implements the four metrics we use to predict, from **clean-model information only**, how hard a given test
sample is to poison:

| Metric | Meaning | Where |
|---|---|---|
| **EPA** — ergodic prediction accuracy | coarse metric from clean **training dynamics** | `scripts/epa_rank_*.py` |
| **DPS** — dominant prediction score | cheaper **surrogate for EPA** | `scripts/dps_rank_*.py` |
| **δ** — poisoning distance | fine-grained, per poison class | `metrics/delta_tau.py` |
| **τ** — poison budget | fine-grained, per poison class | `metrics/delta_tau.py` |

Attacks evaluated: **GM** (gradient matching), **BP** (Bullseye Polytope), **FC** (feature collision), on
**CIFAR-10** and **TinyImageNet**, training from scratch and transfer learning.

## Layout

```
metrics/      δ and τ implementation (Alg. "Poisoning Distance Estimation")
scripts/      entrypoints: rank targets by a metric, build attack jobs, summarize into tables
results/      ALL experimental results
  ├── metrics/    per-target metric values and rankings (EPA, DPS, δ, τ)
  └── attacks/    attack outcome tables (ASR) per dataset / model / setting
third_party/  the public poisoning benchmark we build on, redistributed under its own license
```

Everything outside `third_party/` is ours. Nothing in `metrics/` or `scripts/` imports anything under
`third_party/`; it is kept because the `benchmark_idx` column in our attack tables is defined against its
setup lists. See `NOTICE.md`.

## Quickstart

```bash
pip install -r requirements.txt

# rank CIFAR-10 test samples by EPA (ResNet-18); CIFAR-10 downloads automatically
python scripts/epa_rank_cifar10_resnet18.py --out_dir results/metrics/my_epa --yt 0 \
    --epochs 40 --inits 8 --batch_size 128 --lr 0.1 --device cuda

# same for DPS
python scripts/dps_rank_cifar10_resnet18_all.py --out_dir results/metrics/my_dps \
    --epochs 40 --inits 8 --batch_size 128 --lr 0.1 --device cuda

# smoke test (~30 s on one GPU)
python scripts/epa_rank_cifar10_resnet18.py --out_dir /tmp/smoke --yt 0 --epochs 1 --inits 1 --device cuda
```

See **`REPRODUCTION.md`** for the command behind each table in the paper.

## License

Our code is MIT-licensed — see `LICENSE`. The redistributed benchmark under `third_party/` keeps its own
license at `third_party/poisoning-benchmark/LICENSE`. See `NOTICE.md` for what came from where.
