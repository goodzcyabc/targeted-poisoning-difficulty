# Reproduction

Environment: Python 3.10, PyTorch + torchvision built for your GPU/CUDA, numpy.
Verified with torch 2.5.1+cu121 / torchvision 0.20.1. CIFAR-10 downloads automatically.
TinyImageNet experiments expect the dataset at `./data/tiny-imagenet-200`.

The metric/ranking steps need a GPU. The `summarize_*.py` scripts use only the Python standard library and
run on CPU from the repository root, reading what is already in `results/`.

## DPS as a surrogate for EPA  (appendix table)

```bash
python scripts/dps_rank_cifar10_resnet18_all.py --out_dir results/metrics/cifar10_dps_all \
    --epochs 40 --inits 8 --batch_size 128 --eval_batch_size 256 --lr 0.1 --device cuda
python scripts/summarize_cifar10_dps_tables.py
```
Ranks all 10,000 CIFAR-10 test samples by DPS and reports, for each top-k% prefix, the fraction whose
dominant class equals the ground-truth label. Shipped output:
`results/metrics/cifar10_dps_all/dps_tables/`.

## EPA vs. attack success  (main experiments)

```bash
python scripts/epa_rank_cifar10_resnet18.py --out_dir results/metrics/cifar10_epa_y0 --yt 0 \
    --epochs 40 --inits 8 --batch_size 128 --lr 0.1 --device cuda
python scripts/summarize_cifar10_epa_gm.py          # EPA vs GM ASR
python scripts/summarize_cifar10_epa_epsilon_sweep.py   # effect of the poison budget epsilon
```

Other settings: `epa_rank_cifar10_vgg13_yt0.py`, `epa_rank_cifar10_cnn3_yt0.py`,
`epa_rank_tinyimagenet_yt0.py`, `epa_rank_multi_class.py`, summarized by the matching
`summarize_epa_*` script.

## Poisoning distance δ and budget τ  (fine-grained metrics)

`metrics/delta_tau.py` implements the binary search of Algorithm "Poisoning Distance Estimation"
(tolerance `alpha = 1e-4`) and the budget bound τ.

```bash
python scripts/delta_rank_fixed_yp.py              # rank targets by delta for a fixed poison class
python scripts/compute_delta_tau_one_candidate.py  # delta and tau for one candidate
python scripts/summarize_cifar10_corr_table.py     # correlation of the metrics with ASR
```

Two details worth knowing: `num_classes` defaults to 200 (TinyImageNet) and must be set to 10 for CIFAR-10;
and `g(D_c)` is estimated from 50 batches of 256 rather than the full clean set.

## Attack evaluation

Attack outcome tables under `results/attacks/` were produced by running the poisons through the benchmark
harness (`poison_test.py`); each row's ASR is the `target_acc_reinit` column.
