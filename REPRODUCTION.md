# Reproduction

Environment: Python 3.10, PyTorch + torchvision built for your GPU/CUDA, numpy.
Verified with torch 2.5.1+cu121 / torchvision 0.20.1. CIFAR-10 downloads automatically.
TinyImageNet experiments expect the dataset at `./data/tiny-imagenet-200`.

Model checkpoints are excluded from this release (`*.pth`, ~114 MB each). The TinyImageNet metric scripts
take `--ckpt_path`, which should point at a VGG16 with a 200-way head trained on TinyImageNet-200.

The metric/ranking steps need a GPU. The `summarize_*.py` scripts use only the Python standard library and
run on CPU from the repository root, reading what is already in `results/`.

## What ships, and what you have to run

`results/metrics/` holds the metric/ranking outputs and `results/attacks/` the attack outcome tables, for
the settings reported in the paper. Every `summarize_*.py` script in `scripts/` has its inputs present, so
the seven of them reproduce their tables offline with no GPU.

Two `*_rank_*` scripts ship without matching attack results, because those attack runs were never completed:
`epa_rank_cifar10_cnn3_yt0.py` (its metric output *is* shipped, in `results/metrics/cifar10_epa_cnn3_y0/`)
and `epa_rank_multi_class.py`. They run, but you have to generate the attack outcomes yourself; there is no
summarizer for them. Likewise `results/attacks/delta_yp1/` ships the eight raw GM tables with no summarizer
script — those numbers were read off the tables directly.

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
python scripts/summarize_cifar10_epa_gm.py              # EPA vs GM ASR
python scripts/summarize_cifar10_epa_epsilon_sweep.py   # effect of the poison budget epsilon
```

Other settings: `epa_rank_cifar10_vgg13_yt0.py` and `epa_rank_tinyimagenet_yt0.py`, summarized by
`summarize_epa_cifar10_vgg13_table.py` and `summarize_epa_tinyimagenet_table.py`.

## Poisoning distance δ and budget τ  (fine-grained metrics)

`metrics/delta_tau.py` implements the binary search of Algorithm "Poisoning Distance Estimation"
(tolerance `alpha = 1e-4`) and the budget bound τ. Both scripts below are TinyImageNet/VGG16 and were run as:

```bash
DATA=./data/tiny-imagenet-200
CKPT=/path/to/VGG16_Tinyimagenet_all.pth

# rank every target of class yt by delta, for a fixed poison class yp
python scripts/delta_rank_fixed_yp.py \
    --data_path "$DATA" --class_order_file "$DATA/words200.txt" \
    --out_dir results/metrics/delta_yt0_yp1 --yt 0 --yp 1 \
    --model vgg16 --num_classes 200 --ckpt_path "$CKPT" --topk 4

# delta and tau for a single candidate (one row of the candidate table)
python scripts/compute_delta_tau_one_candidate.py \
    --data_path "$DATA" \
    --candidate_tsv results/metrics/epa_yt0/candidate_yp_samples_midrange.tsv \
    --row_idx 1 --out_dir results/metrics/dtau_row1 \
    --model vgg16 --num_classes 200 --ckpt_path "$CKPT"

python scripts/summarize_cifar10_corr_table.py    # correlation of the metrics with ASR
```

`delta_rank_fixed_yp.py` writes `delta_rankings_yt{yt}_yp{yp}.csv` plus the `delta_top{k}` / `delta_bottom{k}`
JSON files naming the selected targets; `compute_delta_tau_one_candidate.py` writes one `row_NNN.tsv` per
candidate, so it is meant to be run as an array over `--row_idx`.

Three details worth knowing:

- `num_classes` defaults to 200 (TinyImageNet) and must be set to 10 for CIFAR-10.
- `g(D_c)` is estimated from 50 batches of 256 (`--tau_num_batches`, `--tau_batch_size`) rather than the
  full clean set.
- The candidate set and the sort are deterministic — all 50 TinyImageNet val images of class `yt`, ordered
  by a stable sort. The δ *values*, however, are only resolved to the `alpha = 1e-4` tolerance of the binary
  search, and near-tied candidates reorder across GPU types. Re-running the command above reproduced 5 of
  the 8 targets in `results/attacks/delta_yp1/`; the ones that swapped differ in δ by less than 1e-3, and
  several bottom-group δ values are exactly tied. The top/bottom *groups* stay well separated
  (δ ≈ 0.0051–0.0093 vs ≈ 0.0023–0.0024), which is what the correlation results rely on; membership right at
  the cut is not bit-reproducible.

## Attack evaluation

Attack outcome tables under `results/attacks/` were produced with the public **gradient matching**
("Witches' Brew") framework, which is cited in the paper and is *not* redistributed here; each row's ASR is
the `target_acc_reinit` column. Because that framework is external, these tables cannot be regenerated from
this repository alone — they are shipped as the record of the runs behind the paper. The `benchmark_idx`
column indexes into the benchmark setup lists at
`third_party/poisoning-benchmark/poison_setups/*.pickle`.
