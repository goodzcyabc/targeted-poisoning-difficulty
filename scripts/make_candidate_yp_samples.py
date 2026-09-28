#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple


def load_wnids(data_path: str) -> List[str]:
    wnids_path = os.path.join(data_path, "wnids.txt")
    if not os.path.isfile(wnids_path):
        raise FileNotFoundError(f"Missing wnids.txt at {wnids_path}")
    with open(wnids_path, "r", encoding="utf-8") as f:
        wnids = [line.strip() for line in f if line.strip()]
    if len(wnids) != 200:
        raise RuntimeError(f"Expected 200 wnids, got {len(wnids)}")
    return wnids


def load_val_samples(data_path: str) -> List[Tuple[int, str, int]]:
    """
    Returns a list of:
      (val_ds_index, rel_path, label)
    using TinyImageNet val_annotations.txt
    """
    wnids = load_wnids(data_path)
    class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}

    val_root = os.path.join(data_path, "val")
    ann_path = os.path.join(val_root, "val_annotations.txt")
    img_root = os.path.join(val_root, "images")

    if not os.path.isfile(ann_path):
        raise FileNotFoundError(f"Missing val_annotations.txt at {ann_path}")
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"Missing val/images at {img_root}")

    samples: List[Tuple[int, str, int]] = []

    with open(ann_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            img_name, wnid = parts[0], parts[1]
            if wnid not in class_to_idx:
                continue
            abs_path = os.path.join(img_root, img_name)
            rel_path = os.path.relpath(abs_path, data_path)
            label = class_to_idx[wnid]
            samples.append((idx, rel_path, label))

    if len(samples) != 10000:
        print(f"[WARN] expected 10000 val samples, got {len(samples)}")

    return samples


def choose_unique_yp_candidates(
    all_val_samples: List[Tuple[int, str, int]],
    target_idx: int,
    yt: int,
    num_candidates: int,
) -> List[Tuple[int, str, int]]:
    """
    Pick exactly num_candidates OTHER val samples such that:
      - sample != target itself
      - sample label != yt
      - labels (yp) are unique within this target

    We first build one representative sample per unique yp, then sample yp classes.
    """
    # Map yp -> list of candidate samples of that class
    by_label: Dict[int, List[Tuple[int, str, int]]] = {}

    for cand_idx, cand_rel_path, cand_label in all_val_samples:
        if cand_idx == target_idx:
            continue
        if cand_label == yt:
            continue
        by_label.setdefault(cand_label, []).append((cand_idx, cand_rel_path, cand_label))

    unique_yp_labels = sorted(by_label.keys())
    if len(unique_yp_labels) < num_candidates:
        raise RuntimeError(
            f"Not enough unique yp labels available for target_idx={target_idx}, yt={yt}. "
            f"Need {num_candidates}, got only {len(unique_yp_labels)} unique labels."
        )

    chosen_labels = random.sample(unique_yp_labels, num_candidates)

    # For each chosen yp, randomly choose one sample from that class
    chosen_samples: List[Tuple[int, str, int]] = []
    for yp in chosen_labels:
        chosen_samples.append(random.choice(by_label[yp]))

    return chosen_samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True,
                    help="TinyImageNet root")
    ap.add_argument("--selected_json", required=True,
                    help="Path to epa_selected_yt0_midrange.json")
    ap.add_argument("--out_tsv", required=True,
                    help="Output TSV path")
    ap.add_argument("--num_candidates", type=int, default=50,
                    help="How many unique yp candidates to draw per target")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)

    with open(args.selected_json, "r", encoding="utf-8") as f:
        selected = json.load(f)

    # Expect these exact keys from your current midrange JSON
    target_keys = ["bottom1", "middle1_midrange", "top1"]
    for k in target_keys:
        if k not in selected:
            raise KeyError(f"Missing key '{k}' in {args.selected_json}")

    val_samples = load_val_samples(args.data_path)

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows: List[List[str]] = []

    for tag in target_keys:
        target = selected[tag]
        yt = int(target["yt"])
        target_idx = int(target["val_ds_index"])
        target_rel_path = str(target["rel_path"])
        target_epa = float(target["epa"])

        chosen = choose_unique_yp_candidates(
            all_val_samples=val_samples,
            target_idx=target_idx,
            yt=yt,
            num_candidates=args.num_candidates,
        )

        # Sanity: no duplicate yp for this target
        yps = [cand_label for _, _, cand_label in chosen]
        assert len(set(yps)) == len(yps), f"Duplicate yp detected for target {tag}"

        for cand_idx, cand_rel_path, cand_label in chosen:
            rows.append([
                tag,                    # bottom1 / middle1_midrange / top1
                str(target_idx),        # target_val_ds_index
                str(yt),                # yt
                target_rel_path,        # target_rel_path
                f"{target_epa:.10f}",   # target EPA
                str(cand_idx),          # candidate_val_ds_index
                str(cand_label),        # yp
                cand_rel_path,          # candidate_rel_path
            ])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "target_tag",
            "target_val_ds_index",
            "yt",
            "target_rel_path",
            "target_epa",
            "candidate_val_ds_index",
            "yp",
            "candidate_rel_path",
        ])
        w.writerows(rows)

    print(f"[out] wrote {out_path}")
    print(f"[info] total rows = {len(rows)}")
    print(f"[info] expected rows = {len(target_keys) * args.num_candidates}")

    # extra visibility: show yp uniqueness per target
    for tag in target_keys:
        tag_rows = [r for r in rows if r[0] == tag]
        tag_yps = [int(r[6]) for r in tag_rows]
        print(f"[check] {tag}: {len(tag_yps)} rows, {len(set(tag_yps))} unique yp")


if __name__ == "__main__":
    main()