#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def make_poisonkey(yt: int, yp: int, target_val_ds_index: int) -> str:
    return f"{yt}-{yp}-{target_val_ds_index}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dedup_tsv", required=True,
                    help="Input dedup pair file, e.g. pairs48_midrange_dedup.tsv")
    ap.add_argument("--out_tsv", required=True,
                    help="Output poison job TSV")
    ap.add_argument("--budget", type=float, default=0.0005,
                    help="Poison budget, default 0.0005")
    args = ap.parse_args()

    in_path = Path(args.dedup_tsv)
    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows_out = []

    with open(in_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        for row in r:
            target_tag = row["target_tag"]
            selection_tag = row["selection_tag"]
            target_val_idx = int(row["target_val_ds_index"])
            yt = int(row["yt"])
            yp = int(row["yp"])

            poisonkey = make_poisonkey(yt, yp, target_val_idx)

            # group name keeps source info for later debugging/analysis
            group = f"{target_tag}__{selection_tag}__yp{yp}"

            rows_out.append([
                "GM",
                group,
                str(yt),
                str(yp),
                str(target_val_idx),
                poisonkey,
                f"{args.budget:g}",
            ])

            rows_out.append([
                "BP",
                group,
                str(yt),
                str(yp),
                str(target_val_idx),
                poisonkey,
                f"{args.budget:g}",
            ])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "method",
            "group",
            "yt",
            "yp",
            "target_val_ds_index",
            "poisonkey",
            "budget",
        ])
        w.writerows(rows_out)

    print(f"[out] wrote {out_path}")
    print(f"[info] unique dedup pairs = {len(rows_out)//2}")
    print(f"[info] total poison jobs = {len(rows_out)}")


if __name__ == "__main__":
    main()