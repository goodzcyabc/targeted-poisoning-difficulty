#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_json", required=True)
    ap.add_argument("--yt", type=int, default=0)
    ap.add_argument("--yp", type=int, default=1)
    ap.add_argument("--sample_size", type=int, default=200)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    with open(args.metrics_json, "r", encoding="utf-8") as f:
        rows = json.load(f)

    rng = random.Random(args.seed)
    if args.sample_size > len(rows):
        raise ValueError(f"sample_size={args.sample_size} > num_rows={len(rows)}")

    sampled = rng.sample(rows, args.sample_size)

    jobs = []
    for r in sampled:
        tid = int(r["test_ds_index"])
        for method, recipe in [("GM", "gradient-matching"), ("BP", "bullseye")]:
            jobs.append({
                "yt": int(args.yt),
                "yp": int(args.yp),
                "target_test_idx": tid,
                "method": method,
                "recipe": recipe,
                "poisonkey": f"{args.yt}-{args.yp}-{tid}",
            })

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)

    print(f"[out] wrote {args.out_json}")
    print(f"[out] num_sampled_targets={args.sample_size}")
    print(f"[out] num_jobs={len(jobs)}")


if __name__ == "__main__":
    main()