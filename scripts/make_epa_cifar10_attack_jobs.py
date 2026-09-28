#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epa_dir", required=True)
    ap.add_argument("--yt", type=int, default=0)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    lowest = load_json(os.path.join(args.epa_dir, f"epa_lowest4_yt{args.yt}.json"))
    middle = load_json(os.path.join(args.epa_dir, f"epa_middle4_yt{args.yt}.json"))
    highest = load_json(os.path.join(args.epa_dir, f"epa_highest4_yt{args.yt}.json"))

    jobs = []

    for group_name, rows in [("high", highest), ("middle", middle), ("low", lowest)]:
        for r in rows:
            for yp in [1, 2]:
                for method, recipe in [("GM", "gradient-matching"), ("BP", "bullseye")]:
                    jobs.append({
                        "yt": int(r["yt"]),
                        "yp": int(yp),
                        "target_test_idx": int(r["test_ds_index"]),
                        "epa": float(r["epa"]),
                        "group": group_name,
                        "method": method,
                        "recipe": recipe,
                        "poisonkey": f'{int(r["yt"])}-{int(yp)}-{int(r["test_ds_index"])}'
                    })

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(jobs, f, indent=2)

    print(f"[out] wrote {args.out_json}")
    print(f"[out] num_jobs={len(jobs)}")


if __name__ == "__main__":
    main()