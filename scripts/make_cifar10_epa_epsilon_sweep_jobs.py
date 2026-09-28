#!/usr/bin/env python3
import argparse
import json
import os

def eps_tag(e):
    return str(e).replace(".", "p")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics_json", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--yt", type=int, default=0)
    ap.add_argument("--yp", type=int, default=1)
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    with open(args.metrics_json, "r") as f:
        rows = json.load(f)

    rows = sorted(rows, key=lambda r: float(r["epa"]))
    low = rows[:args.n]
    high = rows[-args.n:]

    high_eps = [0.01, 0.015, 0.02, 0.03, 0.05]
    low_eps = [0.01, 0.0075, 0.005, 0.0025, 0.001]

    jobs = []

    for group, selected, eps_list in [
        ("high", high, high_eps),
        ("low", low, low_eps),
    ]:
        for r in selected:
            tid = int(r["test_ds_index"])
            for eps in eps_list:
                jobs.append({
                    "yt": args.yt,
                    "yp": args.yp,
                    "target_test_idx": tid,
                    "poisonkey": f"{args.yt}-{args.yp}-{tid}",
                    "epa": float(r["epa"]),
                    "dps": float(r.get("dps", -1)),
                    "group": group,
                    "epsilon": eps,
                    "eps_tag": eps_tag(eps),
                    "method": "GM",
                    "recipe": "gradient-matching",
                })

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(jobs, f, indent=2)

    print(f"wrote {args.out_json}")
    print(f"num jobs = {len(jobs)}")

if __name__ == "__main__":
    main()