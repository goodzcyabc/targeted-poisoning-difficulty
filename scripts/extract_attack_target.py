#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--task_id", type=int, required=True)
    args = ap.parse_args()

    with open(args.json, "r", encoding="utf-8") as f:
        buckets = json.load(f)

    order = ["lowest", "p20", "p40", "p60", "p80", "highest"]

    rows = []
    for bucket in order:
        for r in buckets[bucket]:
            rr = dict(r)
            rr["bucket"] = bucket
            rows.append(rr)

    if args.task_id < 1 or args.task_id > len(rows):
        raise SystemExit(f"task_id={args.task_id} out of range 1..{len(rows)}")

    r = rows[args.task_id - 1]
    print(r["test_ds_index"])


if __name__ == "__main__":
    main()