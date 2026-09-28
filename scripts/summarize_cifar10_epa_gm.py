#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import math
import os
from collections import defaultdict


PB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ATTACK_TARGETS_JSON = f"{PB_ROOT}/results/metrics/cifar10_epa_y0/attack_targets.json"

# 改成你 CIFAR-10 那张总表的真实路径
TABLE_PATH = f"{PB_ROOT}/results/attacks/cifar10_epa_y0_yp1/GM/table_ResNet18_single-class.csv"

YT = 0
YP = 1


def read_csv_auto(path):
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
    delim = "\t" if "\t" in sample else ","
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delim))


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def pearson_corr(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if not (math.isnan(x) or math.isnan(y))]
    n = len(pairs)
    if n < 2:
        return float("nan")

    xs = [p[0] for p in pairs]
    ys = [p[1] for p in pairs]

    mx = sum(xs) / n
    my = sum(ys) / n

    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))

    if denx == 0 or deny == 0:
        return float("nan")

    return num / (denx * deny)


def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def load_expected():
    with open(ATTACK_TARGETS_JSON, "r", encoding="utf-8") as f:
        buckets = json.load(f)

    expected_map = {}
    order = ["lowest", "p20", "p40", "p60", "p80", "highest"]

    for bucket in order:
        for r in buckets[bucket]:
            poisonkey = f"{YT}-{YP}-{r['test_ds_index']}"
            expected_map[poisonkey] = {
                "bucket": bucket,
                "epa": float(r["epa"]),
                "test_ds_index": int(r["test_ds_index"]),
                "yt": int(r["yt"]),
            }

    return expected_map


def main():
    expected_map = load_expected()
    expected_keys = set(expected_map.keys())

    print("\n===== BASIC =====")
    print("table path:", TABLE_PATH)
    print("expected poisonkeys:", len(expected_keys))

    if not os.path.isfile(TABLE_PATH):
        print("Table file not found.")
        return

    rows = read_csv_auto(TABLE_PATH)

    # 如果同一个 poisonkey 出现多次，取最后一行
    matched_last = {}
    for r in rows:
        poisonkey = r.get("poisonkey", "")
        if poisonkey in expected_keys:
            matched_last[poisonkey] = r

    merged = []
    for poisonkey in sorted(matched_last.keys()):
        r = matched_last[poisonkey]
        asr = to_float(r.get("target_acc_reinit", "nan"))

        out = dict(expected_map[poisonkey])
        out["poisonkey"] = poisonkey
        out["ASR"] = asr
        merged.append(out)

    print("matched experiments:", len(merged))

    missing = sorted(expected_keys - set(matched_last.keys()))
    print("missing experiments:", len(missing))
    if missing:
        print("missing poisonkeys:")
        for k in missing:
            print(" ", k)

    if len(merged) == 0:
        print("No matched experiments found.")
        return

    bucket_to_asr = defaultdict(list)
    for r in merged:
        bucket_to_asr[r["bucket"]].append(r["ASR"])

    print("\n===== BUCKET MEANS =====")
    order = ["lowest", "p20", "p40", "p60", "p80", "highest"]
    for b in order:
        print(f"{b:8s} mean ASR: {mean(bucket_to_asr[b])}")

    epa_vals = [r["epa"] for r in merged]
    asr_vals = [r["ASR"] for r in merged]

    print("\n===== CORRELATION =====")
    print("corr(EPA, ASR):", pearson_corr(epa_vals, asr_vals))

    print("\n===== DETAILS =====")
    merged_sorted = sorted(merged, key=lambda r: r["epa"])
    for r in merged_sorted:
        print(
            f'bucket={r["bucket"]:8s}  '
            f'epa={r["epa"]:.6f}  '
            f'ASR={r["ASR"]:.6f}  '
            f'poisonkey={r["poisonkey"]}'
        )


if __name__ == "__main__":
    main()