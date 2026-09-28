#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import glob
import json
import math
from pathlib import Path


TOP_JSON = "results/metrics/dps_yt0_yp1/dps_top6_yt0_yp1.json"
BOTTOM_JSON = "results/metrics/dps_yt0_yp1/dps_bottom6_yt0_yp1.json"
GM_GLOB = "results/attacks/dps_yp1/GM/*.csv"


def read_csv_auto(path):
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(2048)
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


def main():
    with open(BOTTOM_JSON, "r", encoding="utf-8") as f:
        bottom = json.load(f)
    with open(TOP_JSON, "r", encoding="utf-8") as f:
        top = json.load(f)

    pair_map = {}

    for r in bottom:
        poisonkey = f'{r["yt"]}-{r["yp"]}-{r["val_ds_index"]}'
        pair_map[poisonkey] = {
            "group": "bottom6",
            "dps": float(r["dps"]),
            "target_id": int(r["val_ds_index"]),
            "yt": int(r["yt"]),
            "yp": int(r["yp"]),
        }

    for r in top:
        poisonkey = f'{r["yt"]}-{r["yp"]}-{r["val_ds_index"]}'
        pair_map[poisonkey] = {
            "group": "top6",
            "dps": float(r["dps"]),
            "target_id": int(r["val_ds_index"]),
            "yt": int(r["yt"]),
            "yp": int(r["yp"]),
        }

    merged = []

    for path in glob.glob(GM_GLOB):
        rows = read_csv_auto(path)
        if not rows:
            continue
        r = rows[0]
        poisonkey = r.get("poisonkey", "")
        if poisonkey not in pair_map:
            continue

        # 你们当前口径里，target_acc_reinit 就是主结果列
        asr = to_float(r.get("target_acc_reinit", "nan"))

        out = dict(pair_map[poisonkey])
        out["poisonkey"] = poisonkey
        out["ASR"] = asr
        merged.append(out)

    print("\n===== BASIC =====")
    print("merged experiments:", len(merged))

    top_asr = [r["ASR"] for r in merged if r["group"] == "top6"]
    bottom_asr = [r["ASR"] for r in merged if r["group"] == "bottom6"]

    print("\n===== GROUP MEANS =====")
    print("top6 mean ASR   :", mean(top_asr))
    print("bottom6 mean ASR:", mean(bottom_asr))

    xpas = [r["dps"] for r in merged]
    asrs = [r["ASR"] for r in merged]

    print("\n===== CORRELATION =====")
    print("corr(DPS, ASR):", pearson_corr(xpas, asrs))

    print("\n===== DETAILS =====")
    merged_sorted = sorted(merged, key=lambda r: r["ASR"], reverse=True)
    for r in merged_sorted:
        print(
            f'group={r["group"]:7s}  '
            f'dps={r["dps"]:.6f}  '
            f'ASR={r["ASR"]:.6f}  '
            f'poisonkey={r["poisonkey"]}'
        )


if __name__ == "__main__":
    main()