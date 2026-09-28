#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import glob
import json
import math
import os
from collections import defaultdict


PB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

JOBS_JSON = f"{PB_ROOT}/results/metrics/cifar10_epa_vgg13_y0/attack_jobs.json"
GM_GLOB = f"{PB_ROOT}/results/attacks/cifar10_epa_vgg13/GM/*.csv"
BP_GLOB = f"{PB_ROOT}/results/attacks/cifar10_epa_vgg13/BP/*.csv"
OUT_DIR = f"{PB_ROOT}/results/metrics/cifar10_epa_vgg13_y0"


def read_csv_auto(path):
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
    delim = "\t" if "\t" in sample else ","
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delim))


def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if not xs:
        return float("nan")
    return sum(xs) / len(xs)


def std(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) <= 1:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def fmt_pm(xs):
    return f"{mean(xs):.3f} $\\pm$ {std(xs):.3f}"


def collect_last_rows(glob_pat):
    last = {}
    for path in sorted(glob.glob(glob_pat)):
        rows = read_csv_auto(path)
        for r in rows:
            poisonkey = r.get("poisonkey", "")
            if poisonkey:
                last[poisonkey] = r
    return last


def write_tex(table_map, out_path):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(r"$x_t$ & \multicolumn{2}{c}{$y_p=1$} & \multicolumn{2}{c}{$y_p=2$} \\")
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    lines.append(r" & GM & BP & GM & BP \\")
    lines.append(r"\midrule")

    for group in ["high", "middle", "low"]:
        label = {
            "high": "high EPA",
            "middle": "middle EPA",
            "low": "low EPA",
        }[group]

        gm1 = table_map[(group, 1, "GM")]
        bp1 = table_map[(group, 1, "BP")]
        gm2 = table_map[(group, 2, "GM")]
        bp2 = table_map[(group, 2, "BP")]

        lines.append(f"{label} & {gm1} & {bp1} & {gm2} & {bp2} \\\\")

    lines.append(r"\bottomrule")
    lines.append(
        r"\caption{Poisoning difficulty prediction using EPA on CIFAR-10 with a weaker backbone (VGG13). "
        r"We select 4 highest, 4 middle, and 4 lowest EPA targets from $y_t=0$, "
        r"and evaluate GM and BP attacks toward $y_p=1,2$ under attack budget $\epsilon=1\%$.}"
    )
    lines.append(r"\label{tab:cifar10_epa_vgg13}")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(JOBS_JSON, "r", encoding="utf-8") as f:
        jobs = json.load(f)

    expected = {}
    for j in jobs:
        expected[(j["poisonkey"], j["method"])] = j

    gm_last = collect_last_rows(GM_GLOB)
    bp_last = collect_last_rows(BP_GLOB)

    grouped = defaultdict(list)
    detailed = []

    for (poisonkey, method), meta in expected.items():
        row = gm_last.get(poisonkey) if method == "GM" else bp_last.get(poisonkey)
        if row is None:
            continue

        asr = to_float(row.get("target_acc_reinit", "nan"))

        key = (meta["group"], meta["yp"], meta["method"])
        grouped[key].append(asr)

        detailed.append({
            "group": meta["group"],
            "yp": meta["yp"],
            "method": meta["method"],
            "poisonkey": poisonkey,
            "epa": meta["epa"],
            "ASR": asr,
        })

    out_rows = []
    table_map = {}

    for group in ["high", "middle", "low"]:
        for yp in [1, 2]:
            for method in ["GM", "BP"]:
                xs = grouped[(group, yp, method)]
                s = fmt_pm(xs)
                table_map[(group, yp, method)] = s
                out_rows.append({
                    "group": group,
                    "yp": yp,
                    "method": method,
                    "num_samples": len(xs),
                    "mean_asr": mean(xs),
                    "std_asr": std(xs),
                    "display": s,
                })

    out_json = os.path.join(OUT_DIR, "cifar10_epa_vgg13_table.json")
    out_csv = os.path.join(OUT_DIR, "cifar10_epa_vgg13_table.csv")
    out_tex = os.path.join(OUT_DIR, "cifar10_epa_vgg13_table.tex")
    out_detail = os.path.join(OUT_DIR, "cifar10_epa_vgg13_detail.json")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "yp", "method", "num_samples", "mean_asr", "std_asr", "display"])
        for r in out_rows:
            w.writerow([
                r["group"], r["yp"], r["method"], r["num_samples"],
                f'{r["mean_asr"]:.6f}', f'{r["std_asr"]:.6f}', r["display"]
            ])

    with open(out_detail, "w", encoding="utf-8") as f:
        json.dump(detailed, f, indent=2)

    write_tex(table_map, out_tex)

    print("[out] wrote:")
    print(out_json)
    print(out_csv)
    print(out_tex)
    print(out_detail)


if __name__ == "__main__":
    main()