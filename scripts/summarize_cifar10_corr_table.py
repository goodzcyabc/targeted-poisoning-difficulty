#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import glob
import json
import math
import os


PB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

METRICS_JSON = f"{PB_ROOT}/results/metrics/cifar10_metrics_y0/metrics_y0.json"
GM_GLOB = f"{PB_ROOT}/results/attacks/cifar10_corr/GM/*.csv"
BP_GLOB = f"{PB_ROOT}/results/attacks/cifar10_corr/BP/*.csv"
OUT_DIR = f"{PB_ROOT}/results/metrics/cifar10_metrics_y0"


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


def collect_last_rows(glob_pat):
    last = {}
    for path in sorted(glob.glob(glob_pat)):
        rows = read_csv_auto(path)
        for r in rows:
            poisonkey = r.get("poisonkey", "")
            if poisonkey:
                last[poisonkey] = r
    return last


def write_tex(table_rows, out_path):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lcc}")
    lines.append(r"\toprule")
    lines.append(r"Attack & Correlation Coefficients (ASR, EPA) & Cor(ASR, DPS) \\")
    lines.append(r"\midrule")
    for r in table_rows:
        lines.append(f'{r["attack"]} & {r["corr_asr_epa"]:.3f} & {r["corr_asr_dps"]:.3f} \\\\')
    lines.append(r"\bottomrule")
    lines.append(
        r"\caption{Correlation coefficients between attack success rate (ASR) and target stability metrics on CIFAR-10 for $y_t=0$ and $y_p=1$.}"
    )
    lines.append(r"\label{tab:cifar10_corr_epa_dps}")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    metric_map = {}
    for r in metrics:
        poisonkey = f'0-1-{int(r["test_ds_index"])}'
        metric_map[poisonkey] = {
            "epa": float(r["epa"]),
            "dps": float(r["dps"]),
        }

    gm_last = collect_last_rows(GM_GLOB)
    bp_last = collect_last_rows(BP_GLOB)

    out_rows = []

    for attack_name, source in [("GM", gm_last), ("BP", bp_last)]:
        xs_epa, xs_dps, ys_asr = [], [], []

        for poisonkey, row in source.items():
            if poisonkey not in metric_map:
                continue
            asr = to_float(row.get("target_acc_reinit", "nan"))
            xs_epa.append(metric_map[poisonkey]["epa"])
            xs_dps.append(metric_map[poisonkey]["dps"])
            ys_asr.append(asr)

        out_rows.append({
            "attack": attack_name,
            "corr_asr_epa": pearson_corr(ys_asr, xs_epa),
            "corr_asr_dps": pearson_corr(ys_asr, xs_dps),
            "num_points": len(ys_asr),
        })

    out_json = os.path.join(OUT_DIR, "corr_table.json")
    out_csv = os.path.join(OUT_DIR, "corr_table.csv")
    out_tex = os.path.join(OUT_DIR, "corr_table.tex")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["attack", "corr_asr_epa", "corr_asr_dps", "num_points"])
        for r in out_rows:
            w.writerow([r["attack"], f'{r["corr_asr_epa"]:.6f}', f'{r["corr_asr_dps"]:.6f}', r["num_points"]])

    write_tex(out_rows, out_tex)

    print("[out] wrote:")
    print(out_json)
    print(out_csv)
    print(out_tex)


if __name__ == "__main__":
    main()