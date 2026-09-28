#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import math
import os


PB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DPS_RANKING_PATH = f"{PB_ROOT}/results/metrics/cifar10_dps_y0_p1/dps_rankings.json"


def read_csv_auto(path):
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
    delim = "\t" if "\t" in sample else ","
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter=delim))


def read_rows(path):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    elif ext in [".csv", ".tsv"]:
        return read_csv_auto(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def to_int(x):
    try:
        return int(x)
    except Exception:
        return None


def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def write_latex_table(out_tex_path, out_rows):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"Top-ranked prefix & \#Samples & Dominant $= y_t$ (\%) \\")
    lines.append(r"\midrule")

    for r in out_rows:
        pct = 100.0 * r["proportion"]
        lines.append(
            f'{r["prefix"]} & {r["num_samples"]} & {pct:.1f} \\\\'
        )

    lines.append(r"\bottomrule")
    lines.append(
        r"\caption{Relationship between DPS ranking and dominant-label consistency on CIFAR-10 "
        r"for $(y_t, y_p) = (0, 1)$. Samples are sorted by DPS in descending order. "
        r'For each top-ranked prefix, we report the percentage of samples whose dominant label '
        r"(excluding $y_p$) equals the true class $y_t$.}"
    )
    lines.append(r"\label{tab:cifar10_dps_prefix}")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    if not os.path.isfile(DPS_RANKING_PATH):
        print("Ranking file not found:")
        print(DPS_RANKING_PATH)
        return

    raw_rows = read_rows(DPS_RANKING_PATH)

    rows = []
    for r in raw_rows:
        yt = to_int(r.get("yt"))
        dps = to_float(r.get("dps"))
        mode_label = to_int(r.get("mode_label_excluding_yp"))

        if yt is None or mode_label is None or math.isnan(dps):
            continue

        rows.append({
            "yt": yt,
            "dps": dps,
            "mode_label_excluding_yp": mode_label,
            "dominant_is_yt": int(mode_label == yt),
        })

    if len(rows) == 0:
        print("No usable rows found.")
        print("Need fields: yt, dps, mode_label_excluding_yp")
        return

    rows.sort(key=lambda r: r["dps"], reverse=True)

    n = len(rows)
    cut_fracs = [0.2, 0.4, 0.6, 0.8, 1.0]

    print("\n===== DPS PREFIX TABLE =====\n")
    print(f"{'prefix':<12} {'#samples':>8} {'#dom=yt':>10} {'proportion':>12}")

    out_rows = []

    for frac in cut_fracs:
        k = max(1, int(math.ceil(frac * n)))
        prefix = rows[:k]

        num_dom_yt = sum(r["dominant_is_yt"] for r in prefix)
        prop = num_dom_yt / k

        label = f"Top {int(frac*100)}%"

        print(f"{label:<12} {k:>8d} {num_dom_yt:>10d} {prop:>12.6f}")

        out_rows.append({
            "prefix": label,
            "num_samples": k,
            "num_dominant_eq_yt": num_dom_yt,
            "proportion": prop,
        })

    out_dir = os.path.dirname(DPS_RANKING_PATH)
    out_json = os.path.join(out_dir, "dps_prefix_table.json")
    out_csv = os.path.join(out_dir, "dps_prefix_table.csv")
    out_tex = os.path.join(out_dir, "dps_prefix_table.tex")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out_rows, f, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["prefix", "num_samples", "num_dominant_eq_yt", "proportion"])
        for r in out_rows:
            w.writerow([
                r["prefix"],
                r["num_samples"],
                r["num_dominant_eq_yt"],
                f'{r["proportion"]:.6f}'
            ])

    write_latex_table(out_tex, out_rows)

    print("\n[out] wrote:")
    print(out_json)
    print(out_csv)
    print(out_tex)


if __name__ == "__main__":
    main()