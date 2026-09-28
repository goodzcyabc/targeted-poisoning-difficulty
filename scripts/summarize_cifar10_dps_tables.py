#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import json
import math
import os
from pathlib import Path


PB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DPS_RANKING_PATH = f"{PB_ROOT}/results/metrics/cifar10_dps_all/dps_rankings_all.json"


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


def write_latex_table(out_tex_path, title_text, out_rows, label):
    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lrr}")
    lines.append(r"\toprule")
    lines.append(r"Top-ranked prefix & \#Samples & Dominant $= y_t$ (\%) \\")
    lines.append(r"\midrule")

    for r in out_rows:
        pct = 100.0 * r["proportion"]
        lines.append(f'{r["prefix"]} & {r["num_samples"]} & {pct:.1f} \\\\')

    lines.append(r"\bottomrule")
    lines.append(
        rf"\caption{{{title_text}. Samples are sorted by DPS in descending order.}}"
    )
    lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def summarize_one(rows, title_text, stem, out_dir):
    rows = sorted(rows, key=lambda r: r["dps"], reverse=True)

    n = len(rows)
    cut_fracs = [0.2, 0.4, 0.6, 0.8, 1.0]

    out_rows = []

    print(f"\n===== {stem} =====")
    print(f"{'prefix':<12} {'#samples':>8} {'#dom=yt':>10} {'proportion':>12}")

    for frac in cut_fracs:
        k = max(1, int(math.ceil(frac * n)))
        prefix = rows[:k]

        num_dom_yt = sum(r["dominant_is_yt"] for r in prefix)
        prop = num_dom_yt / k

        label = f"Top {int(frac * 100)}%"

        print(f"{label:<12} {k:>8d} {num_dom_yt:>10d} {prop:>12.6f}")

        out_rows.append({
            "prefix": label,
            "num_samples": k,
            "num_dominant_eq_yt": num_dom_yt,
            "proportion": prop,
        })

    out_json = out_dir / f"{stem}_prefix_table.json"
    out_csv = out_dir / f"{stem}_prefix_table.csv"
    out_tex = out_dir / f"{stem}_prefix_table.tex"

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
                f'{r["proportion"]:.6f}',
            ])

    write_latex_table(
        out_tex_path=out_tex,
        title_text=title_text,
        out_rows=out_rows,
        label=f"tab:{stem}_prefix",
    )


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
        dominant_is_yt = to_int(r.get("dominant_is_yt"))

        if yt is None or dominant_is_yt is None or math.isnan(dps):
            continue

        rows.append({
            "yt": yt,
            "dps": dps,
            "dominant_is_yt": dominant_is_yt,
        })

    if len(rows) == 0:
        print("No usable rows found.")
        return

    out_dir = Path(os.path.dirname(DPS_RANKING_PATH)) / "dps_tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    # global table over all 10000 samples
    summarize_one(
        rows=rows,
        title_text="Relationship between DPS ranking and dominant-label consistency over all CIFAR-10 test samples",
        stem="all_10000",
        out_dir=out_dir,
    )

    # one table per class
    for yt in range(10):
        cls_rows = [r for r in rows if r["yt"] == yt]
        summarize_one(
            rows=cls_rows,
            title_text=f"Relationship between DPS ranking and dominant-label consistency for CIFAR-10 class $y_t={yt}$",
            stem=f"class_{yt}",
            out_dir=out_dir,
        )

    print("\n[out] wrote all tables to:")
    print(out_dir)


if __name__ == "__main__":
    main()