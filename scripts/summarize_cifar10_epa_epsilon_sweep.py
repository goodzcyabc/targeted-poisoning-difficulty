#!/usr/bin/env python3
import csv
import glob
import json
import math
import os
from collections import defaultdict

PB_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JOBS_JSON = f"{PB_ROOT}/results/metrics/cifar10_metrics_y0/epsilon_sweep_jobs_n10.json"
RESULT_ROOT = f"{PB_ROOT}/results/attacks/cifar10_epa_epsilon_sweep/GM"
OUT_DIR = f"{PB_ROOT}/results/metrics/cifar10_metrics_y0"

def read_csv_auto(path):
    with open(path, "r") as f:
        sample = f.read(4096)
    delim = "\t" if "\t" in sample else ","
    with open(path, "r") as f:
        return list(csv.DictReader(f, delimiter=delim))

def to_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")

def std(xs):
    xs = [x for x in xs if not math.isnan(x)]
    if len(xs) <= 1:
        return 0.0
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))

def main():
    jobs = json.load(open(JOBS_JSON))
    expected = {(j["group"], j["eps_tag"], j["poisonkey"]): j for j in jobs}

    merged = []
    for j in jobs:
        pattern = os.path.join(
            RESULT_ROOT, j["group"], f"eps_{j['eps_tag']}", "*.csv"
        )
        last = None
        for path in sorted(glob.glob(pattern)):
            for r in read_csv_auto(path):
                if r.get("poisonkey") == j["poisonkey"]:
                    last = r
        if last is None:
            continue
        merged.append({
            **j,
            "ASR": to_float(last.get("target_acc_reinit", "nan")),
        })

    grouped = defaultdict(list)
    for r in merged:
        grouped[(r["group"], r["epsilon"])].append(r["ASR"])

    rows = []
    for group in ["high", "low"]:
        eps_list = sorted({r["epsilon"] for r in jobs if r["group"] == group})
        if group == "high":
            eps_list = sorted(eps_list, reverse=True)
        else:
            eps_list = sorted(eps_list)

        for eps in eps_list:
            xs = grouped[(group, eps)]
            rows.append({
                "group": group,
                "epsilon": eps,
                "num": len(xs),
                "mean_asr": mean(xs),
                "std_asr": std(xs),
            })

    os.makedirs(OUT_DIR, exist_ok=True)
    out_csv = os.path.join(OUT_DIR, "epsilon_sweep_n10_gm.csv")
    out_json = os.path.join(OUT_DIR, "epsilon_sweep_n10_gm.json")
    out_tex = os.path.join(OUT_DIR, "epsilon_sweep_n10_gm.tex")

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "epsilon", "num", "mean_asr", "std_asr"])
        for r in rows:
            w.writerow([r["group"], r["epsilon"], r["num"], f'{r["mean_asr"]:.6f}', f'{r["std_asr"]:.6f}'])

    with open(out_json, "w") as f:
        json.dump(rows, f, indent=2)

    lines = []
    lines.append(r"\begin{table}[t]")
    lines.append(r"\centering")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\toprule")
    lines.append(r"Group & $\epsilon$ & \#Targets & GM ASR \\")
    lines.append(r"\midrule")
    for r in rows:
        lines.append(
            f'{r["group"]} EPA & {r["epsilon"]:.4f} & {r["num"]} & '
            f'{r["mean_asr"]:.3f} $\\pm$ {r["std_asr"]:.3f} \\\\'
        )
    lines.append(r"\bottomrule")
    lines.append(r"\caption{GM attack success under varying poison budget on CIFAR-10. High-EPA targets are evaluated as $\epsilon$ decreases, while low-EPA targets are evaluated as $\epsilon$ increases.}")
    lines.append(r"\label{tab:cifar10_epa_epsilon_sweep}")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")

    with open(out_tex, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("matched:", len(merged), "/", len(jobs))
    print(out_csv)
    print(out_json)
    print(out_tex)

if __name__ == "__main__":
    main()