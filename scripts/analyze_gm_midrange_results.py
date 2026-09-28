import pandas as pd
from pathlib import Path

pairs = pd.read_csv(
    "results/metrics/epa_yt0/pairs48_midrange_dedup.tsv",
    sep="\t"
)

rows = []

for f in Path("results/attacks/midrange/GM").glob("*.csv"):
    df = pd.read_csv(f, sep="\t")

    if len(df) == 0:
        continue

    r = df.iloc[0]

    rows.append({
        "poisonkey": r["poisonkey"],
        "ASR": 1 - r["target_acc_reinit"]   # convert acc → ASR
    })

gm = pd.DataFrame(rows)

pairs["poisonkey"] = (
    pairs["yt"].astype(str)
    + "-"
    + pairs["yp"].astype(str)
    + "-"
    + pairs["target_val_ds_index"].astype(str)
)

merged = pairs.merge(gm, on="poisonkey")

print("\n===== BASIC STATS =====\n")

print("num experiments:", len(merged))
print()

print("delta correlation with ASR:")
print(merged["delta"].corr(merged["ASR"]))

print("\ntau correlation with ASR:")
print(merged["tau"].corr(merged["ASR"]))

print()

print("===== GROUP MEANS =====")

print("\nby selection_tag:")
print(
    merged.groupby("selection_tag")["ASR"].mean().sort_values()
)

print("\nby target_tag:")
print(
    merged.groupby("target_tag")["ASR"].mean().sort_values()
)

print("\n===== TOP ASR =====")

print(
    merged.sort_values("ASR", ascending=False)[
        ["selection_tag","delta","tau","ASR"]
    ].head(10)
)

print("\n===== LOWEST ASR =====")

print(
    merged.sort_values("ASR")[
        ["selection_tag","delta","tau","ASR"]
    ].head(10)
)