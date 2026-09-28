#!/usr/bin/env python3

import argparse
import csv
from pathlib import Path
from collections import defaultdict


def read_all_rows(parts_dir):
    rows = []

    for f in sorted(Path(parts_dir).glob("row_*.tsv")):
        with open(f) as fp:
            r = csv.DictReader(fp, delimiter="\t")
            for row in r:
                row["delta"] = float(row["delta"]) if row["delta"] != "inf" else float("inf")
                row["tau"] = float(row["tau"]) if row["tau"] != "inf" else float("inf")
                row["yt"] = int(row["yt"])
                row["yp"] = int(row["yp"])
                row["target_val_ds_index"] = int(row["target_val_ds_index"])
                row["candidate_val_ds_index"] = int(row["candidate_val_ds_index"])
                row["source_file"] = f.name
                rows.append(row)

    return rows


def select_extremes(rows, k):
    by_target = defaultdict(list)

    for r in rows:
        by_target[r["target_tag"]].append(r)

    raw = []

    for tag, group in by_target.items():

        g_delta = sorted(group, key=lambda x: x["delta"])
        g_tau = sorted(group, key=lambda x: x["tau"])

        delta_bottom = g_delta[:k]
        delta_top = g_delta[-k:]
        tau_bottom = g_tau[:k]
        tau_top = g_tau[-k:]

        for r in delta_bottom:
            rr = dict(r)
            rr["selection_tag"] = "delta_bottom"
            raw.append(rr)

        for r in delta_top:
            rr = dict(r)
            rr["selection_tag"] = "delta_top"
            raw.append(rr)

        for r in tau_bottom:
            rr = dict(r)
            rr["selection_tag"] = "tau_bottom"
            raw.append(rr)

        for r in tau_top:
            rr = dict(r)
            rr["selection_tag"] = "tau_top"
            raw.append(rr)

    return raw


def deduplicate(raw_rows):
    dedup = {}

    for r in raw_rows:
        key = (r["target_tag"], r["yp"])

        if key not in dedup:
            dedup[key] = dict(r)
            dedup[key]["selection_tag"] = {r["selection_tag"]}
        else:
            dedup[key]["selection_tag"].add(r["selection_tag"])

    out = []

    for v in dedup.values():
        v["selection_tag"] = ",".join(sorted(v["selection_tag"]))
        out.append(v)

    return out


def write_tsv(path, rows):
    if len(rows) == 0:
        return

    fields = list(rows[0].keys())

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts_dir", required=True)
    ap.add_argument("--out_all_tsv", required=True)
    ap.add_argument("--out_raw_tsv", required=True)
    ap.add_argument("--out_dedup_tsv", required=True)
    ap.add_argument("--k", type=int, default=4)

    args = ap.parse_args()

    rows = read_all_rows(args.parts_dir)

    write_tsv(args.out_all_tsv, rows)

    raw = select_extremes(rows, args.k)

    write_tsv(args.out_raw_tsv, raw)

    dedup = deduplicate(raw)

    write_tsv(args.out_dedup_tsv, dedup)

    print("all rows:", len(rows))
    print("raw rows:", len(raw))
    print("dedup rows:", len(dedup))


if __name__ == "__main__":
    main()