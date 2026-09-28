#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms, models

# allow importing results/metrics/metrics_delta_tau.py
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BENCH_ROOT = os.path.dirname(SCRIPT_DIR)
EXTRA_DIR = os.path.join(BENCH_ROOT, "extra_delta_tau")
if EXTRA_DIR not in sys.path:
    sys.path.insert(0, EXTRA_DIR)

from metrics_delta_tau import DeltaTauConfig, compute_delta  # noqa: E402


def load_class_order_and_names_from_words(path: str) -> tuple[List[str], Dict[str, str]]:
    wnids: List[str] = []
    words_map: Dict[str, str] = {}
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing words/class-order file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" in line:
                wnid, desc = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                wnid = parts[0]
                desc = parts[1] if len(parts) > 1 else ""
            wnid = wnid.strip()
            desc = desc.strip()
            wnids.append(wnid)
            words_map[wnid] = desc

    if len(wnids) != 200:
        raise RuntimeError(f"Expected 200 classes in {path}, got {len(wnids)}")

    return wnids, words_map


class TinyImageNetValDataset(Dataset):
    def __init__(self, data_path: str, wnids: List[str], transform=None):
        self.data_path = data_path
        self.transform = transform
        self.wnids = wnids
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        self.samples: List[Tuple[str, int]] = []

        val_root = os.path.join(data_path, "val")
        ann_path = os.path.join(val_root, "val_annotations.txt")
        img_root = os.path.join(val_root, "images")

        if not os.path.isfile(ann_path):
            raise FileNotFoundError(f"Missing val_annotations.txt at {ann_path}")
        if not os.path.isdir(img_root):
            raise FileNotFoundError(f"Missing val/images at {img_root}")

        with open(ann_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) < 2:
                    continue
                img_name, wnid = parts[0], parts[1]
                if wnid not in self.class_to_idx:
                    continue
                path = os.path.join(img_root, img_name)
                y = self.class_to_idx[wnid]
                self.samples.append((path, y))

        if len(self.samples) != 10000:
            print(f"[WARN] expected 10000 val samples, got {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def build_model(model_name: str, num_classes: int) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "vgg16":
        m = models.vgg16(weights=None)
        in_dim = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_dim, num_classes)
        return m

    if model_name == "vgg16_bn":
        m = models.vgg16_bn(weights=None)
        in_dim = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_dim, num_classes)
        return m

    if model_name == "resnet18":
        m = models.resnet18(weights=None)
        in_dim = m.fc.in_features
        m.fc = nn.Linear(in_dim, num_classes)
        return m

    raise ValueError(f"Unsupported model: {model_name}")


def normalize_state_dict_keys(sd: dict) -> dict:
    out = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        if nk.startswith("model."):
            nk = nk[len("model."):]
        out[nk] = v
    return out


def robust_load_checkpoint(model: nn.Module, ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")

    candidate = ckpt
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            candidate = ckpt["state_dict"]
        elif "model" in ckpt and isinstance(ckpt["model"], dict):
            candidate = ckpt["model"]

    if not isinstance(candidate, dict):
        raise RuntimeError(f"Unsupported checkpoint format at {ckpt_path}")

    candidate = normalize_state_dict_keys(candidate)
    missing, unexpected = model.load_state_dict(candidate, strict=False)
    print(f"[ckpt] loaded {ckpt_path}")
    print(f"[ckpt] missing={len(missing)} unexpected={len(unexpected)}")


def dump_json(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--class_order_file", required=True,
                    help="Use words200.txt as class-order file")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--yt", type=int, default=0)
    ap.add_argument("--yp", type=int, required=True)
    ap.add_argument("--model", default="vgg16")
    ap.add_argument("--num_classes", type=int, default=200)
    ap.add_argument("--ckpt_path", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--topk", type=int, default=4)
    args = ap.parse_args()

    if args.yt == args.yp:
        raise ValueError(f"Need yp != yt, but got yt={args.yt}, yp={args.yp}")

    os.makedirs(args.out_dir, exist_ok=True)

    wnids, words_map = load_class_order_and_names_from_words(args.class_order_file)
    idx_to_wnid = {i: w for i, w in enumerate(wnids)}

    print(f"[mapping] yt={args.yt} -> {idx_to_wnid[args.yt]} ({words_map.get(idx_to_wnid[args.yt], '')})")
    print(f"[mapping] yp={args.yp} -> {idx_to_wnid[args.yp]} ({words_map.get(idx_to_wnid[args.yp], '')})")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_ds = TinyImageNetValDataset(args.data_path, wnids=wnids, transform=tfm)

    model = build_model(args.model, args.num_classes)
    robust_load_checkpoint(model, args.ckpt_path)
    model = model.to(device).eval()

    cfg = DeltaTauConfig(
        alpha=1e-4,
        m_init=1.0,
        max_doublings=40,
        device=str(device),
        tau_num_batches=50,
        tau_batch_size=256,
        num_classes=args.num_classes,
    )

    rows: List[dict] = []

    for val_idx, (path, label) in enumerate(val_ds.samples):
        if int(label) != int(args.yt):
            continue

        x_t, y_true = val_ds[val_idx]
        if int(y_true) != int(args.yt):
            raise RuntimeError(f"label mismatch at val_idx={val_idx}")

        rel_path = os.path.relpath(path, args.data_path)
        x_t = x_t.unsqueeze(0).to(device)

        dres = compute_delta(
            model=model,
            x_t=x_t,
            y_p=args.yp,
            cfg=cfg,
        )

        rows.append({
            "yt": int(args.yt),
            "yt_wnid": idx_to_wnid[int(args.yt)],
            "yt_name": words_map.get(idx_to_wnid[int(args.yt)], ""),
            "yp": int(args.yp),
            "yp_wnid": idx_to_wnid[int(args.yp)],
            "yp_name": words_map.get(idx_to_wnid[int(args.yp)], ""),
            "val_ds_index": int(val_idx),
            "rel_path": rel_path,
            "delta": float(dres["delta"]),
            "eta_star": float(dres["eta_star"]),
            "pred_before": int(dres["pred_before"]),
            "pred_after": int(dres["pred_after"]),
            "g_norm": float(dres["g_norm"]),
        })

        print(
            f'[delta] idx={val_idx} '
            f'pred_before={dres["pred_before"]} pred_after={dres["pred_after"]} '
            f'delta={dres["delta"]:.6f}'
        )

    if len(rows) != 50:
        print(f"[WARN] expected 50 targets for yt={args.yt}, got {len(rows)}")

    rows_sorted = sorted(rows, key=lambda r: r["delta"])
    bottom = rows_sorted[:args.topk]
    top = rows_sorted[-args.topk:]

    ranking_csv = os.path.join(args.out_dir, f"delta_rankings_yt{args.yt}_yp{args.yp}.csv")
    top_json = os.path.join(args.out_dir, f"delta_top{args.topk}_yt{args.yt}_yp{args.yp}.json")
    bottom_json = os.path.join(args.out_dir, f"delta_bottom{args.topk}_yt{args.yt}_yp{args.yp}.json")

    with open(ranking_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "yt", "yt_wnid", "yt_name",
            "yp", "yp_wnid", "yp_name",
            "val_ds_index", "rel_path",
            "delta", "eta_star",
            "pred_before", "pred_after", "g_norm"
        ])
        for r in rows:
            w.writerow([
                r["yt"], r["yt_wnid"], r["yt_name"],
                r["yp"], r["yp_wnid"], r["yp_name"],
                r["val_ds_index"], r["rel_path"],
                f'{r["delta"]:.10f}',
                f'{r["eta_star"]:.10f}',
                r["pred_before"], r["pred_after"], f'{r["g_norm"]:.10f}',
            ])

    dump_json(bottom_json, bottom)
    dump_json(top_json, top)

    print(f"[out] wrote ranking: {ranking_csv}")
    print(f"[out] wrote bottom{args.topk}: {bottom_json}")
    print(f"[out] wrote top{args.topk}: {top_json}")


if __name__ == "__main__":
    main()