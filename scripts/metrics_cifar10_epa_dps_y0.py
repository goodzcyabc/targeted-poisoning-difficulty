#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_model():
    return resnet18(num_classes=10)


def train_one_epoch(model, loader, device, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.item()) * bs
        total += bs

    return total_loss / max(1, total)


@torch.no_grad()
def eval_targets(model, target_images, device):
    model.eval()
    preds = []

    for x in target_images:
        x = x.unsqueeze(0).to(device)
        logits = model(x)
        preds.append(int(torch.argmax(logits, dim=1).item()))

    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--yt", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--inits", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=str("./data"),
        train=True,
        download=True,
        transform=transform,
    )

    test_set = torchvision.datasets.CIFAR10(
        root=str("./data"),
        train=False,
        download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    yt_indices = [i for i, (_, y) in enumerate(test_set) if y == args.yt]
    target_images = [test_set[i][0] for i in yt_indices]

    if len(yt_indices) != 1000:
        print(f"[WARN] expected 1000 test samples for yt={args.yt}, got {len(yt_indices)}")

    correct_counts = np.zeros(len(yt_indices), dtype=np.int64)
    pred_counters = [dict() for _ in range(len(yt_indices))]

    criterion = nn.CrossEntropyLoss()

    for m in range(args.inits):
        seed_all(args.seed0 + m)

        model = build_model().to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        for ep in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
            preds = eval_targets(model, target_images, device)

            for i, p in enumerate(preds):
                if p == args.yt:
                    correct_counts[i] += 1
                pred_counters[i][p] = pred_counters[i].get(p, 0) + 1

            print(
                f"[m={m+1}/{args.inits}] "
                f"epoch {ep+1:02d}/{args.epochs} "
                f"train_loss={train_loss:.4f}"
            )

    denom = float(args.inits * args.epochs)

    rows = []
    for idx, c, counter in zip(yt_indices, correct_counts.tolist(), pred_counters):
        mode_label, mode_count = max(counter.items(), key=lambda kv: kv[1])
        epa = c / denom
        dps = mode_count / denom

        rows.append({
            "yt": int(args.yt),
            "test_ds_index": int(idx),
            "epa": float(epa),
            "dps": float(dps),
            "mode_label": int(mode_label),
            "mode_count": int(mode_count),
            "M": int(args.inits),
            "N": int(args.epochs),
        })

    out_json = os.path.join(args.out_dir, f"metrics_y{args.yt}.json")
    out_csv = os.path.join(args.out_dir, f"metrics_y{args.yt}.csv")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["yt", "test_ds_index", "epa", "dps", "mode_label", "mode_count", "M", "N"])
        for r in rows:
            w.writerow([
                r["yt"],
                r["test_ds_index"],
                f'{r["epa"]:.10f}',
                f'{r["dps"]:.10f}',
                r["mode_label"],
                r["mode_count"],
                r["M"],
                r["N"],
            ])

    print(f"[out] wrote {out_json}")
    print(f"[out] wrote {out_csv}")


if __name__ == "__main__":
    main()