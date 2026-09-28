#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_model(model_name: str = "vgg13", num_classes: int = 10) -> nn.Module:
    if model_name.lower() == "vgg13":
        m = models.vgg13(weights=None)
        in_dim = m.classifier[-1].in_features
        m.classifier[-1] = nn.Linear(in_dim, num_classes)
        return m
    raise ValueError(f"Unsupported model: {model_name}")


@torch.no_grad()
def eval_targets(model: nn.Module, target_images: List[torch.Tensor], device: torch.device) -> List[int]:
    model.eval()
    preds = []
    for x in target_images:
        x = x.unsqueeze(0).to(device)
        logits = model(x)
        preds.append(int(torch.argmax(logits, dim=1).item()))
    return preds


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


def dump_json(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--yt", type=int, default=0)
    ap.add_argument("--model", default="vgg13")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--inits", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--group_k", type=int, default=4)
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
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()

    for m in range(args.inits):
        seed_all(args.seed0 + m)

        model = build_model(args.model, 10).to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.lr)

        for ep in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
            preds = eval_targets(model, target_images, device)

            for i, p in enumerate(preds):
                if p == args.yt:
                    correct_counts[i] += 1

            print(
                f"[m={m+1}/{args.inits}] "
                f"epoch {ep+1:02d}/{args.epochs} "
                f"train_loss={train_loss:.4f}"
            )

    denom = float(args.inits * args.epochs)
    epa = correct_counts.astype(np.float64) / denom

    rows = []
    for idx, score, c in zip(yt_indices, epa.tolist(), correct_counts.tolist()):
        rows.append({
            "yt": int(args.yt),
            "test_ds_index": int(idx),
            "epa": float(score),
            "correct": int(c),
            "M": int(args.inits),
            "N": int(args.epochs),
        })

    rows_sorted = sorted(rows, key=lambda r: r["epa"])

    k = args.group_k
    lowest = rows_sorted[:k]
    highest = rows_sorted[-k:]

    n = len(rows_sorted)
    mid_start = max(0, (n // 2) - (k // 2))
    middle = rows_sorted[mid_start:mid_start + k]

    ranking_csv = os.path.join(args.out_dir, f"epa_rankings_yt{args.yt}.csv")
    lowest_json = os.path.join(args.out_dir, f"epa_lowest{k}_yt{args.yt}.json")
    middle_json = os.path.join(args.out_dir, f"epa_middle{k}_yt{args.yt}.json")
    highest_json = os.path.join(args.out_dir, f"epa_highest{k}_yt{args.yt}.json")

    with open(ranking_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["yt", "test_ds_index", "epa", "correct", "M", "N"])
        for r in rows:
            w.writerow([
                r["yt"],
                r["test_ds_index"],
                f'{r["epa"]:.10f}',
                r["correct"],
                r["M"],
                r["N"],
            ])

    dump_json(lowest_json, lowest)
    dump_json(middle_json, middle)
    dump_json(highest_json, highest)

    print(f"[out] wrote ranking: {ranking_csv}")
    print(f"[out] wrote lowest{k}: {lowest_json}")
    print(f"[out] wrote middle{k}: {middle_json}")
    print(f"[out] wrote highest{k}: {highest_json}")
    print(f"[done] total_time={(time.time() - start_time)/60:.1f} min")


if __name__ == "__main__":
    main()