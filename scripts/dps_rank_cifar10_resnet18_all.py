#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def build_model() -> nn.Module:
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
def eval_testset_preds(model, loader, device):
    model.eval()
    preds = []

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
        preds.append(batch_preds)

    return np.concatenate(preds, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--inits", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--eval_batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_set = torchvision.datasets.CIFAR10(
        root=str(Path.home() / "datasets"),
        train=True,
        download=True,
        transform=transform,
    )

    test_set = torchvision.datasets.CIFAR10(
        root=str(Path.home() / "datasets"),
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

    test_loader = DataLoader(
        test_set,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    test_labels = np.array(test_set.targets, dtype=np.int64)
    num_test = len(test_set)
    num_classes = 10

    # counts[i, c] = number of times sample i predicted as class c across M x N
    counts = np.zeros((num_test, num_classes), dtype=np.int64)

    for m in range(args.inits):
        seed_all(args.seed0 + m)

        model = build_model().to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
        )
        criterion = nn.CrossEntropyLoss()

        for ep in range(args.epochs):
            train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
            preds = eval_testset_preds(model, test_loader, device)

            counts[np.arange(num_test), preds] += 1

            acc = float((preds == test_labels).mean())
            print(
                f"[m={m+1}/{args.inits}] "
                f"epoch {ep+1:02d}/{args.epochs} "
                f"train_loss={train_loss:.4f} "
                f"test_acc={acc:.4f}"
            )

    denom = args.inits * args.epochs

    rows = []
    for i in range(num_test):
        row_counts = counts[i]
        mode_label = int(np.argmax(row_counts))
        mode_count = int(row_counts[mode_label])
        dps = mode_count / float(denom)
        yt = int(test_labels[i])

        rows.append({
            "yt": yt,
            "test_ds_index": int(i),
            "dps": float(dps),
            "mode_label": mode_label,
            "mode_count": mode_count,
            "dominant_is_yt": int(mode_label == yt),
            "M": int(args.inits),
            "N": int(args.epochs),
        })

    with open(out_dir / "dps_rankings_all.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with open(out_dir / "dps_rankings_all.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "yt", "test_ds_index", "dps",
            "mode_label", "mode_count",
            "dominant_is_yt", "M", "N"
        ])
        for r in rows:
            w.writerow([
                r["yt"],
                r["test_ds_index"],
                f'{r["dps"]:.10f}',
                r["mode_label"],
                r["mode_count"],
                r["dominant_is_yt"],
                r["M"],
                r["N"],
            ])

    print(f"[out] wrote {out_dir / 'dps_rankings_all.json'}")
    print(f"[out] wrote {out_dir / 'dps_rankings_all.csv'}")


if __name__ == "__main__":
    main()