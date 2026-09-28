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
def eval_targets(model, target_images, device):
    model.eval()
    preds = []

    for x in target_images:
        x = x.unsqueeze(0).to(device)
        logits = model(x)
        preds.append(int(logits.argmax(1).item()))

    return preds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--yt", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--inits", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
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

    yt_indices = [i for i, (_, y) in enumerate(test_set) if y == args.yt]
    target_images = [test_set[i][0] for i in yt_indices]

    if len(yt_indices) != 1000:
        print(f"[WARN] expected 1000 test samples for yt={args.yt}, got {len(yt_indices)}")

    correct_counts = np.zeros(len(yt_indices), dtype=np.int64)

    for m in range(args.inits):
        seed_all(args.seed0 + m)

        model = build_model().to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        criterion = nn.CrossEntropyLoss()

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

    denom = args.inits * args.epochs
    epa = correct_counts / float(denom)

    rows = []
    for idx, e, c in zip(yt_indices, epa.tolist(), correct_counts.tolist()):
        rows.append({
            "yt": int(args.yt),
            "test_ds_index": int(idx),
            "epa": float(e),
            "correct": int(c),
            "M": int(args.inits),
            "N": int(args.epochs),
        })

    rows.sort(key=lambda x: x["epa"])

    # write full csv
    with open(out_dir / "epa_rankings.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)

    with open(out_dir / "epa_rankings.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["yt", "test_ds_index", "epa", "correct", "M", "N"])
        for r in rows:
            w.writerow([r["yt"], r["test_ds_index"], f'{r["epa"]:.10f}', r["correct"], r["M"], r["N"]])

    n = len(rows)

    def take_bucket(name, start_idx):
        return rows[start_idx:start_idx + 10]

    buckets = {
        "lowest": rows[:10],
        "p20": take_bucket("p20", int(0.2 * n)),
        "p40": take_bucket("p40", int(0.4 * n)),
        "p60": take_bucket("p60", int(0.6 * n)),
        "p80": take_bucket("p80", int(0.8 * n)),
        "highest": rows[-10:],
    }

    with open(out_dir / "attack_targets.json", "w", encoding="utf-8") as f:
        json.dump(buckets, f, indent=2)

    print(f"[out] wrote {out_dir / 'epa_rankings.csv'}")
    print(f"[out] wrote {out_dir / 'attack_targets.json'}")


if __name__ == "__main__":
    main()