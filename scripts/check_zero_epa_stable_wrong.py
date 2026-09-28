#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import os
import random
from collections import Counter
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def load_wnids(data_path: str) -> List[str]:
    wnids_path = os.path.join(data_path, "wnids.txt")
    if not os.path.isfile(wnids_path):
        raise FileNotFoundError(f"Missing wnids.txt at {wnids_path}")
    with open(wnids_path, "r", encoding="utf-8") as f:
        wnids = [line.strip() for line in f if line.strip()]
    if len(wnids) != 200:
        raise RuntimeError(f"Expected 200 wnids, got {len(wnids)}")
    return wnids


class TinyImageNetTrainDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.transform = transform
        self.wnids = load_wnids(data_path)
        self.class_to_idx = {wnid: i for i, wnid in enumerate(self.wnids)}
        self.samples: List[Tuple[str, int]] = []

        train_root = os.path.join(data_path, "train")
        for wnid in self.wnids:
            img_dir = os.path.join(train_root, wnid, "images")
            if not os.path.isdir(img_dir):
                continue
            for fn in sorted(os.listdir(img_dir)):
                if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((os.path.join(img_dir, fn), self.class_to_idx[wnid]))

        if not self.samples:
            raise RuntimeError(f"No train samples found under {train_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


class TinyImageNetValDataset(Dataset):
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
        self.transform = transform
        self.wnids = load_wnids(data_path)
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


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    amp: bool,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs = x.size(0)
        total_loss += float(loss.detach().cpu().item()) * bs
        total += bs

    return total_loss / max(1, total)

@torch.no_grad()
def predict_targets(
    model: nn.Module,
    val_ds: TinyImageNetValDataset,
    target_ids: List[int],
    device: torch.device,
) -> Dict[int, int]:
    model.eval()
    out = {}
    for tid in target_ids:
        x, _ = val_ds[tid]
        x = x.unsqueeze(0).to(device)
        logits = model(x)
        pred = int(torch.argmax(logits, dim=1).item())
        out[tid] = pred
    return out


def read_zero_epa_targets(epa_csv: str) -> List[dict]:
    rows = []
    with open(epa_csv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            epa = float(row["epa"])
            if epa == 0.0:
                rows.append({
                    "yt": int(row["yt"]),
                    "val_ds_index": int(row["val_ds_index"]),
                    "rel_path": row["rel_path"],
                    "epa": epa,
                    "correct": int(row["correct"]),
                    "M": int(row["M"]),
                    "N": int(row["N"]),
                })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--epa_csv", required=True)
    ap.add_argument("--out_tsv", required=True)
    ap.add_argument("--model", default="vgg16")
    ap.add_argument("--num_classes", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--inits", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--amp", action="store_true")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = TinyImageNetTrainDataset(args.data_path, transform=train_tf)
    val_ds = TinyImageNetValDataset(args.data_path, transform=val_tf)

    zero_targets = read_zero_epa_targets(args.epa_csv)
    if not zero_targets:
        print("[info] no zero-EPA targets found")
        return

    target_ids = [r["val_ds_index"] for r in zero_targets]
    target_meta = {r["val_ds_index"]: r for r in zero_targets}

    print(f"[info] zero-EPA targets: {len(target_ids)}")
    print(f"[info] target_ids: {target_ids}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    criterion = nn.CrossEntropyLoss()

    # predictions across inits
    preds_by_target: Dict[int, List[int]] = {tid: [] for tid in target_ids}

    for m in range(args.inits):
        seed = args.seed0 + m
        seed_all(seed)

        model = build_model(args.model, args.num_classes).to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        print(f"\n[run {m+1}/{args.inits}] seed={seed}")

        for ep in range(args.epochs):
            tr_loss = train_one_epoch(
                model=model,
                loader=train_loader,
                device=device,
                optimizer=optimizer,
                criterion=criterion,
                amp=args.amp,
            )
            if (ep + 1) % 10 == 0 or ep == 0 or ep + 1 == args.epochs:
                print(f"  epoch {ep+1:02d}/{args.epochs}  train_loss={tr_loss:.4f}")

        preds = predict_targets(model, val_ds, target_ids, device)
        for tid, pred in preds.items():
            preds_by_target[tid].append(pred)

        print("  final preds:", {tid: preds[tid] for tid in target_ids})

    out_path = Path(args.out_tsv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "yt",
            "val_ds_index",
            "rel_path",
            "epa",
            "correct",
            "mode_label",
            "mode_count",
            "mode_ratio",
            "num_unique_labels",
            "preds_8",
        ])

        for tid in target_ids:
            preds = preds_by_target[tid]
            c = Counter(preds)
            mode_label, mode_count = c.most_common(1)[0]
            mode_ratio = mode_count / len(preds)
            num_unique_labels = len(c)
            meta = target_meta[tid]

            w.writerow([
                meta["yt"],
                tid,
                meta["rel_path"],
                f'{meta["epa"]:.10f}',
                meta["correct"],
                mode_label,
                mode_count,
                f"{mode_ratio:.6f}",
                num_unique_labels,
                ",".join(map(str, preds)),
            ])

            print(
                f'target_id={tid}  '
                f'mode_label={mode_label}  '
                f'mode_count={mode_count}  '
                f'mode_ratio={mode_ratio:.3f}  '
                f'unique={num_unique_labels}  '
                f'preds={preds}'
            )

    print(f"\n[out] wrote {out_path}")


if __name__ == "__main__":
    main()