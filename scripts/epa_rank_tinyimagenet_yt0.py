#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from typing import Dict, List, Tuple

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


class TinyImageNetTrainDataset(Dataset):
    def __init__(self, data_path: str, wnids: List[str], transform=None):
        self.transform = transform
        self.wnids = wnids
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


class TinyImageNetValSubset(Dataset):
    def __init__(self, base_ds: TinyImageNetValDataset, indices: List[int]):
        self.base_ds = base_ds
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        base_idx = self.indices[i]
        x, y = self.base_ds[base_idx]
        abs_path, _ = self.base_ds.samples[base_idx]
        rel_path = os.path.relpath(abs_path, self.base_ds.data_path)
        return x, y, base_idx, rel_path


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

    raise ValueError(f"Unsupported model: {model_name}")


@torch.no_grad()
def eval_on_subset(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds = []
    tgts = []

    for batch in loader:
        x, y, _, _ = batch
        x = x.to(device, non_blocking=True)
        logits = model(x)
        p = torch.argmax(logits, dim=1).detach().cpu().numpy()
        preds.append(p)
        tgts.append(y.numpy())

    return np.concatenate(preds), np.concatenate(tgts)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
) -> float:
    model.train()
    total_loss = 0.0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += float(loss.detach().cpu().item()) * bs
        total += bs

    return total_loss / max(1, total)


def dump_json(path: str, rows: List[dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--class_order_file", required=True, help="Use words200.txt as class-order file")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--yt", type=int, default=0)
    ap.add_argument("--model", default="vgg16")
    ap.add_argument("--num_classes", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--inits", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--group_k", type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    wnids, words_map = load_class_order_and_names_from_words(args.class_order_file)
    idx_to_wnid = {i: w for i, w in enumerate(wnids)}

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

    train_ds = TinyImageNetTrainDataset(args.data_path, wnids=wnids, transform=train_tf)
    val_ds = TinyImageNetValDataset(args.data_path, wnids=wnids, transform=val_tf)

    yt_indices: List[int] = []
    yt_items: List[Tuple[int, str]] = []

    for i, (path, label) in enumerate(val_ds.samples):
        if int(label) == int(args.yt):
            yt_indices.append(i)
            yt_items.append((i, os.path.relpath(path, args.data_path)))

    if len(yt_indices) != 50:
        print(f"[WARN] expected 50 val images for yt={args.yt}, got {len(yt_indices)}")

    subset = TinyImageNetValSubset(val_ds, yt_indices)

    subset_loader = DataLoader(
        subset,
        batch_size=64,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    correct_counts = np.zeros(len(yt_indices), dtype=np.int64)
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()

    for m in range(args.inits):
        seed_all(args.seed0 + m)

        model = build_model(args.model, args.num_classes).to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
        )

        for ep in range(args.epochs):
            tr_loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
            preds, tgts = eval_on_subset(model, subset_loader, device)
            correct_counts += (preds == tgts).astype(np.int64)

            print(
                f"[m={m+1}/{args.inits}] "
                f"epoch {ep+1:02d}/{args.epochs} "
                f"train_loss={tr_loss:.4f} "
                f"subset_acc={(preds == tgts).mean():.4f}"
            )

    denom = float(args.inits * args.epochs)
    epa = correct_counts.astype(np.float64) / denom

    rows = []
    for (val_idx, rel_path), score, c in zip(yt_items, epa.tolist(), correct_counts.tolist()):
        rows.append({
            "yt": int(args.yt),
            "yt_wnid": idx_to_wnid[int(args.yt)],
            "yt_name": words_map.get(idx_to_wnid[int(args.yt)], ""),
            "val_ds_index": int(val_idx),
            "rel_path": rel_path,
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
        w.writerow([
            "yt", "yt_wnid", "yt_name",
            "val_ds_index", "rel_path",
            "epa", "correct", "M", "N"
        ])
        for r in rows:
            w.writerow([
                r["yt"], r["yt_wnid"], r["yt_name"],
                r["val_ds_index"], r["rel_path"],
                f'{r["epa"]:.10f}', r["correct"], r["M"], r["N"]
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