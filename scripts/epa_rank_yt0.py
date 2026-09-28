#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models


# ---------------------------
# utils
# ---------------------------

def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def _safe_mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


# ---------------------------
# TinyImageNet datasets
# ---------------------------

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
    """
    Correctly reads TinyImageNet train split:
      train/<wnid>/images/*.JPEG
    """
    def __init__(self, data_path: str, transform=None):
        self.data_path = data_path
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

        if len(self.samples) == 0:
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
    """
    Correctly reads TinyImageNet val split using val_annotations.txt:
      val/images/*.JPEG
      val/val_annotations.txt
    """
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
            print(f"[WARN] Expected 10000 val samples, got {len(self.samples)}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y

class TinyImageNetValSubset(Dataset):
    """
    Fixed subset over val indices, preserving original val index and relative path.
    """
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


# ---------------------------
# model factory
# ---------------------------

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


# ---------------------------
# train / eval
# ---------------------------

@torch.no_grad()
def eval_on_subset(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()

    preds: List[int] = []
    tgts: List[int] = []

    for batch in loader:
        x, y, _, _ = batch
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        p = torch.argmax(logits, dim=1)

        preds.extend(p.detach().cpu().tolist())
        tgts.extend(y.detach().cpu().tolist())

    return np.array(preds, dtype=np.int64), np.array(tgts, dtype=np.int64)


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


# ---------------------------
# main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--yt", type=int, default=0)
    ap.add_argument("--model", type=str, default="vgg16")
    ap.add_argument("--num_classes", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--inits", type=int, default=8)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight_decay", type=float, default=5e-4)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    _safe_mkdir(args.out_dir)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

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

    # Correctly keep only label == yt
    yt_indices: List[int] = []
    yt_items: List[Tuple[int, str, int]] = []  # (val_ds_index, rel_path, label)

    for i, (path, label) in enumerate(val_ds.samples):
        if int(label) == int(args.yt):
            yt_indices.append(i)
            rel_path = os.path.relpath(path, args.data_path)
            yt_items.append((i, rel_path, int(label)))

    if len(yt_indices) != 50:
        print(f"[WARN] expected 50 val images for yt={args.yt}, got {len(yt_indices)}")

    yt_subset = TinyImageNetValSubset(val_ds, yt_indices)

    yt_loader = DataLoader(
        yt_subset,
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
    
    state_path = os.path.join(args.out_dir, f"epa_state_yt{args.yt}.json")
    ranking_csv = os.path.join(args.out_dir, f"epa_rankings_yt{args.yt}.csv")
    top1_path = os.path.join(args.out_dir, f"epa_top1_yt{args.yt}.json")
    bottom1_path = os.path.join(args.out_dir, f"epa_bottom1_yt{args.yt}.json")

    correct_counts = np.zeros(len(yt_indices), dtype=np.int64)
    done_inits = 0

    if args.resume and os.path.isfile(state_path):
        with open(state_path, "r", encoding="utf-8") as f:
            st = json.load(f)
        if st.get("yt") == args.yt and st.get("len_subset") == len(yt_indices):
            correct_counts = np.array(st["correct_counts"], dtype=np.int64)
            done_inits = int(st["done_inits"])
            print(f"[resume] loaded state: done_inits={done_inits}/{args.inits}")
        else:
            print("[resume] state mismatch; ignoring old state")

    criterion = nn.CrossEntropyLoss()

    start_time = time.time()

    for m in range(done_inits, args.inits):
        seed = args.seed0 + m
        seed_all(seed)

        model = build_model(args.model, args.num_classes).to(device)
        optimizer = optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )

        for ep in range(args.epochs):
            tr_loss = train_one_epoch(model, train_loader, device, optimizer, criterion, amp=args.amp)
            preds_np, tgts_np = eval_on_subset(model, yt_loader, device)

            # should be exactly aligned with yt_indices order because shuffle=False
            correct_counts += (preds_np == tgts_np).astype(np.int64)

            print(
                f"[m={m+1}/{args.inits}] "
                f"epoch {ep+1:02d}/{args.epochs} "
                f"train_loss={tr_loss:.4f} "
                f"yt_subset_acc={(preds_np == tgts_np).mean():.4f}"
            )

        st = {
            "yt": args.yt,
            "model": args.model,
            "epochs": args.epochs,
            "inits": args.inits,
            "done_inits": m + 1,
            "len_subset": len(yt_indices),
            "yt_indices": yt_indices,
            "correct_counts": correct_counts.tolist(),
            "data_path": args.data_path,
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(st, f, indent=2)
        print(f"[state] wrote {state_path}")

    denom = float(args.inits * args.epochs)
    epa = correct_counts.astype(np.float64) / denom

    with open(ranking_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["yt", "val_ds_index", "rel_path", "epa", "correct", "M", "N"])
        for (val_idx, rel_path, label), e, c in zip(yt_items, epa.tolist(), correct_counts.tolist()):
            w.writerow([label, val_idx, rel_path, f"{e:.10f}", int(c), args.inits, args.epochs])

    print(f"[out] wrote rankings: {ranking_csv}")

    top_i = int(np.argmax(epa))
    bot_i = int(np.argmin(epa))

    def dump_one(path: str, idx_in_subset: int) -> None:
        val_idx, rel_path, label = yt_items[idx_in_subset]
        payload = {
            "yt": int(label),
            "val_ds_index": int(val_idx),
            "rel_path": rel_path,
            "epa": float(epa[idx_in_subset]),
            "correct": int(correct_counts[idx_in_subset]),
            "M": args.inits,
            "N": args.epochs,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"[out] wrote {path}: {payload}")

    dump_one(top1_path, top_i)
    dump_one(bottom1_path, bot_i)

    print(f"[done] total_time={(time.time() - start_time)/60:.1f} min device={device}")


if __name__ == "__main__":
    main()