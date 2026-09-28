#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_class_order_and_names_from_words(path: str):
    wnids = []
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing words/class-order file: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" in line:
                wnid, _ = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                wnid = parts[0]
            wnids.append(wnid.strip())

    if len(wnids) != 200:
        raise RuntimeError(f"Expected 200 classes in {path}, got {len(wnids)}")

    return wnids


class TinyImageNetTrainDataset(Dataset):
    def __init__(self, data_path: str, wnids, transform=None):
        self.transform = transform
        self.class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}
        self.samples = []

        train_root = os.path.join(data_path, "train")
        for wnid in wnids:
            img_dir = os.path.join(train_root, wnid, "images")
            if not os.path.isdir(img_dir):
                continue
            for fn in sorted(os.listdir(img_dir)):
                if fn.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((os.path.join(img_dir, fn), self.class_to_idx[wnid]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


class TinyImageNetValDataset(Dataset):
    def __init__(self, data_path: str, wnids, transform=None):
        self.transform = transform
        self.class_to_idx = {wnid: i for i, wnid in enumerate(wnids)}
        self.samples = []

        val_root = os.path.join(data_path, "val")
        ann_path = os.path.join(val_root, "val_annotations.txt")
        img_root = os.path.join(val_root, "images")

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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def build_model():
    m = models.vgg16(weights=None)
    m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, 200)
    return m


@torch.no_grad()
def eval_loader(model, loader, device):
    model.eval()
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        correct += (preds == y).sum().item()
        total += y.size(0)

    return correct / max(1, total)


def train_one_epoch(model, loader, device, optimizer, criterion):
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
        total_loss += float(loss.item()) * bs
        total += bs

    return total_loss / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--class_order_file", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    seed_all(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    wnids = load_class_order_and_names_from_words(args.class_order_file)

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

    train_ds = TinyImageNetTrainDataset(args.data_path, wnids, transform=train_tf)
    val_ds = TinyImageNetValDataset(args.data_path, wnids, transform=val_tf)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=256,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_model().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    print("=== START TRAINING ===")
    start = time.time()

    for ep in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, device, optimizer, criterion)
        val_acc = eval_loader(model, val_loader, device)

        print(
            f"Epoch {ep+1:03d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_acc={val_acc:.4f}"
        )

    final_val_acc = eval_loader(model, val_loader, device)
    print(f"\n=== FINAL VAL ACC: {final_val_acc:.4f} ===")
    print(f"=== TOTAL TIME: {(time.time()-start)/60:.1f} min ===")


if __name__ == "__main__":
    main()