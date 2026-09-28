#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models

# ---- FIX 1: make extra_delta_tau importable no matter where python is launched from ----
BENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BENCH_ROOT not in sys.path:
    sys.path.insert(0, BENCH_ROOT)

from extra_delta_tau.metrics_delta_tau import (
    DeltaTauConfig,
    compute_delta,
    estimate_gDc_at_wp,
    compute_tau,
    _flatten_params,
    _set_params_from_vector,
)


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


def normalize_state_dict_keys(sd: dict) -> dict:
    """Strip common prefixes like module. / model."""
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
    """
    FIX 2: robust checkpoint loading for common .pth formats.
    Supports:
      - raw state_dict
      - {"state_dict": ...}
      - prefixed keys: module.*, model.*
    """
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
    print(f"[ckpt] loaded from {ckpt_path}")
    print(f"[ckpt] missing keys: {len(missing)}")
    print(f"[ckpt] unexpected keys: {len(unexpected)}")

    # Hard fail only if loading clearly went wrong
    # final classifier should usually match if this is really TinyImageNet VGG16
    if len(candidate) == 0:
        raise RuntimeError(f"Checkpoint appears empty: {ckpt_path}")

def read_candidate_row(candidate_tsv: str, row_idx_1based: int) -> dict:
    with open(candidate_tsv, "r", encoding="utf-8") as f:
        r = csv.DictReader(f, delimiter="\t")
        rows = list(r)

    if row_idx_1based < 1 or row_idx_1based > len(rows):
        raise IndexError(f"row_idx={row_idx_1based} out of range 1..{len(rows)}")

    return rows[row_idx_1based - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--candidate_tsv", required=True)
    ap.add_argument("--row_idx", type=int, required=True, help="1-based row index in candidate_tsv")
    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--model", default="vgg16")
    ap.add_argument("--num_classes", type=int, default=200)
    ap.add_argument("--ckpt_path", required=True)

    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tau_num_batches", type=int, default=50)
    ap.add_argument("--tau_batch_size", type=int, default=256)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    tfm = transforms.Compose([
        transforms.Resize(64),
        transforms.CenterCrop(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = TinyImageNetTrainDataset(args.data_path, transform=tfm)
    val_ds = TinyImageNetValDataset(args.data_path, transform=tfm)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.tau_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    row = read_candidate_row(args.candidate_tsv, args.row_idx)

    target_tag = row["target_tag"]
    target_val_idx = int(row["target_val_ds_index"])
    yt = int(row["yt"])
    target_rel_path = row["target_rel_path"]
    target_epa = float(row["target_epa"])
    candidate_val_idx = int(row["candidate_val_ds_index"])
    yp = int(row["yp"])
    candidate_rel_path = row["candidate_rel_path"]

    model = build_model(args.model, args.num_classes)
    robust_load_checkpoint(model, args.ckpt_path)
    model = model.to(device).eval()

    cfg = DeltaTauConfig(
        alpha=1e-4,
        m_init=1.0,
        max_doublings=40,
        device=str(device),
        tau_num_batches=args.tau_num_batches,
        tau_batch_size=args.tau_batch_size,
        num_classes=args.num_classes,
    )

    x_t, y_true = val_ds[target_val_idx]
    if int(y_true) != yt:
        raise RuntimeError(
            f"Label mismatch for target idx={target_val_idx}: expected yt={yt}, got y_true={y_true}"
        )

    x_t = x_t.unsqueeze(0).to(device)

    # keep clean params for tau
    w_c_vec = _flatten_params(model).clone()

    dres = compute_delta(
        model=model,
        x_t=x_t,
        y_p=yp,
        cfg=cfg,
    )

    delta = float(dres["delta"])
    eta_star = float(dres["eta_star"])
    pred_before = int(dres["pred_before"])
    pred_after = int(dres["pred_after"])
    g_norm = float(dres["g_norm"])

    tau = float("inf")

    if math.isfinite(delta):
        loss_fn = nn.CrossEntropyLoss()

        for p in model.parameters():
            if p.grad is not None:
                p.grad.zero_()

        logits = model(x_t)
        y_target = torch.tensor([yp], device=device, dtype=torch.long)
        loss = loss_fn(logits, y_target)
        grads = torch.autograd.grad(
            loss,
            [p for p in model.parameters() if p.requires_grad],
            create_graph=False,
        )
        g_xt_yp = torch.cat([g.detach().flatten() for g in grads])

        wp_vec = w_c_vec - eta_star * g_xt_yp
        _set_params_from_vector(model, wp_vec)

        gDc_wp = estimate_gDc_at_wp(
            model=model,
            train_loader=train_loader,
            cfg=cfg,
        )

        tau_res = compute_tau(
            w_c_vec=w_c_vec,
            g_xt_yp=g_xt_yp,
            eta_star=eta_star,
            gDc_wp=gDc_wp,
            cfg=cfg,
        )
        tau = float(tau_res["tau"])

        _set_params_from_vector(model, w_c_vec)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"row_{args.row_idx:03d}.tsv"

    with open(out_file, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f, delimiter="\t")
        w.writerow([
            "target_tag",
            "target_val_ds_index",
            "yt",
            "target_rel_path",
            "target_epa",
            "candidate_val_ds_index",
            "yp",
            "candidate_rel_path",
            "delta",
            "eta_star",
            "tau",
            "pred_before",
            "pred_after",
            "g_norm",
        ])
        w.writerow([
            target_tag,
            target_val_idx,
            yt,
            target_rel_path,
            f"{target_epa:.10f}",
            candidate_val_idx,
            yp,
            candidate_rel_path,
            f"{delta:.10f}" if math.isfinite(delta) else "inf",
            f"{eta_star:.10f}" if math.isfinite(eta_star) else "inf",
            f"{tau:.10f}" if math.isfinite(tau) else "inf",
            pred_before,
            pred_after,
            f"{g_norm:.10f}",
        ])

    print(f"[out] wrote {out_file}")


if __name__ == "__main__":
    main()