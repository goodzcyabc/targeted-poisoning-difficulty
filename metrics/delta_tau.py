#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, Any

import torch
import torch.nn as nn


@dataclass
class DeltaTauConfig:
    # delta search
    alpha: float = 1e-4
    m_init: float = 1.0
    max_doublings: int = 40
    device: str = "cuda"

    # tau estimation
    tau_num_batches: int = 50
    tau_batch_size: int = 256

    # TinyImageNet
    num_classes: int = 200


def _lambertw_real(x: float) -> float:
    """
    Real-valued Lambert W for x>0 using mpmath.
    """
    try:
        import mpmath as mp
        return float(mp.lambertw(x))
    except Exception as e:
        raise RuntimeError(
            "LambertW not available. Install mpmath in the active env."
        ) from e


def _flatten_params(model: nn.Module) -> torch.Tensor:
    parts = []
    for p in model.parameters():
        if p.requires_grad:
            parts.append(p.detach().flatten())
    if len(parts) == 0:
        raise RuntimeError("Model has no trainable parameters.")
    return torch.cat(parts)


def _set_params_from_vector(model: nn.Module, vec: torch.Tensor) -> None:
    with torch.no_grad():
        offset = 0
        for p in model.parameters():
            if not p.requires_grad:
                continue
            numel = p.numel()
            p.copy_(vec[offset:offset + numel].view_as(p))
            offset += numel
        if offset != vec.numel():
            raise ValueError(f"Vector length mismatch: used {offset}, got {vec.numel()}")


@torch.no_grad()
def _predict_label(model: nn.Module, x: torch.Tensor) -> int:
    model.eval()
    logits = model(x)
    return int(torch.argmax(logits, dim=1).item())


def compute_delta(
    model: nn.Module,
    x_t: torch.Tensor,
    y_p: int,
    cfg: DeltaTauConfig,
    loss_fn: Optional[nn.Module] = None,
) -> Dict[str, Any]:
    """
    Compute delta for one (x_t, y_p) on a clean model w_c.

    Definition used here:
      g = ∇_w CE(f(x_t; w_c), y_p)
      delta = min eta s.t. f(x_t; w_c - eta * g) predicts y_p

    Returns:
      {
        delta, eta_star, bracket_low, bracket_high,
        num_doublings, num_bisect,
        pred_before, pred_after, g_norm
      }
    """
    device = torch.device(cfg.device)
    model = model.to(device)
    model.eval()

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    x = x_t.to(device)
    if x.dim() == 3:
        x = x.unsqueeze(0)

    y = torch.tensor([y_p], device=device, dtype=torch.long)

    # clean weights
    w_c = _flatten_params(model).clone()

    # clear grads
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()

    # compute g at w_c
    logits = model(x)
    loss = loss_fn(logits, y)
    grads = torch.autograd.grad(
        loss,
        [p for p in model.parameters() if p.requires_grad],
        create_graph=False
    )
    g = torch.cat([g_i.detach().flatten() for g_i in grads])
    g_norm = float(torch.norm(g).item())

    pred_before = _predict_label(model, x)

    if g_norm == 0.0:
        return {
            "delta": float("inf"),
            "eta_star": float("inf"),
            "bracket_low": None,
            "bracket_high": None,
            "num_doublings": 0,
            "num_bisect": 0,
            "pred_before": pred_before,
            "pred_after": pred_before,
            "g_norm": g_norm,
        }

    def predict_with_eta(eta: float) -> int:
        w_new = w_c - eta * g
        _set_params_from_vector(model, w_new)
        return _predict_label(model, x)

    # bracket search
    m_low = 0.0
    m_high = cfg.m_init
    num_doublings = 0

    pred_high = predict_with_eta(m_high)
    while pred_high != y_p and num_doublings < cfg.max_doublings:
        m_low = m_high
        m_high *= 2.0
        num_doublings += 1
        pred_high = predict_with_eta(m_high)

    if pred_high != y_p:
        _set_params_from_vector(model, w_c)
        pred_after = _predict_label(model, x)
        return {
            "delta": float("inf"),
            "eta_star": float("inf"),
            "bracket_low": float(m_low),
            "bracket_high": float(m_high),
            "num_doublings": int(num_doublings),
            "num_bisect": 0,
            "pred_before": int(pred_before),
            "pred_after": int(pred_after),
            "g_norm": g_norm,
        }

    # binary search
    num_bisect = 0
    while (m_high - m_low) > cfg.alpha:
        m_mid = 0.5 * (m_low + m_high)
        pred_mid = predict_with_eta(m_mid)
        if pred_mid == y_p:
            m_high = m_mid
        else:
            m_low = m_mid
        num_bisect += 1

    eta_star = float(m_high)

    # final check at wp
    w_p = w_c - eta_star * g
    _set_params_from_vector(model, w_p)
    pred_after = _predict_label(model, x)

    # restore clean weights
    _set_params_from_vector(model, w_c)

    return {
        "delta": float(eta_star),
        "eta_star": float(eta_star),
        "bracket_low": float(m_low),
        "bracket_high": float(m_high),
        "num_doublings": int(num_doublings),
        "num_bisect": int(num_bisect),
        "pred_before": int(pred_before),
        "pred_after": int(pred_after),
        "g_norm": g_norm,
    }


def estimate_gDc_at_wp(
    model: nn.Module,
    train_loader: Iterable[Tuple[torch.Tensor, torch.Tensor]],
    cfg: DeltaTauConfig,
    loss_fn: Optional[nn.Module] = None,
) -> torch.Tensor:
    """
    Estimate average clean-data gradient g(D_c; w_p)
    at the model's *current* parameter setting.

    IMPORTANT:
      Caller must set model parameters to w_p before calling this.
    """
    device = torch.device(cfg.device)
    model = model.to(device)
    model.eval()

    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    g_acc = None
    num = 0

    for batch_idx, (x, y) in enumerate(train_loader):
        if batch_idx >= cfg.tau_num_batches:
            break

        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = loss_fn(logits, y)

        grads = torch.autograd.grad(
            loss,
            [p for p in model.parameters() if p.requires_grad],
            create_graph=False
        )
        g_vec = torch.cat([g_i.detach().flatten() for g_i in grads])

        if g_acc is None:
            g_acc = torch.zeros_like(g_vec)
        g_acc += g_vec
        num += 1

    if g_acc is None or num == 0:
        raise RuntimeError("train_loader produced no batches; cannot estimate g(Dc; wp).")

    return g_acc / float(num)

def compute_tau(
    w_c_vec: torch.Tensor,
    g_xt_yp: torch.Tensor,
    eta_star: float,
    gDc_wp: torch.Tensor,
    cfg: DeltaTauConfig,
) -> Dict[str, Any]:
    """
    Compute tau using:
      wp = wc - eta_star * g_xt_yp
      tau = max( <wp, g(Dc; wp)> / W(C - 1/e), 0 )
    """
    wp = w_c_vec - eta_star * g_xt_yp

    ip = float(torch.dot(wp.detach().cpu(), gDc_wp.detach().cpu()).item())

    denom_arg = cfg.num_classes - (1.0 / math.e)
    W = _lambertw_real(denom_arg)
    if W <= 0:
        raise RuntimeError(f"LambertW returned non-positive value W({denom_arg})={W}")

    tau_raw = ip / W
    tau = max(tau_raw, 0.0)

    return {
        "tau": float(tau),
        "tau_raw": float(tau_raw),
        "ip_wp_gDc": float(ip),
        "lambertw": float(W),
        "denom_arg": float(denom_arg),
    }


__all__ = [
    "DeltaTauConfig",
    "compute_delta",
    "estimate_gDc_at_wp",
    "compute_tau",
    "_flatten_params",
    "_set_params_from_vector",
]