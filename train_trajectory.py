"""Train LSTM trajectory model for bbox prediction.

Bidirectional LSTM encoder with MLP trajectory head. Predicts residuals
from constant-velocity baseline for faster convergence.

Usage:
    python train_trajectory.py

Produces: model_trajectory.pt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sequence import TrajectoryDataset, HORIZON_KEYS, HORIZONS_FRAMES
from features import REQUEST_FIELDS

DATA = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model_trajectory.pt"
DEVICE = torch.device("cpu")


class AttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.attn(lstm_out), dim=1)
        return (lstm_out * weights).sum(dim=1)


class TrajectoryLSTM(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 128,
                 num_layers: int = 2, num_horizons: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_horizons = num_horizons

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2,
        )
        self.attn_pool = AttentionPool(hidden_dim * 2)

        self.trajectory_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_horizons * 4),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        pooled = self.attn_pool(lstm_out)
        deltas = self.trajectory_head(pooled)
        return deltas.view(-1, self.num_horizons, 4)


class TrajectoryLoss(nn.Module):
    def __init__(self, horizon_weights: list[float] | None = None):
        super().__init__()
        if horizon_weights is None:
            horizon_weights = [1.0, 1.2, 1.5, 2.0]
        self.horizon_weights = torch.tensor(horizon_weights, dtype=torch.float32)

    def forward(self, pred_deltas: torch.Tensor, targets: torch.Tensor,
                cv_base: torch.Tensor) -> torch.Tensor:
        predictions = cv_base + pred_deltas
        diff = predictions - targets

        cx_pred = (predictions[:, :, 0] + predictions[:, :, 2]) * 0.5
        cy_pred = (predictions[:, :, 1] + predictions[:, :, 3]) * 0.5
        cx_tgt = (targets[:, :, 0] + targets[:, :, 2]) * 0.5
        cy_tgt = (targets[:, :, 1] + targets[:, :, 3]) * 0.5

        center_loss = (cx_pred - cx_tgt)**2 + (cy_pred - cy_tgt)**2
        size_loss = diff[:, :, 2]**2 + diff[:, :, 3]**2

        w = self.horizon_weights.to(pred_deltas.device).unsqueeze(0)
        center_loss = (center_loss * w).mean()
        size_loss = (size_loss * w).mean()

        return center_loss + 0.1 * size_loss


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    n_batches = 0
    for seq, target, cv_base in loader:
        seq = seq.to(device)
        target = target.to(device)
        cv_base = cv_base.to(device)

        optimizer.zero_grad()
        pred_deltas = model(seq)
        loss = criterion(pred_deltas, target, cv_base)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, df=None, fw=1920.0, fh=1080.0):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []
    all_cv = []

    for seq, target, cv_base in loader:
        seq = seq.to(device)
        target = target.to(device)
        cv_base = cv_base.to(device)

        pred_deltas = model(seq)
        loss = criterion(pred_deltas, target, cv_base)

        total_loss += loss.item()
        n_batches += 1

        predictions = cv_base + pred_deltas
        all_preds.append(predictions.cpu().numpy())
        all_targets.append(target.cpu().numpy())
        all_cv.append(cv_base.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)

    preds = np.concatenate(all_preds, axis=0)
    tgts = np.concatenate(all_targets, axis=0)
    cvs = np.concatenate(all_cv, axis=0)

    ades_model = []
    ades_cv = []
    for j in range(4):
        pcx = (preds[:, j, 0] + preds[:, j, 2]) * 0.5 * fw
        pcy = (preds[:, j, 1] + preds[:, j, 3]) * 0.5 * fh
        tcx = (tgts[:, j, 0] + tgts[:, j, 2]) * 0.5 * fw
        tcy = (tgts[:, j, 1] + tgts[:, j, 3]) * 0.5 * fh
        ades_model.append(np.sqrt((pcx - tcx)**2 + (pcy - tcy)**2).mean())

        ccx = (cvs[:, j, 0] + cvs[:, j, 2]) * 0.5 * fw
        ccy = (cvs[:, j, 1] + cvs[:, j, 3]) * 0.5 * fh
        ades_cv.append(np.sqrt((ccx - tcx)**2 + (ccy - tcy)**2).mean())

    return avg_loss, ades_model, ades_cv


def main() -> None:
    print("Loading data...")
    train = pd.read_parquet(DATA / "train.parquet")
    dev = pd.read_parquet(DATA / "dev.parquet")
    print(f"  train: {len(train):,}   dev: {len(dev):,}")

    print("\nBuilding datasets...")
    from sequence import build_datasets
    train_ds, dev_ds = build_datasets(train, dev, REQUEST_FIELDS)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_ds, batch_size=512, shuffle=False, num_workers=0)

    model = TrajectoryLSTM(input_dim=8, hidden_dim=96, num_layers=2, num_horizons=4)
    model = model.to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {total_params:,}")

    criterion = TrajectoryLoss(horizon_weights=[1.0, 1.2, 1.5, 2.0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-5)

    best_ade = float("inf")
    best_epoch = 0
    patience = 12
    epochs_no_improve = 0

    print(f"\nTraining for up to 60 epochs (patience={patience})...")
    print(f"{'Epoch':>5} {'Train Loss':>10} {'Dev Loss':>10} "
          f"{'ADE@0.5':>8} {'ADE@1.0':>8} {'ADE@1.5':>8} {'ADE@2.0':>8} "
          f"{'Mean ADE':>8} {'CV ADE':>8} {'Best':>6}")

    for epoch in range(60):
        t0 = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, DEVICE)
        dev_loss, ades_model, ades_cv = evaluate(model, dev_loader, criterion, DEVICE)
        scheduler.step()

        mean_ade = np.mean(ades_model)
        mean_cv = np.mean(ades_cv)
        improved = "  *" if mean_ade < best_ade else ""

        if mean_ade < best_ade:
            best_ade = mean_ade
            best_epoch = epoch
            epochs_no_improve = 0
            torch.save({
                "model_state_dict": model.state_dict(),
                "hidden_dim": 96,
                "num_layers": 2,
                "input_dim": 8,
                "num_horizons": 4,
            }, MODEL_PATH)
        else:
            epochs_no_improve += 1

        elapsed = time.time() - t0
        if epoch % 5 == 0 or epochs_no_improve == 0 or epoch == 79:
            print(f"{epoch:5d} {train_loss:10.4f} {dev_loss:10.4f} "
                  f"{ades_model[0]:8.1f} {ades_model[1]:8.1f} "
                  f"{ades_model[2]:8.1f} {ades_model[3]:8.1f} "
                  f"{mean_ade:8.1f} {mean_cv:8.1f} {improved:>6}")

        if epochs_no_improve >= patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    print(f"\nBest dev mean ADE: {best_ade:.1f} px at epoch {best_epoch}")
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
