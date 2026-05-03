"""Sequence data preparation for the LSTM trajectory model.

Converts raw tracklet data into normalized tensors suitable for
training a PyTorch LSTM. Uses vectorized operations for speed.

Input format per sample:
  [16 x 8] = [x1, y1, x2, y2, ego_speed, ego_yaw, cx_norm, cy_norm]

Output format per sample:
  4 x [x1, y1, x2, y2] = ground-truth bboxes at 4 horizons

Also computes constant-velocity baselines for residual learning.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

HORIZONS_FRAMES = [8, 15, 23, 30]
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]


class TrajectoryDataset(Dataset):
    def __init__(self, sequences, targets, cv_base, augment: bool = False):
        self.sequences = sequences
        self.targets = targets
        self.cv_base = cv_base
        self.augment = augment

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = torch.tensor(self.sequences[idx], dtype=torch.float32)
        target = torch.tensor(self.targets[idx], dtype=torch.float32)
        cv_base = torch.tensor(self.cv_base[idx], dtype=torch.float32)

        if self.augment:
            noise = torch.randn_like(seq) * 0.005
            seq = seq + noise

        return seq, target, cv_base


def build_vectorized(df) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(df)
    fw = df["frame_w"].values.astype(np.float64)
    fh = df["frame_h"].values.astype(np.float64)

    hist_raw = df["bbox_history"].values
    ego_speed_raw = df["ego_speed_history"].values
    ego_yaw_raw = df["ego_yaw_history"].values

    sequences = np.zeros((n, 16, 8), dtype=np.float32)
    targets = np.zeros((n, 4, 4), dtype=np.float32)
    cv_base = np.zeros((n, 4, 4), dtype=np.float32)

    for i in range(n):
        hist = np.stack([np.asarray(r, dtype=np.float64) for r in hist_raw[i]])
        ego_speed = np.asarray(ego_speed_raw[i], dtype=np.float64)
        ego_yaw = np.asarray(ego_yaw_raw[i], dtype=np.float64)
        w = fw[i]
        h = fh[i]

        cx = (hist[:, 0] + hist[:, 2]) * 0.5
        cy = (hist[:, 1] + hist[:, 3]) * 0.5

        seq = np.zeros((16, 8), dtype=np.float32)
        seq[:, 0] = hist[:, 0] / w
        seq[:, 1] = hist[:, 1] / h
        seq[:, 2] = hist[:, 2] / w
        seq[:, 3] = hist[:, 3] / h
        ego_std = ego_speed.std()
        yaw_std = ego_yaw.std()
        seq[:, 4] = (ego_speed - ego_speed.mean()) / (ego_std + 1e-6)
        seq[:, 5] = (ego_yaw - ego_yaw.mean()) / (yaw_std + 1e-6)
        seq[:, 6] = cx / w
        seq[:, 7] = cy / h
        sequences[i] = np.nan_to_num(seq, nan=0.0, posinf=1.0, neginf=0.0)

        for j, key in enumerate(HORIZON_KEYS):
            tgt = np.asarray(df.iloc[i][key], dtype=np.float64).copy()
            tgt[0] /= w
            tgt[1] /= h
            tgt[2] /= w
            tgt[3] /= h
            targets[i, j] = tgt.astype(np.float32)

        w_last = hist[-1, 2] - hist[-1, 0]
        h_last = hist[-1, 3] - hist[-1, 1]
        vx = float(np.diff(cx[-5:]).mean()) if len(cx) >= 5 else 0.0
        vy = float(np.diff(cy[-5:]).mean()) if len(cy) >= 5 else 0.0
        for j, nf in enumerate(HORIZONS_FRAMES):
            nx = (cx[-1] + vx * nf) / w
            ny = (cy[-1] + vy * nf) / h
            cv_base[i, j] = [
                nx - w_last / (2 * w),
                ny - h_last / (2 * h),
                nx + w_last / (2 * w),
                ny + h_last / (2 * h),
            ]

    return sequences, targets, cv_base


def build_datasets(train_df, dev_df, request_fields: list[str]):
    import sys
    print(f"  Vectorizing {len(train_df)} train + {len(dev_df)} dev samples...", file=sys.stderr)

    train_seq, train_tgt, train_cv = build_vectorized(train_df)
    dev_seq, dev_tgt, dev_cv = build_vectorized(dev_df)

    train_ds = TrajectoryDataset(train_seq, train_tgt, train_cv, augment=True)
    dev_ds = TrajectoryDataset(dev_seq, dev_tgt, dev_cv, augment=False)
    return train_ds, dev_ds
