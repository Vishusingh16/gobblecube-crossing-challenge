"""Train per-horizon XGBoost trajectory regressors.

Predicts bbox center deltas from constant-velocity baseline using
gradient boosted trees. Often outperforms small LSTMs on tabular features.

Usage:
    python train_traj_xgb.py

Produces: model_traj_xgb.pkl
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from features import extract_features, featurize_df, REQUEST_FIELDS

DATA = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model_traj_xgb.pkl"
HORIZONS_FRAMES = [8, 15, 23, 30]
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]


def _as_2d(x) -> np.ndarray:
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])


def _cv_baseline(req: dict) -> list[dict]:
    hist = _as_2d(req["bbox_history"])
    fw = float(req["frame_w"])
    fh = float(req["frame_h"])
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w_last = hist[-1, 2] - hist[-1, 0]
    h_last = hist[-1, 3] - hist[-1, 1]
    vx = float(np.diff(cx[-5:]).mean()) if len(cx) >= 5 else 0.0
    vy = float(np.diff(cy[-5:]).mean()) if len(cy) >= 5 else 0.0

    bboxes = []
    for nf in HORIZONS_FRAMES:
        nx = cx[-1] + vx * nf
        ny = cy[-1] + vy * nf
        bboxes.append({
            "cx": nx, "cy": ny,
            "x1": nx - w_last / 2, "y1": ny - h_last / 2,
            "x2": nx + w_last / 2, "y2": ny + h_last / 2,
            "w": w_last, "h": h_last,
        })
    return bboxes


def build_trajectory_features(req: dict) -> np.ndarray:
    base_feats = extract_features(req)
    hist = _as_2d(req["bbox_history"])
    fw = float(req["frame_w"])
    fh = float(req["frame_h"])
    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    vx = np.diff(cx)
    vy = np.diff(cy)

    extra = [
        cx[-1] / fw,
        cy[-1] / fh,
        vx[-1] / fw,
        vy[-1] / fh,
        vx[-4:].mean() / fw,
        vy[-4:].mean() / fh,
        vx[-8:].mean() / fw,
        vy[-8:].mean() / fh,
        (hist[-1, 2] - hist[-1, 0]) / fw,
        (hist[-1, 3] - hist[-1, 1]) / fh,
        np.diff(vx[-4:]).mean() / fw if len(vx) >= 4 else 0.0,
        np.diff(vy[-4:]).mean() / fh if len(vy) >= 4 else 0.0,
        np.sqrt(vx[-4:].mean()**2 + vy[-4:].mean()**2) / fw,
        np.arctan2(vy[-4:].mean(), vx[-4:].mean() + 1e-8),
        (vx[-1] - vx[-4:].mean()) / (fw + 1e-6),
        (vy[-1] - vy[-4:].mean()) / (fh + 1e-6),
        cx[-1] * vy[-4:].mean() / (fw * fh),
        cy[-1] * vx[-4:].mean() / (fw * fh),
    ]
    return np.concatenate([base_feats, np.asarray(extra, dtype=np.float32)])


def build_traj_features_df(df, request_fields: list[str]) -> np.ndarray:
    n = len(df)
    sample = build_trajectory_features({k: df.iloc[0][k] for k in request_fields})
    X = np.empty((n, len(sample)), dtype=np.float32)
    X[0] = sample
    for i in range(1, n):
        X[i] = build_trajectory_features({k: df.iloc[i][k] for k in request_fields})
    return X


def main() -> None:
    print("Loading data...")
    train = pd.read_parquet(DATA / "train.parquet")
    dev = pd.read_parquet(DATA / "dev.parquet")
    print(f"  train: {len(train):,}   dev: {len(dev):,}")

    print("\nFeaturizing...")
    t0 = time.time()
    X_train = build_traj_features_df(train, REQUEST_FIELDS)
    X_dev = build_traj_features_df(dev, REQUEST_FIELDS)
    print(f"  {time.time() - t0:.1f}s  feature shape: {X_train.shape}")

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
    X_dev = np.nan_to_num(X_dev, nan=0.0, posinf=1.0, neginf=-1.0)

    models = {}
    for j, (nf, key) in enumerate(zip(HORIZONS_FRAMES, HORIZON_KEYS)):
        print(f"\n--- Horizon: {key} (+{nf} frames) ---")

        y_train_cx = np.zeros(len(train))
        y_train_cy = np.zeros(len(train))
        cv_ade_train = np.zeros(len(train))

        y_dev_cx = np.zeros(len(dev))
        y_dev_cy = np.zeros(len(dev))
        cv_ade_dev = np.zeros(len(dev))

        for i in range(len(train)):
            tgt = np.asarray(train.iloc[i][key], dtype=np.float64)
            tcx = (tgt[0] + tgt[2]) * 0.5
            tcy = (tgt[1] + tgt[3]) * 0.5
            cv = _cv_baseline({k: train.iloc[i][k] for k in REQUEST_FIELDS})[j]
            y_train_cx[i] = tcx - cv["cx"]
            y_train_cy[i] = tcy - cv["cy"]
            cv_ade_train[i] = np.sqrt((cv["cx"] - tcx)**2 + (cv["cy"] - tcy)**2)

        for i in range(len(dev)):
            tgt = np.asarray(dev.iloc[i][key], dtype=np.float64)
            tcx = (tgt[0] + tgt[2]) * 0.5
            tcy = (tgt[1] + tgt[3]) * 0.5
            cv = _cv_baseline({k: dev.iloc[i][k] for k in REQUEST_FIELDS})[j]
            y_dev_cx[i] = tcx - cv["cx"]
            y_dev_cy[i] = tcy - cv["cy"]
            cv_ade_dev[i] = np.sqrt((cv["cx"] - tcx)**2 + (cv["cy"] - tcy)**2)

        print(f"  CV baseline ADE (train): {cv_ade_train.mean():.1f} px")
        print(f"  CV baseline ADE (dev):   {cv_ade_dev.mean():.1f} px")

        reg_cx = XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.03,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", n_jobs=-1,
        )
        reg_cy = XGBRegressor(
            n_estimators=600, max_depth=6, learning_rate=0.03,
            min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
            tree_method="hist", n_jobs=-1,
        )

        t0 = time.time()
        reg_cx.fit(X_train, y_train_cx, eval_set=[(X_dev, y_dev_cx)], verbose=False)
        reg_cy.fit(X_train, y_train_cy, eval_set=[(X_dev, y_dev_cy)], verbose=False)
        print(f"  Training: {time.time() - t0:.1f}s")

        pred_cx = reg_cx.predict(X_dev)
        pred_cy = reg_cy.predict(X_dev)

        for i in range(len(dev)):
            cv = _cv_baseline({k: dev.iloc[i][k] for k in REQUEST_FIELDS})[j]
            pcx = cv["cx"] + pred_cx[i]
            pcy = cv["cy"] + pred_cy[i]
            tgt = np.asarray(dev.iloc[i][key], dtype=np.float64)
            tcx = (tgt[0] + tgt[2]) * 0.5
            tcy = (tgt[1] + tgt[3]) * 0.5
            cv_ade_dev[i] = np.sqrt((pcx - tcx)**2 + (pcy - tcy)**2)

        print(f"  XGB-corrected ADE (dev): {cv_ade_dev.mean():.1f} px")

        models[f"{key}_cx"] = reg_cx
        models[f"{key}_cy"] = reg_cy

    print(f"\nSaving models -> {MODEL_PATH}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(models, f)
    print("Done.")


if __name__ == "__main__":
    main()
