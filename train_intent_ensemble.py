"""Ensemble intent model combining LightGBM and XGBoost predictions.

Uses simple average of predicted probabilities for better calibration.

Usage:
    python train_intent_ensemble.py

Produces: model_intent_ensemble.pkl
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from features import extract_features, featurize_df, REQUEST_FIELDS

DATA = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model_intent_ensemble.pkl"


def main() -> None:
    print("Loading data...")
    train = pd.read_parquet(DATA / "train.parquet")
    dev = pd.read_parquet(DATA / "dev.parquet")

    print("Featurizing...")
    X_train = featurize_df(train, REQUEST_FIELDS)
    X_dev = featurize_df(dev, REQUEST_FIELDS)
    y_train = train["will_cross_2s"].to_numpy(dtype=np.int32)
    y_dev = dev["will_cross_2s"].to_numpy(dtype=np.int32)
    pos_ratio = float(y_train.mean())

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
    X_dev = np.nan_to_num(X_dev, nan=0.0, posinf=1.0, neginf=-1.0)

    print("Training LightGBM...")
    lgbm = LGBMClassifier(
        n_estimators=1200, max_depth=6, learning_rate=0.02,
        num_leaves=48, min_child_samples=15,
        subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        objective="binary", n_jobs=-1, verbose=-1,
    )
    lgbm.fit(X_train, y_train)

    print("Training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=800, max_depth=5, learning_rate=0.03,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=1.0,
        eval_metric="logloss", n_jobs=-1,
    )
    xgb.fit(X_train, y_train, eval_set=[(X_dev, y_dev)], verbose=False)

    lgbm_probs = np.clip(lgbm.predict_proba(X_dev)[:, 1], 1e-6, 1 - 1e-6)
    xgb_probs = np.clip(xgb.predict_proba(X_dev)[:, 1], 1e-6, 1 - 1e-6)
    ens_probs = (lgbm_probs + xgb_probs) / 2

    ll_lgbm = log_loss(y_dev, lgbm_probs)
    ll_xgb = log_loss(y_dev, xgb_probs)
    ll_ens = log_loss(y_dev, ens_probs)
    prior_ll = log_loss(y_dev, np.full_like(ens_probs, pos_ratio))

    print(f"\nDev log-loss:")
    print(f"  LightGBM:  {ll_lgbm:.4f}")
    print(f"  XGBoost:   {ll_xgb:.4f}")
    print(f"  Ensemble:  {ll_ens:.4f}")
    print(f"  Class prior: {prior_ll:.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"lgbm": lgbm, "xgb": xgb}, f)
    print(f"\nSaved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
