"""Train enhanced LightGBM intent classifier.

Uses 47 engineered features from features.py. Handles 7% positive class
with scale_pos_weight for calibrated probabilities.

Usage:
    python train_intent.py

Produces: model_intent.pkl
"""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier

from features import extract_features, featurize_df, REQUEST_FIELDS

DATA = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model_intent.pkl"


def main() -> None:
    print("Loading train + dev...")
    train = pd.read_parquet(DATA / "train.parquet")
    dev = pd.read_parquet(DATA / "dev.parquet")
    print(f"  train: {len(train):,}   dev: {len(dev):,}")
    print(f"  positive rates: train {train.will_cross_2s.mean():.3f}, "
          f"dev {dev.will_cross_2s.mean():.3f}")

    print("\nFeaturizing...")
    t0 = time.time()
    X_train = featurize_df(train, REQUEST_FIELDS)
    X_dev = featurize_df(dev, REQUEST_FIELDS)
    y_train = train["will_cross_2s"].to_numpy(dtype=np.int32)
    y_dev = dev["will_cross_2s"].to_numpy(dtype=np.int32)
    print(f"  {time.time() - t0:.1f}s  feature shape: {X_train.shape}")

    pos_ratio = float(y_train.mean())
    neg_pos_ratio = (1 - pos_ratio) / pos_ratio
    print(f"  Class ratio: {neg_pos_ratio:.1f}:1 (neg:pos)")

    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
    X_dev = np.nan_to_num(X_dev, nan=0.0, posinf=1.0, neginf=-1.0)

    print("\nTraining LightGBM classifier...")
    t0 = time.time()
    clf = LGBMClassifier(
        n_estimators=1200,
        max_depth=6,
        learning_rate=0.02,
        num_leaves=48,
        min_child_samples=15,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=1.0,
        objective="binary",
        metric="binary_logloss",
        n_jobs=-1,
        verbose=-1,
    )
    clf.fit(
        X_train, y_train,
        eval_set=[(X_dev, y_dev)],
        callbacks=[],
    )
    print(f"  {time.time() - t0:.1f}s")

    dev_probs = clf.predict_proba(X_dev)[:, 1]
    dev_probs = np.clip(dev_probs, 1e-6, 1 - 1e-6)
    ll = log_loss(y_dev, dev_probs)
    prior_ll = log_loss(y_dev, np.full_like(dev_probs, pos_ratio))
    print(f"\nDev log-loss:  {ll:.4f}  (class-prior baseline {prior_ll:.4f})")
    print(f"  Improvement: {(1 - ll/prior_ll)*100:.1f}% over class prior")

    n_correct = ((dev_probs > 0.5) == y_dev).mean()
    print(f"  Accuracy: {n_correct:.4f}")
    print(f"  Predicted positive rate: {(dev_probs > 0.5).mean():.4f}")
    print(f"  Mean predicted prob: {dev_probs.mean():.4f}")

    importances = clf.feature_importances_
    top_idx = np.argsort(importances)[-10:][::-1]
    print(f"\nTop 10 features:")
    for i in top_idx:
        print(f"  feat_{i:2d}: {importances[i]:6.0f}")

    print(f"\nSaving model -> {MODEL_PATH}")
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(clf, f)

    print("Done.")


if __name__ == "__main__":
    main()
