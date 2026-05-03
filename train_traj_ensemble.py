"""Ensemble trajectory model: XGBoost + LightGBM per-horizon regressors.

Averages predictions from both models for smoother, more accurate trajectories.

Usage:
    python train_traj_ensemble.py

Produces: model_traj_ensemble.pkl
"""

from __future__ import annotations

import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from train_traj_xgb import build_traj_features_df, _cv_baseline, HORIZON_KEYS, HORIZONS_FRAMES
from features import REQUEST_FIELDS

DATA = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model_traj_ensemble.pkl"


def main() -> None:
    print("Loading data...")
    train = pd.read_parquet(DATA / "train.parquet")
    dev = pd.read_parquet(DATA / "dev.parquet")

    print("Featurizing...")
    X_train = build_traj_features_df(train, REQUEST_FIELDS)
    X_dev = build_traj_features_df(dev, REQUEST_FIELDS)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=1.0, neginf=-1.0)
    X_dev = np.nan_to_num(X_dev, nan=0.0, posinf=1.0, neginf=-1.0)

    models = {}
    for j, (nf, key) in enumerate(zip(HORIZONS_FRAMES, HORIZON_KEYS)):
        print(f"\n--- {key} ---")

        y_train_cx = np.zeros(len(train))
        y_train_cy = np.zeros(len(train))

        for i in range(len(train)):
            tgt = np.asarray(train.iloc[i][key], dtype=np.float64)
            tcx = (tgt[0] + tgt[2]) * 0.5
            tcy = (tgt[1] + tgt[3]) * 0.5
            cv = _cv_baseline({k: train.iloc[i][k] for k in REQUEST_FIELDS})[j]
            y_train_cx[i] = tcx - cv["cx"]
            y_train_cy[i] = tcy - cv["cy"]

        y_dev_cx = np.zeros(len(dev))
        y_dev_cy = np.zeros(len(dev))
        for i in range(len(dev)):
            tgt = np.asarray(dev.iloc[i][key], dtype=np.float64)
            tcx = (tgt[0] + tgt[2]) * 0.5
            tcy = (tgt[1] + tgt[3]) * 0.5
            cv = _cv_baseline({k: dev.iloc[i][k] for k in REQUEST_FIELDS})[j]
            y_dev_cx[i] = tcx - cv["cx"]
            y_dev_cy[i] = tcy - cv["cy"]

        xgb_cx = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.03,
                               min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                               tree_method="hist", n_jobs=-1)
        xgb_cy = XGBRegressor(n_estimators=600, max_depth=6, learning_rate=0.03,
                               min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                               tree_method="hist", n_jobs=-1)
        lgbm_cx = LGBMRegressor(n_estimators=600, max_depth=6, learning_rate=0.03,
                                 num_leaves=48, min_child_samples=10,
                                 subsample=0.8, colsample_bytree=0.8,
                                 n_jobs=-1, verbose=-1)
        lgbm_cy = LGBMRegressor(n_estimators=600, max_depth=6, learning_rate=0.03,
                                 num_leaves=48, min_child_samples=10,
                                 subsample=0.8, colsample_bytree=0.8,
                                 n_jobs=-1, verbose=-1)

        t0 = time.time()
        xgb_cx.fit(X_train, y_train_cx, eval_set=[(X_dev, y_dev_cx)], verbose=False)
        xgb_cy.fit(X_train, y_train_cy, eval_set=[(X_dev, y_dev_cy)], verbose=False)
        lgbm_cx.fit(X_train, y_train_cx, eval_set=[(X_dev, y_dev_cx)])
        lgbm_cy.fit(X_train, y_train_cy, eval_set=[(X_dev, y_dev_cy)])
        print(f"  Trained in {time.time()-t0:.1f}s")

        pred_cx = (xgb_cx.predict(X_dev) + lgbm_cx.predict(X_dev)) / 2
        pred_cy = (xgb_cy.predict(X_dev) + lgbm_cy.predict(X_dev)) / 2

        ades = []
        for i in range(len(dev)):
            cv = _cv_baseline({k: dev.iloc[i][k] for k in REQUEST_FIELDS})[j]
            pcx = cv["cx"] + pred_cx[i]
            pcy = cv["cy"] + pred_cy[i]
            tgt = np.asarray(dev.iloc[i][key], dtype=np.float64)
            tcx = (tgt[0] + tgt[2]) * 0.5
            tcy = (tgt[1] + tgt[3]) * 0.5
            ades.append(np.sqrt((pcx-tcx)**2 + (pcy-tcy)**2))
        print(f"  Ensemble ADE: {np.mean(ades):.1f} px")

        models[f"{key}_xgb_cx"] = xgb_cx
        models[f"{key}_xgb_cy"] = xgb_cy
        models[f"{key}_lgbm_cx"] = lgbm_cx
        models[f"{key}_lgbm_cy"] = lgbm_cy

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(models, f)
    print(f"\nSaved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
