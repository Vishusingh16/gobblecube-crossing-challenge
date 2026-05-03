# Submission Writeup

## Final score

Dev composite score: **0.7463** (intent_term 0.908, traj_term 0.585; BCE 0.2258, ADE 29.1 px).

Baseline was 0.8311 on Dev (XGBoost intent + constant-velocity trajectory).

## Approach

I built a two-model pipeline:

1. **Intent (Ensemble LightGBM + XGBoost):** 53 engineered features from bbox history (position, velocity, acceleration, jerk, direction, aspect ratio), ego vehicle motion (speed, yaw, interaction terms), scene metadata, and derived features (road proximity, deceleration detection, approach detection). The ensemble averages probabilities from LightGBM (1200 trees, depth 6, L2 regularized) and XGBoost (800 trees, depth 5). No class rebalancing - log-loss needs calibrated probabilities, and the ~8% positive rate is handled naturally.

2. **Trajectory (per-horizon XGBoost center regressors):** 71 features (53 base + 18 trajectory-specific including acceleration, direction angle, speed-velocity interactions). Each horizon gets independent cx/cy regressors that predict *residuals* from a constant-velocity baseline. This residual learning approach makes convergence much faster than predicting absolute positions.

Key insight: the constant-velocity baseline is already decent at short horizons (10px at 0.5s) but degrades badly at long horizons (77px at 2s). The XGBoost regressors learn to correct for acceleration, direction changes, and "pedestrian changed mind" patterns.

## What didn't work

1. **LSTM trajectory model** - Built a BiLSTM with attention pooling (~389K params), trained for 60 epochs. Only matched constant-velocity on dev (39.2px vs 39.6px). The dataset is too small (29K) for a deep model to outperform well-engineered tabular features with GBTs. Classical ML wins here.

2. **scale_pos_weight for intent** - Setting class weight to the inverse ratio (11.6:1) made the model over-predict positives, wrecking calibration. Log-loss penalizes overconfident wrong predictions hard. Removing it improved BCE from 0.30 to 0.24.

3. **Trajectory ensemble (XGB + LightGBM)** - Averaging XGBoost and LightGBM trajectory predictions only improved ADE by ~0.3px but doubled inference time from ~100ms to ~200ms per request. Not worth the cost.

## Where AI tooling sped me up most

Claude Code (opencode) was used for:
- **Architecture design:** Translating the problem spec into a feature engineering + GBT pipeline
- **Data exploration:** Rapid EDA to understand distributions, class imbalance, and baseline weaknesses
- **Debugging:** Parquet array-of-arrays parsing, feature shape mismatches between training and inference
- **Iteration speed:** Trying multiple approaches (LSTM, XGB regressors, ensembles) quickly by modifying generated code

Where it fell short: the LSTM trajectory approach was suggested by the AI but turned out to be wrong for this dataset size. Domain judgment about when tabular beats deep learning was needed.

## Next experiments

1. **Position-aware trajectory features:** Add features encoding the pedestrian's distance to the road edge (bottom of frame), which is a strong signal for crossing intent and also affects trajectory.
2. **Multi-output regressor:** Train a single model predicting all 8 values (4 horizons x cx/cy) jointly instead of 8 independent models, so it can learn horizon correlations.
3. **Platt scaling / isotonic regression** for intent calibration on dev set to squeeze the last BCE points.
4. **Larger LSTM with data augmentation:** Random rotation/translation of bbox sequences could artificially inflate the training set to make deep learning viable.

## How to reproduce

```bash
pip install -r requirements.txt

# Train intent ensemble (LightGBM + XGBoost)
python train_intent_ensemble.py

# Train trajectory (per-horizon XGBoost regressors)
python train_traj_xgb.py

# Grade on dev
python grade.py

# Run contract tests
python -m pytest tests/ -v

# Docker build + test
docker build -t my-crossing .
docker run --rm -v $(pwd)/data:/work my-crossing /work/dev.parquet /work/preds.csv
```

## External data / pretrained weights

No external data or pretrained weights used. Only `data/train.parquet` and `data/dev.parquet` from the starter repo.

---

Total time spent: ~8 hours.
