# CLAUDE.md

## Project Overview

This is the **Crossing Challenge** from Gobblecube's AI Builder hiring process.

**Goal:** Predict pedestrian crossing intent (probability) and future bounding box trajectory (4 horizons) from a 1-second observation window.

**Score:** 0.7463 on Dev (baseline: 0.8311). Lower is better.

## Commands

```bash
python -m pytest tests/ -v          # Run contract tests
python train_intent_ensemble.py     # Train intent (LightGBM + XGBoost ensemble)
python train_traj_xgb.py            # Train trajectory (8 XGBoost regressors)
python grade.py                     # Score on dev set
python smoke_test.py                # Latency benchmark
python eda.py                       # Data exploration
```

## Architecture

- `features.py` - 53-feature engineering module (used by both training and inference)
- `train_intent_ensemble.py` - Ensemble LightGBM + XGBoost intent classifier
- `train_traj_xgb.py` - Per-horizon XGBoost trajectory regressors (residual from CV baseline)
- `predict.py` - Inference entry point, loads both models
- `model_intent_ensemble.pkl` - Trained ensemble intent weights
- `model_traj_xgb.pkl` - Trained trajectory regressor weights

## Key Design Decisions

- GBT over deep learning (dataset too small for LSTM, 29K rows)
- Residual trajectory learning (predict corrections to constant-velocity baseline)
- No class rebalancing for intent (log-loss needs calibrated probabilities)
- Ensemble intent (averaging LightGBM + XGBoost improves calibration)

## Constraints

- Docker image <= 2GB
- Inference <= 200ms per request on CPU
- 4GB RAM / 4 CPUs at scoring
- No external API calls at inference
- `--network=none` during Docker scoring

## Scoring Formula

```
composite = 0.5 * (BCE / 0.2488) + 0.5 * (mean_pixel_ADE / 49.80)
```

## What Failed

1. LSTM trajectory (39.2px vs CV 39.6px - dataset too small)
2. scale_pos_weight=11.6 (broke calibration, BCE 0.30)
3. Trajectory ensemble XGB+LGBM (only 0.3px gain, 2x slower)
4. Raw position prediction (residual learning works much better)

## Next Experiments

1. Platt scaling for intent calibration
2. Multi-output trajectory regressor (joint horizons)
3. Road-edge distance features
4. Data augmentation for LSTM viability
