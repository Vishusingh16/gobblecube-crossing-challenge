# AGENTS.md

## Commands

- Run tests: `python -m pytest tests/ -v`
- Validate submission: `python validate_submission.py`
- Train intent ensemble: `python train_intent_ensemble.py`
- Train trajectory: `python train_traj_xgb.py`
- Grade on dev: `python grade.py`
- Run EDA: `python eda.py`
- Latency check: `python smoke_test.py`

## Final Architecture

- `features.py`: 53-feature engineering (position, velocity, acceleration, ego interaction, scene)
- `train_intent_ensemble.py`: Ensemble LightGBM + XGBoost intent classifier
- `train_traj_xgb.py`: Per-horizon XGBoost center regressors (residual from CV baseline)
- `predict.py`: Inference entry point (loads ensemble intent + XGBoost trajectory)
- `sequence.py`: Sequence builder for LSTM (unused in final, kept for experiments)
- `train_trajectory.py`: LSTM trajectory model (unused in final, kept for experiments)

## Constraints

- Docker image <= 2GB
- Inference <= 200ms per request on CPU
- 4GB RAM / 4 CPUs at scoring
- No external API calls at inference
- `--network=none` during Docker scoring

## Scoring

- Composite = 0.5 * (BCE / 0.2488) + 0.5 * (ADE / 49.80)
- Baseline: 0.74 on Eval, 0.83 on Dev
- Our score: **0.7463** on Dev
- Lower is better

## What Failed

1. LSTM trajectory (dataset too small, 29K rows)
2. scale_pos_weight for intent (broke calibration)
3. Trajectory ensemble XGB+LGBM (marginal gain, 2x slower)

## Next Steps

1. Platt scaling for intent calibration
2. Multi-output trajectory regressor
3. Road-edge distance features
4. Data augmentation for LSTM
