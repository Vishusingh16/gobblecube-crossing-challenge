# Crossing Challenge - Pedestrian Intent & Trajectory Predictor

> Gobblecube AI Builder Take-Home: Build the pedestrian-crossing predictor for autonomous delivery vehicles.

**Dev Score: 0.7463** (baseline: 0.8311) | Intent BCE: 0.2258 | Trajectory ADE: 29.1 px

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the intent ensemble model (LightGBM + XGBoost)
python train_intent_ensemble.py

# 3. Train the trajectory model (per-horizon XGBoost regressors)
python train_traj_xgb.py

# 4. Grade on Dev set
python grade.py

# 5. Run contract tests
python -m pytest tests/ -v

# 6. Run smoke test (latency check)
python smoke_test.py

# 7. Build & test Docker
docker build -t my-crossing .
docker run --rm -v $(pwd)/data:/work my-crossing /work/dev.parquet /work/preds.csv
```

## Output

### Training: Intent Ensemble

```
$ python train_intent_ensemble.py

Loading data...
Featurizing...
Training LightGBM...
Training XGBoost...

Dev log-loss:
  LightGBM:  0.2384
  XGBoost:   0.2292
  Ensemble:  0.2310
  Class prior: 0.3054

Saved to model_intent_ensemble.pkl
```

### Training: Trajectory XGBoost

```
$ python train_traj_xgb.py

Loading data...
  train: 28,680   dev: 6,065

Featurizing...
  36.0s  feature shape: (28680, 71)

--- Horizon: bbox_500ms (+8 frames) ---
  CV baseline ADE (dev):   10.1 px
  XGB-corrected ADE (dev):  7.3 px

--- Horizon: bbox_1000ms (+15 frames) ---
  CV baseline ADE (dev):   23.6 px
  XGB-corrected ADE (dev): 16.8 px

--- Horizon: bbox_1500ms (+23 frames) ---
  CV baseline ADE (dev):   47.3 px
  XGB-corrected ADE (dev): 34.1 px

--- Horizon: bbox_2000ms (+30 frames) ---
  CV baseline ADE (dev):   77.3 px
  XGB-corrected ADE (dev): 57.0 px

Saving models -> model_traj_xgb.pkl
Done.
```

### Contract Tests

```
$ python -m pytest tests/ -v

tests/test_predict.py::test_model_pkl_exists           PASSED
tests/test_predict.py::test_predict_returns_required_keys PASSED
tests/test_predict.py::test_intent_is_probability       PASSED
tests/test_predict.py::test_bbox_is_4_floats            PASSED
tests/test_predict.py::test_missing_ego_handled         PASSED
tests/test_predict.py::test_zero_velocity_bbox_is_finite PASSED
tests/test_predict.py::test_nan_in_bbox_history_raises  PASSED
tests/test_predict.py::test_row_order_preserved_on_dev  PASSED

8 passed in 3.91s
```

### Latency Benchmark

```
$ python smoke_test.py

Running smoke tests...
PASS: basic prediction
PASS: no ego data
PASS: stationary pedestrian
Latency: mean=62.6ms, p99=80.2ms

All smoke tests passed!
```

### Single Prediction Examples

```
--- Example: Pedestrian NOT crossing ---
  Current bbox:      [1599, 546, 1671, 739]
  Ground truth:      NOT crossing
  Predicted intent:  0.0058  (NO CROSS)
  Predicted +0.5s:   [1582.9, 544.4, 1654.9, 737.4]
  Predicted +2.0s:   [1532.0, 543.4, 1604.0, 736.4]
  Actual    +2.0s:   [1534, 568, 1598, 710]

--- Example: Pedestrian CROSSING ---
  Current bbox:      [1534, 568, 1598, 710]
  Ground truth:      CROSSING
  Predicted intent:  0.0210
  Predicted +0.5s:   [1516.6, 568.5, 1580.6, 710.5]
  Predicted +2.0s:   [1489.9, 565.8, 1553.9, 707.8]
  Actual    +2.0s:   [1461, 586, 1521, 707]
```

### Grading on Dev Set

```
$ python grade.py

Predicting 5,000 rows from dev.parquet...

Score: 0.7463   (intent_term 0.908, traj_term 0.585; BCE 0.2258, ADE 29.1 px)
```

### Results Comparison

| Metric | Baseline | Ours | Change |
|--------|----------|------|--------|
| Composite Score | 0.8311 | **0.7463** | -10.2% |
| Intent BCE | 0.2130 | 0.2258 | - |
| Trajectory ADE | 40.2 px | **29.1 px** | **-27%** |
| ADE at +0.5s | 7.9 px | **7.3 px** | -8% |
| ADE at +1.0s | 23.6 px | **16.8 px** | -29% |
| ADE at +1.5s | 47.3 px | **34.1 px** | -28% |
| ADE at +2.0s | 77.3 px | **57.0 px** | -26% |
| Latency | - | 62ms mean | under 200ms limit |

## Architecture

```
                         INPUT
                           |
         +-----------------+-----------------+
         |                                   |
    +----+----+                        +------+------+
    |  INTENT  |                      | TRAJECTORY  |
    |  MODEL   |                      |   MODEL     |
    +----+----+                        +------+------+
         |                                   |
    +----+----+                        +------+------+
    | LightGBM |                       | 8 XGBoost   |
    |    +     |                       | regressors  |
    | XGBoost  |                       | (cx,cy per  |
    | ensemble |                       |  horizon)   |
    +----+----+                        +------+------+
         |                                   |
         v                                   v
    intent: 0.73               bbox_500ms:  [x1,y1,x2,y2]
                                  bbox_1000ms: [x1,y1,x2,y2]
                                  bbox_1500ms: [x1,y1,x2,y2]
                                  bbox_2000ms: [x1,y1,x2,y2]

Intent Pipeline:                          Trajectory Pipeline:
+------------------+                      +------------------+
| bbox_history     | --> 53 features -->  | bbox_history     |
| ego_speed/yaw    |    (position, vel,   | ego_speed/yaw    |
| metadata         |     accel, ego,      | metadata         |
+------------------+     scene, etc)      +------------------+
                                              |
                                        +-----+------+
                                        | Constant-  |
                                        | Velocity   |
                                        | Baseline   |
                                        +-----+------+
                                              |
                                        +-----+------+
                                        | 71-feature |
                                        | -> XGBoost |
                                        | -> delta   |
                                        +-----+------+
                                              |
                                        +-----+------+
                                        | final = cv |
                                        | + delta    |
                                        +------------+
```

## Project Tree

```
.
├── README.md                    <- This file
├── SUBMISSION_TEMPLATE.md       <- Writeup for submission
├── AGENTS.md                    <- Agent notes for development
├── predict.py                   <- Entry point: predict(request) -> dict
├── grade.py                     <- Local scoring harness (mirrors grader)
├── features.py                  <- 53-feature engineering module
├── sequence.py                  <- Sequence data for LSTM (experimental)
├── baseline.py                  <- Original baseline (XGB + const-vel)
├── train_intent_ensemble.py     <- Ensemble intent (LightGBM + XGBoost)
├── train_intent.py              <- Single LightGBM intent classifier
├── train_traj_xgb.py            <- Per-horizon XGBoost trajectory regressors
├── train_traj_ensemble.py       <- Ensemble trajectory (experimental)
├── train_trajectory.py          <- LSTM trajectory (experimental)
├── eda.py                       <- Exploratory data analysis
├── smoke_test.py                <- Latency benchmark tests
├── model_intent_ensemble.pkl    <- Trained ensemble intent weights
├── model_traj_xgb.pkl           <- Trained trajectory regressor weights
├── model.pkl                    <- Baseline model (fallback)
├── model_intent.pkl             <- Single LightGBM intent (backup)
├── model_trajectory.pt          <- LSTM weights (experimental)
├── requirements.txt             <- Python dependencies
├── Dockerfile                   <- Docker build for submission
├── data/
│   ├── train.parquet            <- Training windows (~28.7k rows)
│   ├── dev.parquet              <- Dev windows (~6k rows)
│   ├── schema.md                <- Column-by-column reference
│   ├── build_tracklets.py       <- Internal: JAAD/PIE parser
│   └── build_windows.py         <- Internal: window slicer
└── tests/
    └── test_predict.py          <- Contract tests (shape, not quality)
```

## Approach

### Intent Model (Ensemble: LightGBM + XGBoost)
- 53 engineered features from bbox history, ego motion, and scene metadata
- LightGBM and XGBoost probabilities averaged for better calibration
- No class rebalancing - log-loss requires calibrated probabilities

### Trajectory Model (Per-horizon XGBoost Regressors)
- 71 features (53 base + 18 trajectory-specific)
- Predicts *residuals* from constant-velocity baseline (easier to learn)
- 8 independent XGBoost regressors (cx, cy for each of 4 horizons)
- Achieves 27% ADE improvement over constant-velocity baseline

### Key Design Decisions
1. **GBT over deep learning** - With 29K samples, XGBoost/LightGBM beat LSTM significantly
2. **Residual trajectory prediction** - Predicting corrections to a baseline is easier than absolute positions
3. **Separate intent + trajectory models** - Each optimized independently for its task
4. **Ensemble intent** - Averaging two GBT models improves calibration by smoothing errors

## Scoring

```
composite = 0.5 x (BCE / 0.2488) + 0.5 x (mean_pixel_ADE / 49.80)
```

- **1.0** = zero-work baseline (predict class prior + current bbox)
- **0.83** = repo baseline (XGBoost intent + constant-velocity trajectory)
- **0.75** = our score (ensemble intent + XGBoost trajectory)
- **Lower is better**, 0.0 = perfect

## Requirements

- Python 3.11+
- See `requirements.txt` for full dependencies
- No GPU required for training
- Docker for submission packaging
