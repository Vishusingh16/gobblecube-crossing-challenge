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

# 5. Grade on Dev set
python grade.py

# 6. Run contract tests
python -m pytest tests/ -v

# 7. Build & test Docker
docker build -t my-crossing .
docker run --rm -v $(pwd)/data:/work my-crossing /work/dev.parquet /work/preds.csv
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INPUT (16 frames)                        │
│     bbox_history [16x4]  +  ego_speed [16]  +  ego_yaw [16]    │
│                    + metadata (time, weather, location)          │
└─────────────┬───────────────────────────────┬───────────────────┘
              │                               │
              ▼                               ▼
   ┌─────────────────────┐         ┌─────────────────────────┐
   │   FEATURE ENGINE    │         │   SEQUENCE ENCODER      │
   │   (40+ features)    │         │   (PyTorch LSTM)        │
   │                     │         │                         │
   │  - Position (norm)  │         │  Input: [16 x 8]        │
   │  - Velocity (vx,vy) │         │  [x1,y1,x2,y2,         │
   │  - Acceleration     │         │   ego_spd,ego_yaw,      │
   │  - Jerk             │         │   cx_norm,cy_norm]      │
   │  - Aspect ratio     │         │                         │
   │  - Ego interaction  │         │  2-layer BiLSTM         │
   │  - Scene flags      │         │  hidden=128             │
   │  - Velocity stats   │         │  + attention pooling    │
   └──────────┬──────────┘         └──────────┬──────────────┘
              │                               │
              ▼                               ▼
   ┌─────────────────────┐         ┌─────────────────────────┐
   │   LightGBM          │         │   TRAJECTORY HEAD (MLP) │
   │   CLASSIFIER        │         │                         │
   │                     │         │  last_hidden -> MLP     │
   │  - 500 trees        │         │  -> 4 x [dcx,dcy,dw,dh]│
   │  - depth 7          │         │  (residual from const-  │
   │  - class weighted   │         │   velocity baseline)    │
   │  - calibrated probs │         │                         │
   └──────────┬──────────┘         └──────────┬──────────────┘
              │                               │
              ▼                               ▼
   ┌─────────────────────┐         ┌─────────────────────────┐
   │  intent: P(cross)   │         │  bbox_500ms  [x1,y1,    │
   │  float in [0, 1]    │         │  bbox_1000ms   x2,y2]   │
   └─────────────────────┘         │  bbox_1500ms            │
                                   │  bbox_2000ms            │
                                   └─────────────────────────┘
```

## Project Tree

```
crossing-challenge-starter/
├── README.md                    <- Architecture + startup guide (this file)
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
crossing-challenge-starter/
├── README.md                    <- Architecture + startup guide
├── SUBMISSION_TEMPLATE.md       <- Writeup for submission
├── predict.py                   <- Entry point: predict(request) -> dict
├── grade.py                     <- Local scoring harness (mirrors grader)
├── baseline.py                  <- Original baseline (XGB + const-vel)
├── features.py                  <- Enhanced feature engineering module
├── sequence.py                  <- Sequence data preparation for LSTM
├── train_intent.py              <- LightGBM intent classifier training
├── train_trajectory.py          <- LSTM trajectory model training
├── eda.py                       <- Exploratory data analysis
├── model.pkl                    <- Trained model weights (built by train_*)
├── requirements.txt             <- Python dependencies
├── Dockerfile                   <- Docker build for submission
├── AGENTS.md                    <- Agent notes for development
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

### Intent Model (LightGBM)
- 40+ engineered features from bbox history, ego motion, and scene metadata
- Handles 7% positive class with `scale_pos_weight`
- Produces well-calibrated probabilities for BCE scoring

### Trajectory Model (LSTM)
- Bidirectional LSTM encodes the 16-frame bbox + ego sequence
- Predicts *residuals* from constant-velocity (easier to learn)
- Multi-horizon MLP head outputs 4 bounding boxes at +0.5s, +1s, +1.5s, +2s
- Trained with MSE loss on bbox coordinates

### Key Design Decisions
1. **Residual trajectory prediction** - LSTM predicts deltas from constant-velocity, not absolute positions. Makes convergence much faster.
2. **Separate models** - Intent (tabular GBT) and trajectory (LSTM) are trained independently for robustness.
3. **CPU-only deployment** - PyTorch CPU build keeps Docker image under 2GB.

## Scoring

```
composite = 0.5 x (BCE / 0.2488) + 0.5 x (mean_pixel_ADE / 49.80)
```

- **1.0** = zero-work baseline (predict class prior + current bbox)
- **0.74** = repo baseline (XGBoost intent + constant-velocity trajectory)
- **Lower is better**, 0.0 = perfect

## Requirements

- Python 3.11+
- See `requirements.txt` for full dependencies
- No GPU required for training (LSTM trains in <20 min on CPU)
- Docker for submission packaging
