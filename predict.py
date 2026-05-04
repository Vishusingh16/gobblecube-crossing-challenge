"""Submission entry point - LightGBM intent + XGBoost trajectory.

Contract (do NOT change the signature):

    predict(request: dict) -> dict
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from features import extract_features

MODEL_DIR = Path(__file__).parent
INTENT_MODEL_PATH = MODEL_DIR / "model_intent_ensemble.pkl"
INTENT_SINGLE_PATH = MODEL_DIR / "model_intent.pkl"
TRAJ_XGB_PATH = MODEL_DIR / "model_traj_xgb.pkl"
BASELINE_MODEL_PATH = MODEL_DIR / "model.pkl"

HORIZONS_FRAMES = [8, 15, 23, 30]
HORIZON_KEYS = ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]

_cached_intent = None
_cached_traj_xgb = None
_cached_traj_feats_func = None


def _as_2d(x) -> np.ndarray:
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])


def _load_intent_model():
    global _cached_intent
    if _cached_intent is None:
        if INTENT_MODEL_PATH.exists():
            with open(INTENT_MODEL_PATH, "rb") as f:
                _cached_intent = pickle.load(f)
        elif INTENT_SINGLE_PATH.exists():
            with open(INTENT_SINGLE_PATH, "rb") as f:
                _cached_intent = pickle.load(f)
        elif BASELINE_MODEL_PATH.exists():
            with open(BASELINE_MODEL_PATH, "rb") as f:
                _cached_intent = pickle.load(f)["intent"]
    return _cached_intent


def _load_traj_xgb():
    global _cached_traj_xgb
    if _cached_traj_xgb is None and TRAJ_XGB_PATH.exists():
        with open(TRAJ_XGB_PATH, "rb") as f:
            _cached_traj_xgb = pickle.load(f)
    return _cached_traj_xgb


def _cv_baseline(req: dict) -> list[dict]:
    hist = _as_2d(req["bbox_history"])
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


def _build_trajectory_features(req: dict) -> np.ndarray:
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


def predict(request: dict) -> dict:
    intent_model = _load_intent_model()
    feats = extract_features(request).reshape(1, -1)
    if not np.isfinite(feats).all():
        feats = np.nan_to_num(feats, nan=0.0, posinf=1.0, neginf=-1.0)

    if isinstance(intent_model, dict) and "lgbm" in intent_model:
        p1 = float(intent_model["lgbm"].predict_proba(feats)[0, 1])
        p2 = float(intent_model["xgb"].predict_proba(feats)[0, 1])
        intent_prob = (p1 + p2) / 2.0
    else:
        intent_prob = float(intent_model.predict_proba(feats)[0, 1])
    if not np.isfinite(intent_prob):
        intent_prob = 0.5

    cv_bboxes = _cv_baseline(request)
    traj_models = _load_traj_xgb()

    out: dict = {}
    if traj_models is not None:
        traj_feats = _build_trajectory_features(request).reshape(1, -1)
        if not np.isfinite(traj_feats).all():
            traj_feats = np.nan_to_num(traj_feats, nan=0.0, posinf=1.0, neginf=-1.0)

        for j, key in enumerate(HORIZON_KEYS):
            cv = cv_bboxes[j]
            dcx = float(traj_models[f"{key}_cx"].predict(traj_feats)[0])
            dcy = float(traj_models[f"{key}_cy"].predict(traj_feats)[0])
            if not (np.isfinite(dcx) and np.isfinite(dcy)):
                dcx, dcy = 0.0, 0.0
            cx_pred = cv["cx"] + dcx
            cy_pred = cv["cy"] + dcy
            out[key] = [
                float(cx_pred - cv["w"] / 2),
                float(cy_pred - cv["h"] / 2),
                float(cx_pred + cv["w"] / 2),
                float(cy_pred + cv["h"] / 2),
            ]
    else:
        for j, key in enumerate(HORIZON_KEYS):
            cv = cv_bboxes[j]
            out[key] = [float(cv["x1"]), float(cv["y1"]),
                        float(cv["x2"]), float(cv["y2"])]

    for k in HORIZON_KEYS:
        out[k] = [max(0.0, float(v)) if np.isfinite(v) else 0.0 for v in out[k]]

    out["intent"] = max(0.0, min(1.0, intent_prob))
    return out
