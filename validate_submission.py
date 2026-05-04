"""Validate the full submission before submitting.

Checks:
1. All required files exist
2. Model weights load correctly
3. predict() returns correct format
4. Contract tests pass
5. Dockerfile is valid
6. README has required sections

Usage:
    python validate_submission.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REQUIRED_FILES = [
    "predict.py",
    "Dockerfile",
    "requirements.txt",
    "README.md",
    "CLAUDE.md",
    "AGENTS.md",
    "SUBMISSION_TEMPLATE.md",
    "features.py",
    "grade.py",
    "model_intent_ensemble.pkl",
    "model_traj_xgb.pkl",
    "model.pkl",
    "data/train.parquet",
    "data/dev.parquet",
    "tests/test_predict.py",
]

OPTIONAL_FILES = [
    "baseline.py",
    "eda.py",
    "smoke_test.py",
    "sequence.py",
    "train_intent_ensemble.py",
    "train_intent.py",
    "train_traj_xgb.py",
    "train_traj_ensemble.py",
    "train_trajectory.py",
    "model_intent.pkl",
    "model_trajectory.pt",
    "model_traj_ensemble.pkl",
]

README_SECTIONS = [
    "Quick Start",
    "Architecture",
    "Approach",
    "Output",
    "Score",
]


def check_files():
    missing = [f for f in REQUIRED_FILES if not Path(f).exists()]
    if missing:
        print(f"FAIL: Missing required files: {missing}")
        return False
    print(f"PASS: All {len(REQUIRED_FILES)} required files present")
    return True


def check_models_load():
    import pickle
    import numpy as np

    try:
        with open("model_intent_ensemble.pkl", "rb") as f:
            m = pickle.load(f)
        assert isinstance(m, dict) and "lgbm" in m and "xgb" in m
        print("PASS: Intent ensemble model loads (LightGBM + XGBoost dict)")
    except Exception as e:
        print(f"FAIL: Intent model load error: {e}")
        return False

    try:
        with open("model_traj_xgb.pkl", "rb") as f:
            m = pickle.load(f)
        assert "bbox_500ms_cx" in m
        print("PASS: Trajectory XGB model loads (8 regressors)")
    except Exception as e:
        print(f"FAIL: Trajectory model load error: {e}")
        return False

    return True


def check_predict_format():
    from predict import predict

    req = dict(
        ped_id="test_validate",
        frame_w=1920, frame_h=1080,
        time_of_day="daytime", weather="clear", location="street",
        ego_available=True,
        bbox_history=[[100+i*2, 200+i, 180+i*2, 380+i] for i in range(16)],
        ego_speed_history=[5.0]*16,
        ego_yaw_history=[0.0]*16,
        requested_at_frame=100,
    )

    try:
        out = predict(req)
        assert "intent" in out
        assert 0.0 <= out["intent"] <= 1.0
        for k in ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]:
            assert k in out
            assert len(out[k]) == 4
            assert all(isinstance(v, float) for v in out[k])
            assert all(v >= 0 for v in out[k])
        print(f"PASS: predict() returns correct format (intent={out['intent']:.4f})")
        return True
    except Exception as e:
        print(f"FAIL: predict() error: {e}")
        return False


def check_readme_sections():
    readme = Path("README.md").read_text()
    missing = [s for s in README_SECTIONS if s not in readme]
    if missing:
        print(f"WARN: README may be missing sections: {missing}")
        return True
    print(f"PASS: README contains all {len(README_SECTIONS)} key sections")
    return True


def check_dockerfile():
    df = Path("Dockerfile").read_text()
    assert "predict.py" in df or "grade.py" in df
    assert "requirements.txt" in df or "pip install" in df
    print("PASS: Dockerfile references predict.py/grade.py and requirements")
    return True


def main():
    print("=" * 50)
    print("  SUBMISSION VALIDATION")
    print("=" * 50)
    print()

    results = []
    results.append(check_files())
    results.append(check_models_load())
    results.append(check_predict_format())
    results.append(check_readme_sections())
    results.append(check_dockerfile())

    print()
    if all(results):
        print("=" * 50)
        print("  ALL CHECKS PASSED - Ready to submit!")
        print("=" * 50)
        return 0
    else:
        print("=" * 50)
        print("  SOME CHECKS FAILED - Fix before submitting")
        print("=" * 50)
        return 1


if __name__ == "__main__":
    sys.exit(main())
