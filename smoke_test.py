"""Smoke test - verify full inference pipeline end-to-end.

Usage:
    python smoke_test.py
"""

from __future__ import annotations

import time
import numpy as np
from predict import predict


def make_request(**overrides):
    req = dict(
        ped_id="test00000001",
        frame_w=1920,
        frame_h=1080,
        time_of_day="daytime",
        weather="clear",
        location="street",
        ego_available=True,
        bbox_history=[[100.0 + i * 2, 200.0 + i, 180.0 + i * 2, 380.0 + i] for i in range(16)],
        ego_speed_history=[5.0] * 16,
        ego_yaw_history=[0.0] * 16,
        requested_at_frame=100,
    )
    req.update(overrides)
    return req


def test_basic():
    out = predict(make_request())
    assert 0.0 <= out["intent"] <= 1.0, f"intent out of range: {out['intent']}"
    for key in ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]:
        assert len(out[key]) == 4, f"{key} wrong length"
        for v in out[key]:
            assert np.isfinite(v), f"{key} has non-finite value"
    print("PASS: basic prediction")


def test_latency():
    req = make_request()
    times = []
    for _ in range(100):
        t0 = time.time()
        predict(req)
        times.append(time.time() - t0)
    mean_ms = np.mean(times) * 1000
    p99_ms = np.percentile(times, 99) * 1000
    print(f"Latency: mean={mean_ms:.1f}ms, p99={p99_ms:.1f}ms")
    assert mean_ms < 200, f"Too slow: {mean_ms:.1f}ms > 200ms"


def test_no_ego():
    out = predict(make_request(ego_available=False,
                               ego_speed_history=[0.0] * 16,
                               ego_yaw_history=[0.0] * 16))
    assert 0.0 <= out["intent"] <= 1.0
    print("PASS: no ego data")


def test_stationary():
    out = predict(make_request(bbox_history=[[100.0, 200.0, 180.0, 380.0]] * 16))
    assert np.isfinite(out["intent"])
    for key in ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]:
        for v in out[key]:
            assert np.isfinite(v)
    print("PASS: stationary pedestrian")


if __name__ == "__main__":
    print("Running smoke tests...")
    test_basic()
    test_no_ego()
    test_stationary()
    test_latency()
    print("\nAll smoke tests passed!")
