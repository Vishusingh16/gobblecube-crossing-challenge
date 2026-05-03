"""Enhanced feature engineering for intent classification.

Produces 40+ features from bbox history, ego motion, and scene metadata.
Used by both train_intent.py and predict.py - keep in sync.

Feature groups:
  1. Position (normalized)
  2. Bbox dimensions (normalized)
  3. Velocity (per-frame pixel displacement)
  4. Acceleration (velocity change)
  5. Jerk (acceleration change)
  6. Direction / angle features
  7. Velocity statistics (mean, std, min, max over windows)
  8. Aspect ratio evolution
  9. Ego interaction features
  10. Scene condition flags
"""

from __future__ import annotations

import numpy as np


def _as_2d(x) -> np.ndarray:
    return np.stack([np.asarray(r, dtype=np.float64) for r in x])


def extract_features(req: dict) -> np.ndarray:
    hist = _as_2d(req["bbox_history"])
    fw = float(req["frame_w"])
    fh = float(req["frame_h"])

    cx = (hist[:, 0] + hist[:, 2]) * 0.5
    cy = (hist[:, 1] + hist[:, 3]) * 0.5
    w = hist[:, 2] - hist[:, 0]
    h = hist[:, 3] - hist[:, 1]

    vx = np.diff(cx)
    vy = np.diff(cy)
    speed = np.sqrt(vx**2 + vy**2)

    ax = np.diff(vx) if len(vx) > 1 else np.array([0.0])
    ay = np.diff(vy) if len(vy) > 1 else np.array([0.0])
    accel = np.sqrt(ax**2 + ay**2) if len(ax) > 0 else np.array([0.0])

    jx = np.diff(ax) if len(ax) > 1 else np.array([0.0])
    jy = np.diff(ay) if len(ay) > 1 else np.array([0.0])

    ego_s = np.asarray(req["ego_speed_history"], dtype=np.float64)
    ego_y = np.asarray(req["ego_yaw_history"], dtype=np.float64)
    ego_avail = float(req["ego_available"])

    last_cx_n = cx[-1] / fw
    last_cy_n = cy[-1] / fh
    last_w_n = w[-1] / fw
    last_h_n = h[-1] / fh

    last4_vx = vx[-4:] if len(vx) >= 4 else vx
    last4_vy = vy[-4:] if len(vy) >= 4 else vy
    last8_vx = vx[-8:] if len(vx) >= 8 else vx
    last8_vy = vy[-8:] if len(vy) >= 8 else vy

    vx_mean_last4 = last4_vx.mean()
    vy_mean_last4 = last4_vy.mean()
    vx_mean_last8 = last8_vx.mean()
    vy_mean_last8 = last8_vy.mean()

    speed_mean = speed.mean()
    speed_std = speed.std()
    speed_max = speed.max()
    speed_min = speed.min()
    speed_last = speed[-1] if len(speed) > 0 else 0.0

    accel_mean = accel.mean() if len(accel) > 0 else 0.0
    accel_max = accel.max() if len(accel) > 0 else 0.0

    angle_last4 = np.arctan2(vy_mean_last4, vx_mean_last4 + 1e-8)
    angle_last8 = np.arctan2(vy_mean_last8, vx_mean_last8 + 1e-8)
    angle_diff = angle_last4 - angle_last8

    aspect_ratio = h[-1] / (w[-1] + 1e-6)
    aspect_mean = (h / (w + 1e-6)).mean()

    area = w * h
    area_change = (area[-1] - area[0]) / (area[0] + 1e-6)

    lateral_speed = np.abs(vy).mean()
    longitudinal_speed = np.abs(vx).mean()

    bbox_x_spread = cx.std() / fw
    bbox_y_spread = cy.std() / fh

    dist_to_center_x = (cx[-1] - fw * 0.5) / fw
    dist_to_center_y = (cy[-1] - fh * 0.5) / fh

    speed_var_short = vx[-4:].var() + vy[-4:].var() if len(vx) >= 4 else 0.0
    speed_var_long = vx.var() + vy.var()

    ego_ped_interaction = ego_s.mean() * speed_mean
    ego_yaw_x_ped_speed = np.abs(ego_y).mean() * speed_mean

    y_road_proximity = cy[-1] / fh
    decel = speed[-1] - speed[-4] if len(speed) >= 4 else 0.0
    is_decelerating = 1.0 if decel < -1.0 else 0.0
    is_accelerating = 1.0 if decel > 1.0 else 0.0
    bbox_area_last = w[-1] * h[-1]
    approaching_vehicle = 1.0 if ego_avail > 0.5 and ego_s[-1] > 2.0 and cy[-1] > fh * 0.3 else 0.0

    feats = [
        last_cx_n,
        last_cy_n,
        last_w_n,
        last_h_n,
        vx_mean_last4 / fw,
        vy_mean_last4 / fh,
        vx_mean_last8 / fw,
        vy_mean_last8 / fh,
        vx.std() / fw,
        vy.std() / fh,
        speed_mean / fw,
        speed_std / fw,
        speed_max / fw,
        speed_min / fw,
        speed_last / fw,
        accel_mean / fw,
        accel_max / fw,
        angle_last4,
        angle_last8,
        angle_diff,
        aspect_ratio,
        aspect_mean,
        area_change,
        lateral_speed / fh,
        longitudinal_speed / fw,
        bbox_x_spread,
        bbox_y_spread,
        dist_to_center_x,
        dist_to_center_y,
        speed_var_short / (fw**2),
        speed_var_long / (fw**2),
        ego_avail,
        ego_s.mean(),
        ego_s[-1],
        ego_s.max(),
        ego_s.std(),
        ego_y.mean(),
        ego_y[-1],
        np.abs(ego_y).max(),
        ego_ped_interaction,
        ego_yaw_x_ped_speed,
        1.0 if req.get("time_of_day") == "daytime" else 0.0,
        1.0 if req.get("time_of_day") == "nighttime" else 0.0,
        1.0 if req.get("weather") == "rain" else 0.0,
        1.0 if req.get("weather") == "snow" else 0.0,
        1.0 if req.get("location") == "street" else 0.0,
        1.0 if req.get("location") == "plaza" else 0.0,
        y_road_proximity,
        decel / fw,
        is_decelerating,
        is_accelerating,
        np.log1p(bbox_area_last) / np.log1p(fw * fh),
        approaching_vehicle,
    ]
    return np.asarray(feats, dtype=np.float32)


def featurize_df(df, request_fields: list[str]) -> np.ndarray:
    n = len(df)
    sample = extract_features({k: df.iloc[0][k] for k in request_fields})
    X = np.empty((n, len(sample)), dtype=np.float32)
    X[0] = sample
    for i in range(1, n):
        X[i] = extract_features({k: df.iloc[i][k] for k in request_fields})
    return X


REQUEST_FIELDS = [
    "ped_id", "frame_w", "frame_h",
    "time_of_day", "weather", "location", "ego_available",
    "bbox_history", "ego_speed_history", "ego_yaw_history",
    "requested_at_frame",
]
