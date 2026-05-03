"""Exploratory Data Analysis for the Crossing Challenge.

Prints summary statistics and distributions to help understand the data
before building models.

Usage:
    python eda.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).parent / "data"


def _bbox_to_center(bbox: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cx = (bbox[:, 0] + bbox[:, 2]) * 0.5
    cy = (bbox[:, 1] + bbox[:, 3]) * 0.5
    return cx, cy


def analyze_basics(df: pd.DataFrame, name: str) -> None:
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Positive rate (will_cross_2s): {df['will_cross_2s'].mean():.4f}")
    print(f"  Ego available: {df['ego_available'].mean():.4f}")


def analyze_scene_metadata(df: pd.DataFrame) -> None:
    print("\n--- Scene Metadata ---")
    for col in ["time_of_day", "weather", "location"]:
        vc = df[col].value_counts()
        print(f"  {col}:")
        for val, cnt in vc.items():
            print(f"    {repr(val):20s} {cnt:6d} ({cnt/len(df):.1%})")


def analyze_bbox_history(df: pd.DataFrame) -> None:
    print("\n--- Bbox History Analysis ---")
    sample = df.head(2000)
    last_bboxes = np.stack([np.stack([np.asarray(r, dtype=np.float64) for r in row])[-1] for row in sample["bbox_history"]])
    cx = (last_bboxes[:, 0] + last_bboxes[:, 2]) * 0.5
    cy = (last_bboxes[:, 1] + last_bboxes[:, 3]) * 0.5
    w = last_bboxes[:, 2] - last_bboxes[:, 0]
    h = last_bboxes[:, 3] - last_bboxes[:, 1]
    print(f"  Last-frame center x: mean={cx.mean():.1f}, std={cx.std():.1f}")
    print(f"  Last-frame center y: mean={cy.mean():.1f}, std={cy.std():.1f}")
    print(f"  Last-frame width:    mean={w.mean():.1f}, std={w.std():.1f}")
    print(f"  Last-frame height:   mean={h.mean():.1f}, std={h.std():.1f}")
    print(f"  Aspect ratio (h/w):  mean={(h/(w+1e-6)).mean():.2f}")


def analyze_velocities(df: pd.DataFrame) -> None:
    print("\n--- Velocity Analysis ---")
    sample = df.head(2000)
    speeds = []
    for _, row in sample.iterrows():
        hist = np.stack([np.asarray(r, dtype=np.float64) for r in row["bbox_history"]])
        cx = (hist[:, 0] + hist[:, 2]) * 0.5
        cy = (hist[:, 1] + hist[:, 3]) * 0.5
        vx = np.diff(cx)
        vy = np.diff(cy)
        speed = np.sqrt(vx**2 + vy**2)
        speeds.append(speed)
    speeds = np.array(speeds)
    mean_speed = speeds.mean(axis=1)
    print(f"  Mean per-frame speed: {mean_speed.mean():.2f} px")
    print(f"  Speed percentiles: 25%={np.percentile(mean_speed,25):.1f}, "
          f"50%={np.percentile(mean_speed,50):.1f}, "
          f"75%={np.percentile(mean_speed,75):.1f}, "
          f"95%={np.percentile(mean_speed,95):.1f}")


def analyze_future_targets(df: pd.DataFrame) -> None:
    print("\n--- Future Target Analysis ---")
    sample = df.head(2000)
    for horizon in ["bbox_500ms", "bbox_1000ms", "bbox_1500ms", "bbox_2000ms"]:
        cur = np.stack([np.stack([np.asarray(r, dtype=np.float64) for r in row])[-1] for row in sample["bbox_history"]])
        fut = np.stack([np.asarray(row[horizon], dtype=np.float64) for _, row in sample.iterrows()])
        cur_cx = (cur[:, 0] + cur[:, 2]) * 0.5
        cur_cy = (cur[:, 1] + cur[:, 3]) * 0.5
        fut_cx = (fut[:, 0] + fut[:, 2]) * 0.5
        fut_cy = (fut[:, 1] + fut[:, 3]) * 0.5
        ade = np.sqrt((cur_cx - fut_cx)**2 + (cur_cy - fut_cy)**2).mean()
        print(f"  Zero-vel ADE at {horizon}: {ade:.1f} px")


def analyze_ego_motion(df: pd.DataFrame) -> None:
    print("\n--- Ego Motion Analysis ---")
    ego_mask = df["ego_available"] == True  # noqa: E712
    ego_df = df[ego_mask]
    no_ego_df = df[~ego_mask]
    print(f"  With ego: {len(ego_df)} ({ego_mask.mean():.1%})")
    print(f"  Without ego: {len(no_ego_df)} ({(~ego_mask).mean():.1%})")
    if len(ego_df) > 0:
        ego_speeds = np.stack(ego_df["ego_speed_history"].values)
        ego_yaws = np.stack(ego_df["ego_yaw_history"].values)
        print(f"  Ego speed: mean={ego_speeds.mean():.2f} m/s, std={ego_speeds.std():.2f}")
        print(f"  Ego yaw:   mean={ego_yaws.mean():.4f} rad/s, std={ego_yaws.std():.4f}")
        print(f"  Positive rate (ego=yes):   {ego_df['will_cross_2s'].mean():.4f}")
        print(f"  Positive rate (ego=no):    {no_ego_df['will_cross_2s'].mean():.4f}")


def main() -> None:
    print("Loading data...")
    train = pd.read_parquet(DATA / "train.parquet")
    dev = pd.read_parquet(DATA / "dev.parquet")

    analyze_basics(train, "TRAIN")
    analyze_basics(dev, "DEV")

    analyze_scene_metadata(train)
    analyze_bbox_history(train)
    analyze_velocities(train)
    analyze_future_targets(train)
    analyze_ego_motion(train)

    print("\n--- Cross-tab: time_of_day x will_cross ---")
    ct = pd.crosstab(train["time_of_day"], train["will_cross_2s"], normalize="index")
    print(ct.to_string())

    print("\nDone.")


if __name__ == "__main__":
    main()
