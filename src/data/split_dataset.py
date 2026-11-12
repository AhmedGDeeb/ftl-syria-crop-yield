#!/usr/bin/env python
"""
Build train/val/test splits from the processed FAO + NASA datasets.

Joins yields (per crop, per year) with climate features (per governorate,
per year — averaged to national level since FAO is national). Applies
minimal preprocessing (dropna, dedup, drop partial years), then splits
chronologically (no shuffling — this is time-series data).

Inputs (from data/processed/):
    fao_with_anomaly_flags.csv   (falls back to fao_syria_yield.csv)
    nasa_with_anomaly_flags.csv  (falls back to nasa_syria_climate.csv)

Outputs (stable names, no timestamps):
    data/interim/train.csv
    data/interim/val.csv
    data/interim/test.csv
    data/interim/split_metadata.json

Usage:
    python -u src/data/split_dataset.py
    python -u src/data/split_dataset.py -f
    python -u src/data/split_dataset.py --crop Wheat
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR   = PROJECT_ROOT / "data" / "interim"

# Climate columns to bring in as model features
CLIMATE_FEATURES = [
    "t2m_mean", "t2m_max_extreme", "t2m_min_extreme",
    "precip_total", "rh2m_mean", "solar_mean", "ws2m_mean",
]

# Chronological split boundaries (years, inclusive on the right edge).
TRAIN_END = 2015   # ≤2015       → train
VAL_END   = 2019   # 2016–2019   → val
                   # 2020+       → test

# Partial / edge years to drop before splitting
DROP_YEARS_FROM = 2026


def _load_and_aggregate_nasa(nasa_path: Path) -> pd.DataFrame:
    """
    Aggregate daily NASA data to yearly, then average across governorates
    so it can be joined to national FAO yields.
    """
    df = pd.read_csv(nasa_path, parse_dates=["date"])
    df["year"] = df["date"].dt.year

    yearly = (
        df.groupby(["governorate", "year"])
          .agg(
              t2m_mean        = ("t2m",          "mean"),
              t2m_max_extreme = ("t2m_max",      "max"),
              t2m_min_extreme = ("t2m_min",      "min"),
              precip_total    = ("prectotcorr",  "sum"),
              rh2m_mean       = ("rh2m",         "mean"),
              solar_mean      = ("solar_kwh_m2", "mean"),
              ws2m_mean       = ("ws2m",         "mean"),
              n_days          = ("date",         "count"),
          )
          .reset_index()
    )

    # Keep only years with full-ish coverage
    yearly = yearly[yearly["n_days"] >= 300]

    # National average across governorates (FAO yields are national)
    national = (
        yearly.groupby("year")[CLIMATE_FEATURES]
              .mean()
              .reset_index()
    )
    return national


def _resolve_input_path(preferred_name: str, fallback_name: str) -> Path:
    """Use the anomaly-flagged file if it exists, else the plain processed one."""
    preferred = PROCESSED_DIR / preferred_name
    fallback  = PROCESSED_DIR / fallback_name
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise SystemExit(
        f"Missing both {preferred.name} and {fallback.name} in {PROCESSED_DIR}.\n"
        f"Run build_dataset.py first."
    )


def build_splits(force: bool = False, crop: str | None = None) -> None:
    train_path = INTERIM_DIR / "train.csv"

    if train_path.exists() and not force:
        print(f"[SKIP] {train_path.name} exists. Use -f to rebuild.")
        return

    fao_path  = _resolve_input_path("fao_with_anomaly_flags.csv",  "fao_syria_yield.csv")
    nasa_path = _resolve_input_path("nasa_with_anomaly_flags.csv", "nasa_syria_climate.csv")

    print(f"[INFO] FAO  : {fao_path.name}")
    print(f"[INFO] NASA : {nasa_path.name}")

    # ── 1. Load both ─────────────────────────────────────────────
    fao  = pd.read_csv(fao_path)
    nasa = _load_and_aggregate_nasa(nasa_path)

    # ── 2. Optional single-crop filter ───────────────────────────
    if crop:
        if crop not in fao["crop"].unique():
            raise SystemExit(
                f"Crop '{crop}' not in dataset. "
                f"Available: {sorted(fao['crop'].unique())}"
            )
        fao = fao[fao["crop"] == crop].copy()
        print(f"[INFO] Filtered to crop = {crop}")

    # ── 3. Drop partial / edge years ─────────────────────────────
    fao  = fao[fao["year"] < DROP_YEARS_FROM].copy()
    nasa = nasa[nasa["year"] < DROP_YEARS_FROM].copy()

    # ── 4. Merge on year ─────────────────────────────────────────
    df = fao.merge(nasa, on="year", how="inner")
    print(f"[INFO] Merged rows: {len(df):,}")

    # ── 5. Simple preprocessing ──────────────────────────────────
    keep = (
        ["crop", "year", "yield_hg_ha", "flag"] +
        [c for c in ["yield_anomaly", "drought_year", "conflict_year"]
         if c in df.columns] +
        CLIMATE_FEATURES +
        [c for c in ["climate_anomaly"] if c in df.columns]
    )
    df = df[[c for c in keep if c in df.columns]]

    # 5a. Drop rows with any NaN in features or target
    before = len(df)
    df = df.dropna(subset=["yield_hg_ha"] + CLIMATE_FEATURES).reset_index(drop=True)
    print(f"[INFO] Dropped {before - len(df)} rows with NaN")

    # 5b. Deduplicate (crop, year)
    dup_before = len(df)
    df = df.drop_duplicates(subset=["crop", "year"], keep="last").reset_index(drop=True)
    if len(df) < dup_before:
        print(f"[INFO] Dropped {dup_before - len(df)} duplicate (crop, year) rows")

    # 5c. Drop rows with inf values
    inf_mask = np.isinf(df[CLIMATE_FEATURES].values.astype(float)).any(axis=1)
    if inf_mask.any():
        print(f"[WARN] Dropping {int(inf_mask.sum())} rows with inf values")
        df = df[~inf_mask].reset_index(drop=True)

    if df.empty:
        raise SystemExit("Empty dataset after preprocessing — check upstream.")

    # ── 6. Chronological split (no shuffle — time-series) ───────
    train = df[df["year"] <= TRAIN_END].copy()
    val   = df[(df["year"] > TRAIN_END) & (df["year"] <= VAL_END)].copy()
    test  = df[df["year"] > VAL_END].copy()

    print("\n[INFO] Split sizes:")
    if len(train):
        print(f"  train: {len(train):>6} rows  years {train['year'].min()}–{train['year'].max()}")
    else:
        print("  train:      0 rows  <-- EMPTY")
    if len(val):
        print(f"  val  : {len(val):>6} rows  years {val['year'].min()}–{val['year'].max()}")
    else:
        print("  val  :      0 rows  <-- EMPTY")
    if len(test):
        print(f"  test : {len(test):>6} rows  years {test['year'].min()}–{test['year'].max()}")
    else:
        print("  test :      0 rows  <-- EMPTY")

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise SystemExit("One of train/val/test is empty — adjust TRAIN_END / VAL_END.")

    # ── 7. Save ─────────────────────────────────────────────────
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    train.to_csv(INTERIM_DIR / "train.csv", index=False)
    val.to_csv(  INTERIM_DIR / "val.csv",   index=False)
    test.to_csv( INTERIM_DIR / "test.csv",  index=False)

    feature_columns = (
        CLIMATE_FEATURES +
        [c for c in ["yield_anomaly", "drought_year",
                     "conflict_year", "climate_anomaly"]
         if c in df.columns]
    )

    metadata = {
        "fao_source"     : fao_path.name,
        "nasa_source"    : nasa_path.name,
        "crop_filter"    : crop,
        "train_years"    : [int(train["year"].min()), int(train["year"].max())],
        "val_years"      : [int(val["year"].min()),   int(val["year"].max())],
        "test_years"     : [int(test["year"].min()),  int(test["year"].max())],
        "row_counts"     : {"train": len(train), "val": len(val), "test": len(test)},
        "target"         : "yield_hg_ha",
        "feature_columns": feature_columns,
        "id_columns"     : ["crop", "year"],
        "flag_column"    : "flag",
    }
    with open(INTERIM_DIR / "split_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[SAVED] {INTERIM_DIR / 'train.csv'}")
    print(f"[SAVED] {INTERIM_DIR / 'val.csv'}")
    print(f"[SAVED] {INTERIM_DIR / 'test.csv'}")
    print(f"[SAVED] {INTERIM_DIR / 'split_metadata.json'}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Build chronological train/val/test splits from processed FAO + NASA data."
    )
    parser.add_argument("-f", "--force", action="store_true",
                        help="Rebuild even if splits exist.")
    parser.add_argument("--crop", type=str, default=None,
                        help="Optional: filter to a single crop before splitting.")
    args = parser.parse_args(argv)

    print("============================================================")
    print(f" FTL Syria — Split Dataset  (force={args.force}, crop={args.crop})")
    print(f" Processed : {PROCESSED_DIR}")
    print(f" Interim   : {INTERIM_DIR}")
    print("============================================================")

    build_splits(force=args.force, crop=args.crop)

    print("\n============================================================")
    print(" [DONE]")
    print("============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())