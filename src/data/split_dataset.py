#!/usr/bin/env python
"""
Build train/val/test splits from the processed FAO + NASA datasets.

Joins yields (per crop, per year) with climate features (per governorate,
per year — averaged to national level since FAO is national). Applies
minimal preprocessing (dropna, dedup, drop partial years), then splits
chronologically (no shuffling — this is time-series data).

All configurable values (feature columns, split boundaries, drop-year
guard) come from config.yaml under the `training:` key, with in-code
fallbacks if the file is missing.

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
import yaml

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE   = PROJECT_ROOT / "config.yaml"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
INTERIM_DIR   = PROJECT_ROOT / "data" / "interim"


# ══════════════════════════════════════════════════════════════════
# Config loading (same pattern as ingest.py / build_dataset.py)
# ══════════════════════════════════════════════════════════════════
def _load_config() -> dict:
    if not CONFIG_FILE.is_file():
        print(f"[config] {CONFIG_FILE} not found — using in-code defaults.",
              flush=True)
        return {}
    try:
        with CONFIG_FILE.open(encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[config] failed to parse {CONFIG_FILE}: {exc} — using in-code defaults.",
              flush=True)
        return {}


CFG = _load_config()


def _cfg(*keys, default):
    node = CFG
    for k in keys:
        if not isinstance(node, dict) or k not in node:
            return default
        node = node[k]
    return node if node is not None else default


# ══════════════════════════════════════════════════════════════════
# Configuration (from config.yaml `training:` section)
# ══════════════════════════════════════════════════════════════════
CLIMATE_FEATURES = _cfg("training", "feature_cols", default=[
    "t2m_mean", "t2m_max_extreme", "t2m_min_extreme",
    "precip_total", "rh2m_mean", "solar_mean", "ws2m_mean",
])

ANOMALY_FEATURES = _cfg("training", "anomaly_features", default=[
    "yield_anomaly", "drought_year", "conflict_year", "climate_anomaly",
])

_SPLIT          = _cfg("training", "split", default={})
TRAIN_END       = int(_SPLIT.get("train_end", 2015))
VAL_END         = int(_SPLIT.get("val_end",   2019))
DROP_YEARS_FROM = int(_SPLIT.get("drop_years_from", 2026))


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════
def _load_and_aggregate_nasa(nasa_path: Path) -> pd.DataFrame:
    """Aggregate daily NASA data to yearly, then average across governorates
    so it can be joined to national FAO yields."""
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

    yearly = yearly[yearly["n_days"] >= 300]

    national = (
        yearly.groupby("year")[CLIMATE_FEATURES]
              .mean()
              .reset_index()
    )
    return national


def _resolve_input_path(preferred_name: str, fallback_name: str) -> Path:
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


# ══════════════════════════════════════════════════════════════════
# Main routine
# ══════════════════════════════════════════════════════════════════
def build_splits(force: bool = False, crop: str | None = None) -> None:
    train_path = INTERIM_DIR / "train.csv"

    if train_path.exists() and not force:
        print(f"[SKIP] {train_path.name} exists. Use -f to rebuild.")
        return

    fao_path  = _resolve_input_path("fao_with_anomaly_flags.csv",  "fao_syria_yield.csv")
    nasa_path = _resolve_input_path("nasa_with_anomaly_flags.csv", "nasa_syria_climate.csv")

    print(f"[INFO] FAO  : {fao_path.name}")
    print(f"[INFO] NASA : {nasa_path.name}")
    print(f"[INFO] Split: train_end={TRAIN_END}, val_end={VAL_END}, "
          f"drop_years_from={DROP_YEARS_FROM}")
    print(f"[INFO] Features: {CLIMATE_FEATURES}")

    # 1. Load both
    fao  = pd.read_csv(fao_path)
    nasa = _load_and_aggregate_nasa(nasa_path)

    # 2. Optional single-crop filter
    if crop:
        if crop not in fao["crop"].unique():
            raise SystemExit(
                f"Crop '{crop}' not in dataset. "
                f"Available: {sorted(fao['crop'].unique())}"
            )
        fao = fao[fao["crop"] == crop].copy()
        print(f"[INFO] Filtered to crop = {crop}")

    # 3. Drop partial / edge years
    fao  = fao[fao["year"] < DROP_YEARS_FROM].copy()
    nasa = nasa[nasa["year"] < DROP_YEARS_FROM].copy()

    # 4. Merge on year
    df = fao.merge(nasa, on="year", how="inner")
    print(f"[INFO] Merged rows: {len(df):,}")

    # 5. Simple preprocessing
    keep = (
        ["crop", "year", "yield_hg_ha", "flag"]
        + [c for c in ANOMALY_FEATURES if c in df.columns and c != "climate_anomaly"]
        + CLIMATE_FEATURES
        + ["climate_anomaly"] if "climate_anomaly" in df.columns else []
    )
    # Flatten the conditional properly
    keep = (
        ["crop", "year", "yield_hg_ha", "flag"]
        + [c for c in ANOMALY_FEATURES if c in df.columns]
        + CLIMATE_FEATURES
    )
    # De-duplicate while preserving order (flag stays; anomaly cols unique)
    seen = set()
    keep = [c for c in keep if not (c in seen or seen.add(c))]
    df = df[[c for c in keep if c in df.columns]]

    # 5a. Drop NaN
    before = len(df)
    df = df.dropna(subset=["yield_hg_ha"] + CLIMATE_FEATURES).reset_index(drop=True)
    print(f"[INFO] Dropped {before - len(df)} rows with NaN")

    # 5b. Dedup (crop, year)
    dup_before = len(df)
    df = df.drop_duplicates(subset=["crop", "year"], keep="last").reset_index(drop=True)
    if len(df) < dup_before:
        print(f"[INFO] Dropped {dup_before - len(df)} duplicate (crop, year) rows")

    # 5c. Drop inf
    inf_mask = np.isinf(df[CLIMATE_FEATURES].values.astype(float)).any(axis=1)
    if inf_mask.any():
        print(f"[WARN] Dropping {int(inf_mask.sum())} rows with inf values")
        df = df[~inf_mask].reset_index(drop=True)

    if df.empty:
        raise SystemExit("Empty dataset after preprocessing — check upstream.")

    # 6. Chronological split
    train = df[df["year"] <= TRAIN_END].copy()
    val   = df[(df["year"] > TRAIN_END) & (df["year"] <= VAL_END)].copy()
    test  = df[df["year"] > VAL_END].copy()

    print("\n[INFO] Split sizes:")
    for name, part in [("train", train), ("val", val), ("test", test)]:
        if len(part):
            print(f"  {name:<5}: {len(part):>6} rows  "
                  f"years {part['year'].min()}–{part['year'].max()}")
        else:
            print(f"  {name:<5}:      0 rows  <-- EMPTY")

    if len(train) == 0 or len(val) == 0 or len(test) == 0:
        raise SystemExit("One of train/val/test is empty — adjust split boundaries in config.yaml.")

    # 7. Save
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    train.to_csv(INTERIM_DIR / "train.csv", index=False)
    val.to_csv(  INTERIM_DIR / "val.csv",   index=False)
    test.to_csv( INTERIM_DIR / "test.csv",  index=False)

    feature_columns = (
        CLIMATE_FEATURES
        + [c for c in ANOMALY_FEATURES if c in df.columns]
    )

    metadata = {
        "fao_source"      : fao_path.name,
        "nasa_source"     : nasa_path.name,
        "crop_filter"     : crop,
        "config_split"    : {
            "train_end":       TRAIN_END,
            "val_end":         VAL_END,
            "drop_years_from": DROP_YEARS_FROM,
        },
        "train_years"     : [int(train["year"].min()), int(train["year"].max())],
        "val_years"       : [int(val["year"].min()),   int(val["year"].max())],
        "test_years"      : [int(test["year"].min()),  int(test["year"].max())],
        "row_counts"      : {"train": len(train), "val": len(val), "test": len(test)},
        "target"          : "yield_hg_ha",
        "feature_columns" : feature_columns,
        "id_columns"      : ["crop", "year"],
        "flag_column"     : "flag",
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
    print(f" Config    : {CONFIG_FILE}")
    print("============================================================")

    build_splits(force=args.force, crop=args.crop)

    print("\n============================================================")
    print(" [DONE]")
    print("============================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())