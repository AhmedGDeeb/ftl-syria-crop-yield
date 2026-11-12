#!/usr/bin/env python
"""
Merge raw FAO and NASA CSVs into single tidy datasets, keeping only
project-relevant columns.

Run AFTER src/data/ingest.py has populated data/raw/.

Outputs (names include data year range + creation timestamp):
    data/processed/fao_syria_yield_<start>_<end>_<YYYYMMDD-HHMMSS>.csv
    data/processed/nasa_syria_climate_<start>_<end>_<YYYYMMDD-HHMMSS>.csv

Usage:
    python -u src/data/build_dataset.py                    # both
    python -u src/data/build_dataset.py --source fao
    python -u src/data/build_dataset.py --source nasa -f
"""
import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

# ------------------------------------------------------------------
# FAO — columns to keep
# ------------------------------------------------------------------
FAO_KEEP_COLUMNS = [
    "Area", "Item", "Item Code (CPC)",
    "Element", "Year", "Unit", "Value", "Flag",
]

YIELD_ELEMENT_CODE = 5412
PRIMARY_CROP_ITEMS = [
    "Wheat", "Barley", "Maize", "Olives", "Grapes", "Apples",
    "Potatoes", "Tomatoes", "Rice", "Oats", "Rye", "Sorghum",
    "Chick peas, dry", "Lentils, dry", "Broad beans, dry",
    "Sugar beet", "Cotton seed", "Sunflower seed", "Sesame seed",
]


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _year_range(df: pd.DataFrame, year_col: str) -> tuple[int, int]:
    years = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)
    if years.empty:
        raise SystemExit(f"Cannot determine year range: '{year_col}' is empty.")
    return int(years.min()), int(years.max())


def _read_fao_file(path: Path) -> pd.DataFrame:
    """Read a single per-year FAO CSV; handle latin-1 and mixed types."""
    df = pd.read_csv(path, encoding="latin-1", low_memory=False)

    if "Item Code (CPC)" in df.columns:
        # Two-step cast: float -> keep only whole numbers -> nullable Int64.
        # Direct .astype("Int64") raises on non-integer floats (e.g. 1371.5)
        # and mixed NaN, which is what FAOSTAT's CPC column contains.
        s = pd.to_numeric(df["Item Code (CPC)"], errors="coerce")
        is_whole = s.notna() & (s % 1 == 0)
        df["Item Code (CPC)"] = s.where(is_whole).astype("Int64")

    return df


def _find_existing(out_dir: Path, prefix: str) -> Path | None:
    if not out_dir.exists():
        return None
    matches = sorted(out_dir.glob(f"{prefix}_*.csv"))
    return matches[-1] if matches else None


def build_fao_dataset(force: bool = False) -> Path | None:
    prefix = "fao_syria_yield"
    existing = _find_existing(PROCESSED_DIR, prefix)
    if existing and not force:
        print(f"[SKIP] {existing.name} exists. Use -f to rebuild.")
        return existing

    files = sorted(RAW_DIR.glob("QCL_Syria_AllCrops_*.csv"))
    if not files:
        raise SystemExit(
            f"No FAO files found in {RAW_DIR}. Run ingest.py --source fao first."
        )

    print(f"[INFO] Merging {len(files)} FAO files ...", flush=True)
    frames = []
    for p in files:
        try:
            frames.append(_read_fao_file(p))
        except Exception as e:
            print(f"  [WARN] {p.name}: {e}", flush=True)

    if not frames:
        raise SystemExit("No FAO file could be read.")

    df = pd.concat(frames, ignore_index=True)

    missing = [c for c in FAO_KEEP_COLUMNS if c not in df.columns]
    if missing:
        print(f"[WARN] FAO file is missing columns: {missing}", flush=True)
    df = df[[c for c in FAO_KEEP_COLUMNS if c in df.columns]].copy()

    # Keep only Yield rows
    if "Element" in df.columns:
        df = df[df["Element"].str.strip().str.lower() == "yield"].copy()

    # Keep only target crops
    if "Item" in df.columns:
        crop_pattern = "|".join(re.escape(c) for c in PRIMARY_CROP_ITEMS)
        df = df[df["Item"].str.contains(crop_pattern, case=False, na=False)].copy()

    # Drop rows with no target value
    df = df.dropna(subset=["Value"])

    # Normalize column names
    df = df.rename(columns={
        "Area": "country",
        "Item": "crop",
        "Item Code (CPC)": "crop_code",
        "Element": "element",
        "Year": "year",
        "Unit": "unit",
        "Value": "yield_hg_ha",
        "Flag": "flag",
    })

    if df.empty:
        print("[WARN] FAO merge produced 0 rows. Check Element/Item filters "
              "against the raw file contents.", flush=True)
        return None

    df["year"] = df["year"].astype(int)
    df = df.sort_values(["crop", "year"]).reset_index(drop=True)

    start_year, end_year = _year_range(df, "year")
    ts = _timestamp()
    out_path = PROCESSED_DIR / f"{prefix}_{start_year}_{end_year}_{ts}.csv"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path.name}  rows={len(df)}  "
          f"years={start_year}-{end_year}", flush=True)
    return out_path


# ------------------------------------------------------------------
# NASA
# ------------------------------------------------------------------
NASA_KEEP_COLUMNS = [
    "governorate", "latitude", "longitude", "date",
    "t2m", "t2m_max", "t2m_min", "prectotcorr",
    "rh2m", "solar_kwh_m2", "ws2m",
]

# Canonical governorate metadata (from cities.csv).
# Add to build_dataset.py near the top, next to NASA_KEEP_COLUMNS.
GOVERNORATES = {
    "aleppo":       {"name": "Aleppo",        "name_ar": "حلب",       "lat": 36.17, "lon": 37.61},
    "al_hasakah":   {"name": "Al-Hasakah",    "name_ar": "الحسكة",    "lat": 36.42, "lon": 40.91},
    "ar_raqqah":    {"name": "Ar-Raqqah",     "name_ar": "الرقة",     "lat": 36.01, "lon": 38.91},
    "as_suwayda":   {"name": "As-Suwayda",    "name_ar": "السويداء",  "lat": 32.77, "lon": 36.92},
    "daraa":        {"name": "Daraa",         "name_ar": "درعا",      "lat": 32.83, "lon": 36.17},
    "deir_ez_zor":  {"name": "Deir ez-Zor",   "name_ar": "دير الزور", "lat": 35.25, "lon": 40.28},
    "hama":         {"name": "Hama",          "name_ar": "حماة",      "lat": 35.30, "lon": 37.25},
    "homs":         {"name": "Homs",          "name_ar": "حمص",       "lat": 34.26, "lon": 38.16},
    "idlib":        {"name": "Idlib",         "name_ar": "إدلب",      "lat": 35.83, "lon": 36.68},
    "latakia":      {"name": "Latakia",       "name_ar": "اللاذقية",  "lat": 35.57, "lon": 36.01},
    "quneitra":     {"name": "Quneitra",      "name_ar": "القنيطرة",  "lat": 33.02, "lon": 35.79},
    "tartus":       {"name": "Tartus",        "name_ar": "طرطوس",     "lat": 34.95, "lon": 36.09},
    "damascus":     {"name": "Damascus",      "name_ar": "دمشق",      "lat": 33.46, "lon": 37.05},
}


def _gov_key_from_filename(path: Path) -> str | None:
    """
    Extract canonical governorate key from a filename of the form
        nasa_daily_<gov>_<year>_<month>.csv
    Returns lowercase with '-' and ' ' replaced by '_', e.g. 'deir_ez_zor'.
    Returns None if the filename doesn't match the expected pattern.
    """
    stem = path.stem                       # e.g. "nasa_daily_tartus_2023_10"
    m = re.match(r"nasa_daily_(.+)_(\d{4})_(\d{2})$", stem)
    if not m:
        return None
    raw = m.group(1).lower().replace("-", "_").replace(" ", "_")
    return raw


def build_nasa_dataset(force: bool = False) -> Path | None:
    prefix = "nasa_syria_climate"
    existing = _find_existing(PROCESSED_DIR, prefix)
    if existing and not force:
        print(f"[SKIP] {existing.name} exists. Use -f to rebuild.")
        return existing

    files = sorted(RAW_DIR.glob("nasa_daily_*.csv"))
    if not files:
        raise SystemExit(
            f"No NASA files found in {RAW_DIR}. Run ingest.py --source nasa first."
        )

    print(f"[INFO] Merging {len(files)} NASA files ...", flush=True)

    frames = []
    unknown_keys = set()
    per_gov_counts: dict[str, int] = {}

    for p in files:
        gov_key = _gov_key_from_filename(p)
        if gov_key is None:
            print(f"  [WARN] Cannot parse governorate from filename: {p.name}",
                  flush=True)
            continue
        if gov_key not in GOVERNORATES:
            unknown_keys.add(gov_key)
            # Still load it, but flag below — the file may belong to a governorate
            # we haven't declared in GOVERNORATES.
        try:
            d = pd.read_csv(p)
        except Exception as e:
            print(f"  [WARN] {p.name}: {e}", flush=True)
            continue

        # Inject governorate metadata from the filename
        meta = GOVERNORATES.get(gov_key, {"name": gov_key, "name_ar": "",
                                          "lat": float("nan"),
                                          "lon": float("nan")})
        d["governorate"] = meta["name"]
        d["latitude"]    = meta["lat"]
        d["longitude"]   = meta["lon"]

        frames.append(d)
        per_gov_counts[meta["name"]] = per_gov_counts.get(meta["name"], 0) + len(d)

    if not frames:
        raise SystemExit("No NASA file could be read.")

    df = pd.concat(frames, ignore_index=True)

    missing = [c for c in NASA_KEEP_COLUMNS if c not in df.columns]
    if missing:
        print(f"[WARN] NASA merged data is missing columns: {missing}",
              flush=True)
    df = df[[c for c in NASA_KEEP_COLUMNS if c in df.columns]].copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "governorate"])
    df = df.drop_duplicates(subset=["governorate", "date"], keep="last")

    if df.empty:
        print("[WARN] NASA merge produced 0 rows.", flush=True)
        return None

    # ---- Post-merge diagnostics -------------------------------------
    n_govs = df["governorate"].nunique()
    print(f"[INFO] Governorates in merged NASA data: {n_govs}", flush=True)
    if n_govs < len(GOVERNORATES):
        expected = set(GOVERNORATES[g]["name"] for g in GOVERNORATES)
        found = set(df["governorate"].unique())
        missing_govs = sorted(expected - found)
        print(f"[WARN] Missing governorates: {missing_govs}", flush=True)
    if unknown_keys:
        print(f"[WARN] Unknown governorate keys in filenames: {sorted(unknown_keys)}",
              flush=True)

    print("\n[INFO] Rows per governorate:", flush=True)
    for gov, n in sorted(per_gov_counts.items()):
        print(f"  {gov:<14} {n:>7,}", flush=True)

    # ---- Save --------------------------------------------------------
    start_year = int(df["date"].dt.year.min())
    end_year   = int(df["date"].dt.year.max())

    df = df.sort_values(["governorate", "date"]).reset_index(drop=True)

    ts = _timestamp()
    out_path = PROCESSED_DIR / f"{prefix}_{start_year}_{end_year}_{ts}.csv"

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\n[SAVED] {out_path.name}  rows={len(df):,}  "
          f"governorates={n_govs}  years={start_year}-{end_year}", flush=True)
    return out_path

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
SOURCE_REGISTRY = {
    "fao": build_fao_dataset,
    "nasa": build_nasa_dataset,
}
ALL_SOURCES = list(SOURCE_REGISTRY.keys())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge raw FAO/NASA CSVs into processed, project-ready files."
    )
    parser.add_argument(
        "--source", "-s",
        choices=ALL_SOURCES + ["all"],
        default="all",
        help="Which dataset(s) to build (default: all).",
    )
    parser.add_argument(
        "-f", "--force", action="store_true",
        help="Rebuild even if an output already exists.",
    )
    args = parser.parse_args(argv)

    sources = ALL_SOURCES if args.source == "all" else [args.source]

    print("============================================================", flush=True)
    print(f" FTL Syria — Build Dataset  (sources={sources}, force={args.force})",
          flush=True)
    print(f" Raw dir : {RAW_DIR}", flush=True)
    print(f" Out dir : {PROCESSED_DIR}", flush=True)
    print("============================================================", flush=True)

    for src in sources:
        print(f"\n>>> Building: {src}", flush=True)
        SOURCE_REGISTRY[src](force=args.force)

    print("\n============================================================", flush=True)
    print(" [DONE]", flush=True)
    print("============================================================", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())