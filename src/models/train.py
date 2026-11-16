#!/usr/bin/env python
# train.py — train per-crop DecisionTree + XGBoost, save .pkl for serving
#
# All configuration (feature_cols, target, paths, per-crop params) comes
# from config.yaml under the `training:` key.
#
# Inputs:  data/interim/train.csv, val.csv, test.csv
#          (produced by src/data/split_dataset.py)
# Outputs: models/{dt,xgb}_<crop>.pkl
#          models/feature_schema.json
#          models/training_summary.json
#          models/train.log   (append-only, timestamped)
#
# Usage:
#   python train.py
#   python train.py --crop Wheat
#   python train.py --model xgb
#   python train.py --crop Wheat --model dt
#   python train.py --mlflow

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import yaml

from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ══════════════════════════════════════════════════════════════════
# Config loading
# ══════════════════════════════════════════════════════════════════
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_FILE  = PROJECT_ROOT / "config.yaml"


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
# Logger
# ══════════════════════════════════════════════════════════════════
class Logger:
    def __init__(self, log_path: Path):
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._raw_write(
            f"\n{'═' * 72}\n"
            f"  SESSION START  {datetime.now(timezone.utc).isoformat()}\n"
            f"{'═' * 72}\n"
        )

    def _raw_write(self, text: str):
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(text)

    def _ts(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def info(self, msg: str = ""):
        line = f"[{self._ts()}] {msg}" if msg else ""
        print(line)
        self._raw_write(line + "\n")

    def banner(self, title: str):
        line = f"\n{'=' * 72}\n  {title}\n{'=' * 72}"
        print(line)
        self._raw_write(line + "\n")

    def error(self, msg: str):
        line = f"[{self._ts()}] ERROR: {msg}"
        print(line, file=sys.stderr)
        self._raw_write(line + "\n")

    def table(self, rows: list[list[str]], headers: list[str]):
        widths = [max([len(str(h))] + [len(str(r[i])) for r in rows])
                  for i, h in enumerate(headers)]
        header_line = "  ".join(f"{h:<{w}}" for h, w in zip(headers, widths))
        sep_line    = "  ".join("-" * w for w in widths)
        data_lines  = ["  ".join(f"{str(c):<{w}}"
                                 for c, w in zip(row, widths))
                       for row in rows]
        block = "\n".join([header_line, sep_line] + data_lines)
        print(block)
        self._raw_write(block + "\n")


# ══════════════════════════════════════════════════════════════════
# Configuration (from config.yaml `training:` section)
# ══════════════════════════════════════════════════════════════════
FEATURE_COLS = _cfg("training", "feature_cols", default=[
    "t2m_mean", "t2m_max_extreme", "t2m_min_extreme",
    "precip_total", "rh2m_mean", "solar_mean", "ws2m_mean",
])
ANOMALY_FEATURES = _cfg("training", "anomaly_features", default=[
    "yield_anomaly", "drought_year", "conflict_year", "climate_anomaly",
])
TARGET = _cfg("training", "target", default="yield_hg_ha")

INTERIM_DIR = PROJECT_ROOT / "data" / "interim"
MODELS_DIR  = PROJECT_ROOT / _cfg("training", "models_dir", default="models")
LOG_FILE    = PROJECT_ROOT / _cfg("training", "log_file",   default="models/train.log")

DEFAULT_PARAMS = _cfg("training", "default_params", default={
    "dt":  {"max_depth": 3, "min_samples_leaf": 2,
            "min_samples_split": 2, "ccp_alpha": 0.0},
    "xgb": {"n_estimators": 50, "max_depth": 2, "learning_rate": 0.05,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_weight": 3},
})

BEST_PARAMS = _cfg("training", "best_params", default={})


# ══════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════
def eval_metrics(actual, pred):
    return {
        "rmse":  float(np.sqrt(mean_squared_error(actual, pred))),
        "mae":   float(mean_absolute_error(actual, pred)),
        "r2":    float(r2_score(actual, pred)),
        "smape": float(np.mean(
            2 * np.abs(actual - pred) / (np.abs(actual) + np.abs(pred) + 1e-9)
        ) * 100),
    }


def _model_features() -> list[str]:
    """Canonical training-order feature list: climate + anomaly indicators."""
    return FEATURE_COLS + ANOMALY_FEATURES


def _verify_splits(log: Logger) -> dict[str, pd.DataFrame]:
    """Load and sanity-check data/interim/{train,val,test}.csv."""
    splits = {}
    for name in ("train", "val", "test"):
        p = INTERIM_DIR / f"{name}.csv"
        if not p.is_file():
            log.error(f"Missing {p}")
            log.error("Run: python -u src/data/split_dataset.py -f")
            raise SystemExit(2)
        df = pd.read_csv(p)
        if df.empty:
            log.error(f"{p.name} is empty")
            raise SystemExit(2)
        splits[name] = df
        log.info(f"  {name}.csv: {len(df):>5} rows, "
                 f"{df['crop'].nunique()} crops, "
                 f"years {df['year'].min()}–{df['year'].max()}")
    return splits


def crop_slug(crop: str) -> str:
    return crop.lower().replace(" ", "_").replace("-", "_")


def model_filename(family: str, crop: str) -> str:
    return f"{family}_{crop_slug(crop)}.pkl"


def build_model(family: str, params: dict):
    if family == "dt":
        return DecisionTreeRegressor(random_state=42, **params)
    if family == "xgb":
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=1,
            tree_method="hist",
            verbosity=0,
            **params,
        )
    raise ValueError(f"Unknown model family: {family}")


# ══════════════════════════════════════════════════════════════════
# Training routine
# ══════════════════════════════════════════════════════════════════
def train_crop_model(splits: dict[str, pd.DataFrame], crop: str, family: str,
                     log: Logger, use_mlflow: bool = False) -> dict:
    """Train one (crop, family) model. Return a summary dict."""

    features = _model_features()

    train = splits["train"][splits["train"]["crop"] == crop]
    val   = splits["val"]  [splits["val"]  ["crop"] == crop]
    test  = splits["test"] [splits["test"] ["crop"] == crop]

    if len(train) < 5:
        log.info(f"  [{family}] SKIPPED: only {len(train)} training rows")
        return {"crop": crop, "family": family,
                "skipped": f"only {len(train)} training rows"}
    if len(val) == 0 or len(test) == 0:
        log.info(f"  [{family}] SKIPPED: empty val ({len(val)}) "
                 f"or test ({len(test)})")
        return {"crop": crop, "family": family,
                "skipped": f"empty val ({len(val)}) or test ({len(test)})"}

    # Only use features actually present in the splits
    present = [c for c in features if c in train.columns]
    missing = [c for c in features if c not in train.columns]
    if missing:
        log.info(f"  [{family}] using {len(present)}/{len(features)} features "
                 f"(missing: {missing})")

    X_train, y_train = train[present], train[TARGET]
    X_val,   y_val   = val  [present], val  [TARGET]
    X_test,  y_test  = test [present], test [TARGET]

    X_full = pd.concat([X_train, X_val], axis=0)
    y_full = pd.concat([y_train, y_val], axis=0)

    params = BEST_PARAMS.get(crop, {}).get(family) or DEFAULT_PARAMS[family]
    model = build_model(family, params)
    model.fit(X_full, y_full)

    metrics = {}
    for name, X, y in [("train", X_train, y_train),
                       ("val",   X_val,   y_val),
                       ("test",  X_test,  y_test)]:
        m = eval_metrics(y.values, model.predict(X))
        for k, v in m.items():
            metrics[f"{name}_{k}"] = v

    out_path = MODELS_DIR / model_filename(family, crop)
    joblib.dump(model, out_path)

    importances = {}
    if hasattr(model, "feature_importances_"):
        imp = pd.Series(model.feature_importances_, index=present)
        imp = imp.sort_values(ascending=False)
        importances = {k: float(v) for k, v in imp.items()}

    if use_mlflow:
        try:
            import mlflow
            with mlflow.start_run(run_name=f"{family}_{crop}"):
                mlflow.set_tag("crop", crop)
                mlflow.set_tag("model_family", family)
                mlflow.log_params(params)
                for k, v in metrics.items():
                    mlflow.log_metric(k, v)
                for feat, imp_val in importances.items():
                    mlflow.log_metric(f"importance_{feat}", imp_val)
        except ImportError:
            pass

    log.info(f"  [{family}] saved → {out_path.relative_to(PROJECT_ROOT)}")
    log.info(f"  [{family}] params: {params}")
    log.info(f"  [{family}] n = {len(X_train)} train / "
             f"{len(X_val)} val / {len(X_test)} test")
    for split in ("train", "val", "test"):
        log.info(f"  [{family}] {split:<5}  "
                 f"rmse={metrics[f'{split}_rmse']:>10,.2f}  "
                 f"mae={metrics[f'{split}_mae']:>10,.2f}  "
                 f"r2={metrics[f'{split}_r2']:>7.3f}  "
                 f"smape={metrics[f'{split}_smape']:>7.2f}")
    if importances:
        top = list(importances.items())[:5]
        top_str = ", ".join(f"{k}={v:.2f}" for k, v in top)
        log.info(f"  [{family}] top importances: {top_str}")

    return {
        "crop":        crop,
        "family":      family,
        "params":      params,
        "features":    present,          # exact order used by the model
        "n_train":     len(X_train),
        "n_val":       len(X_val),
        "n_test":      len(X_test),
        "metrics":     metrics,
        "importances": importances,
        "path":        str(out_path.relative_to(PROJECT_ROOT)),
    }


def write_feature_schema(trained: list[dict]):
    """Write models/feature_schema.json for predict.py to validate against.

    IMPORTANT: feature_cols MUST be in the same order the models were trained
    with. XGBoost raises 'feature_names mismatch' if the order differs, even
    when the set of columns is identical. So we preserve training order from
    _model_features() rather than sorting alphabetically.
    """
    canonical = _model_features()

    if trained:
        # Union of features actually used across trained models,
        # BUT in the canonical training order
        present = set()
        for s in trained:
            if "features" in s:
                present.update(s["features"])
        used = [c for c in canonical if c in present]
    else:
        used = canonical

    crops    = sorted({s["crop"]   for s in trained if "skipped" not in s})
    families = sorted({s["family"] for s in trained if "skipped" not in s})

    schema = {
        "feature_cols": used,
        "target":       TARGET,
        "crops":        crops,
        "families":     families,
        "models": {
            crop: {fam: model_filename(fam, crop)
                   for fam in families
                   if (MODELS_DIR / model_filename(fam, crop)).is_file()}
            for crop in crops
        },
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = MODELS_DIR / "feature_schema.json"
    path.write_text(json.dumps(schema, indent=2, ensure_ascii=False),
                    encoding="utf-8")
    return path


# ══════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════
def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Train per-crop DecisionTree + XGBoost models, save .pkl. "
                    "All configuration comes from config.yaml `training:`."
    )
    p.add_argument("--crop", type=str, default=None,
                   choices=list(BEST_PARAMS.keys()) or None,
                   help="Train only this crop (default: all in the split)")
    p.add_argument("--model", type=str, default="both",
                   choices=["dt", "xgb", "both"],
                   help="Which model family (default: both)")
    p.add_argument("--mlflow", action="store_true",
                   help="Also log runs to MLflow (requires mlflow installed)")
    p.add_argument("--log-file", type=Path, default=LOG_FILE,
                   help=f"Append log destination (default: {LOG_FILE})")
    p.add_argument("--reset-log", action="store_true",
                   help="Truncate the log file before starting (default: append)")
    args = p.parse_args(argv)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    if args.reset_log and args.log_file.exists():
        args.log_file.unlink()
    log = Logger(args.log_file)

    log.banner("FTL Syria — train.py")
    log.info(f"Config file:  {CONFIG_FILE}")
    log.info(f"Split dir:    {INTERIM_DIR}")
    log.info(f"Models dir:   {MODELS_DIR}")
    log.info(f"Log file:     {args.log_file}")
    log.info(f"Crop filter:  {args.crop or 'all'}")
    log.info(f"Model filter: {args.model}")
    log.info(f"MLflow:       {'enabled' if args.mlflow else 'disabled'}")
    log.info(f"Target:       {TARGET}")
    log.info(f"Features ({len(FEATURE_COLS)} climate + "
             f"{len(ANOMALY_FEATURES)} anomaly):")
    for c in FEATURE_COLS:
        log.info(f"  climate  {c}")
    for c in ANOMALY_FEATURES:
        log.info(f"  anomaly  {c}")

    log.banner("Loading pre-split data")
    try:
        splits = _verify_splits(log)
    except SystemExit as e:
        return int(e.code) if e.code else 1

    available_crops = sorted(splits["train"]["crop"].unique())
    log.info(f"Crops in train split: {available_crops}")

    if args.crop and args.crop not in available_crops:
        log.error(f"--crop {args.crop} not in training data. "
                  f"Available: {available_crops}")
        return 2

    crops    = [args.crop] if args.crop else available_crops
    families = ["dt", "xgb"] if args.model == "both" else [args.model]

    summaries = []
    for crop in crops:
        log.banner(f"Crop: {crop}")
        if crop not in BEST_PARAMS:
            log.info(f"  [note] no BEST_PARAMS entry for '{crop}' — "
                     f"using default_params")
        for family in families:
            summary = train_crop_model(
                splits, crop, family, log, use_mlflow=args.mlflow,
            )
            summaries.append(summary)

    trained = [s for s in summaries if "skipped" not in s]
    skipped = [s for s in summaries if "skipped" in s]

    if trained:
        schema_path = write_feature_schema(trained)
        log.info(f"\n[saved] {schema_path}")

    summary_path = MODELS_DIR / "training_summary.json"
    summary_path.write_text(
        json.dumps(summaries, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    log.info(f"[saved] {summary_path}")

    log.banner(f"Summary — {len(trained)} trained, {len(skipped)} skipped")

    if trained:
        rows = []
        for s in trained:
            m = s["metrics"]
            rows.append([
                s["crop"], s["family"],
                f"{m['train_rmse']:,.0f}",
                f"{m['val_rmse']:,.0f}",
                f"{m['test_rmse']:,.0f}",
                f"{m['test_r2']:.3f}",
            ])
        log.table(
            rows,
            headers=["crop", "model", "train_rmse", "val_rmse",
                     "test_rmse", "test_r2"],
        )

    if skipped:
        log.info("\nSkipped:")
        for s in skipped:
            log.info(f"  {s['crop']:<16} {s['family']:<5} — {s['skipped']}")

    log.info(f"\nModels in: {MODELS_DIR}")
    log.info(f"Full log:  {args.log_file}")
    log.info("Session complete.\n")

    return 0 if not skipped else 1


if __name__ == "__main__":
    sys.exit(main())