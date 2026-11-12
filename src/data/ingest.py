#!/usr/bin/env python
import argparse
import json
import sys
import time
from calendar import monthrange
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from requests.exceptions import ConnectionError, Timeout

# ------------------------------------------------------------------
# Paths (anchored to project root, not CWD)
# ------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
TOKEN_FILE = PROJECT_ROOT / ".faostat_token"

# ------------------------------------------------------------------
# FAOSTAT constants
# ------------------------------------------------------------------
FAO_BASE_URL = "https://faostatservices.fao.org/api/v1/en"
FAO_DATA_ENDPOINT = f"{FAO_BASE_URL}/data/QCL"

DEFAULT_START_YEAR = 1991
DEFAULT_CURRENT_YEAR = date.today().year
MAX_RETRIES = 3
MAX_DEBUG_BYTES = 2000

# ------------------------------------------------------------------
# NASA POWER constants
# ------------------------------------------------------------------
NASA_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
NASA_PARAMETERS = [
    "T2M", "T2M_MAX", "T2M_MIN",
    "PRECTOTCORR", "RH2M",
    "ALLSKY_SFC_SW_DWN", "WS2M",
]
NASA_COLUMN_MAPPING = {
    "T2M": "t2m",
    "T2M_MAX": "t2m_max",
    "T2M_MIN": "t2m_min",
    "PRECTOTCORR": "prectotcorr",
    "RH2M": "rh2m",
    "ALLSKY_SFC_SW_DWN": "solar_kwh_m2",
    "WS2M": "ws2m",
}

GOVERNORATES = [
    {"name": "Aleppo",     "name_ar": "حلب",       "lat": 36.17, "lon": 37.61},
    {"name": "Al-Hasakah", "name_ar": "الحسكة",    "lat": 36.42, "lon": 40.91},
    {"name": "Ar-Raqqah",  "name_ar": "الرقة",      "lat": 36.01, "lon": 38.91},
    {"name": "As-Suwayda", "name_ar": "السويداء",  "lat": 32.77, "lon": 36.92},
    {"name": "Daraa",      "name_ar": "درعا",       "lat": 32.83, "lon": 36.17},
    {"name": "Deir ez-Zor","name_ar": "دير الزور", "lat": 35.25, "lon": 40.28},
    {"name": "Hama",       "name_ar": "حماة",       "lat": 35.30, "lon": 37.25},
    {"name": "Homs",       "name_ar": "حمص",        "lat": 34.26, "lon": 38.16},
    {"name": "Idlib",      "name_ar": "إدلب",       "lat": 35.83, "lon": 36.68},
    {"name": "Latakia",    "name_ar": "اللاذقية",   "lat": 35.57, "lon": 36.01},
    {"name": "Quneitra",   "name_ar": "القنيطرة",   "lat": 33.02, "lon": 35.79},
    {"name": "Tartus",     "name_ar": "طرطوس",      "lat": 34.95, "lon": 36.09},
    {"name": "Damascus",   "name_ar": "دمشق",       "lat": 33.46, "lon": 37.05},
]


# ==================================================================
# Shared helpers
# ==================================================================
def _preview_bytes(content: bytes, limit: int = MAX_DEBUG_BYTES) -> str:
    if not content:
        return "<empty body>"
    head = content[:limit]
    try:
        text = head.decode("utf-8", errors="replace")
    except Exception:
        text = repr(head)
    if len(content) > limit:
        text += f"\n... <truncated, total {len(content)} bytes>"
    return text


def log_response(response: requests.Response) -> None:
    print("   --- HTTP response details ---", flush=True)
    print(f"   URL              : {response.url}", flush=True)
    print(f"   Status code      : {response.status_code} {response.reason}", flush=True)
    print(f"   Elapsed          : {response.elapsed}", flush=True)
    print(f"   Encoding         : {response.encoding}", flush=True)
    print(f"   Content-Type     : {response.headers.get('Content-Type', '<none>')}", flush=True)
    print(f"   Content-Length   : {response.headers.get('Content-Length', '<none>')}", flush=True)
    print(f"   Bytes received   : {len(response.content)}", flush=True)

    for h in ("Date", "Server", "X-RateLimit-Remaining", "Retry-After"):
        if h in response.headers:
            print(f"   Header {h:<18}: {response.headers[h]}", flush=True)

    ctype = (response.headers.get("Content-Type") or "").lower()
    if "json" in ctype or response.content[:1] in (b"{", b"["):
        try:
            preview = json.dumps(response.json(), indent=2, ensure_ascii=False)
            if len(preview) > MAX_DEBUG_BYTES:
                preview = preview[:MAX_DEBUG_BYTES] + "\n... <truncated>"
            print("   --- JSON body (truncated) ---", flush=True)
            print(preview, flush=True)
            return
        except Exception:
            pass

    print("   --- Body preview (truncated) ---", flush=True)
    print(_preview_bytes(response.content), flush=True)


def looks_like_csv(response: requests.Response) -> bool:
    ctype = (response.headers.get("Content-Type") or "").lower()
    if "csv" in ctype:
        return True
    first_line = response.content.split(b"\n", 1)[0]
    return b"," in first_line and b"{" not in first_line[:1]


def month_iter(start_year: int, end_year: int):
    """Yield (year, month) pairs covering the inclusive range."""
    for y in range(start_year, end_year + 1):
        for m in range(1, 13):
            yield y, m


def month_range_dates(year: int, month: int) -> tuple[str, str]:
    """Return (YYYYMMDD, YYYYMMDD) for first and last day of the month,
    clipped to today if the month is the current month."""
    last_day = monthrange(year, month)[1]
    first = date(year, month, 1)
    last = date(year, month, last_day)
    today = date.today()
    if (year, month) == (today.year, today.month):
        last = min(last, today)
    return first.strftime("%Y%m%d"), last.strftime("%Y%m%d")


def resolve_year_range(args) -> tuple[int, int, int | None]:
    """Return (start_year, end_year, single_year_or_None)."""
    if args.year:
        return args.year, args.year, args.year
    if args.start_year > args.end_year:
        raise SystemExit(
            f"--start-year ({args.start_year}) must be <= "
            f"--end-year ({args.end_year})."
        )
    return args.start_year, args.end_year, None


# ==================================================================
# FAOSTAT
# ==================================================================
def get_or_create_token() -> str:
    if TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text(encoding="utf-8").strip()
        if token:
            print(f"[INFO] Loaded existing token from {TOKEN_FILE}", flush=True)
            return token

    if not sys.stdin.isatty():
        print("[INFO] No token file and non-interactive shell — proceeding without token.",
              flush=True)
        return ""

    print("\n[INFO] FAOSTAT API token/key not found locally.", flush=True)
    user_token = input(
        "Enter your FAOSTAT API Token/Key (or press Enter to skip if public): "
    ).strip()
    TOKEN_FILE.write_text(user_token, encoding="utf-8")
    print(f"[INFO] Token saved to {TOKEN_FILE}\n", flush=True)
    return user_token


def download_fao(start_year: int, end_year: int, force: bool) -> None:
    """Year-by-year FAOSTAT QCL download for Syria."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    token = get_or_create_token()
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    print(f"\n[INFO] FAOSTAT: extracting Syria QCL ({start_year}..{end_year})...\n",
          flush=True)

    for year in range(start_year, end_year + 1):
        file_name = f"QCL_Syria_AllCrops_{year}.csv"
        file_path = RAW_DATA_DIR / file_name

        if file_path.exists() and not force:
            print(f"[SKIP] {file_name} exists. Use -f to re-download.", flush=True)
            continue

        print(f"[INFO] FAOSTAT year {year} -> {file_name} ...", flush=True)
        params = {"area": "210", "years": str(year), "output_type": "table"}

        success = False
        for attempt in range(1, MAX_RETRIES + 1):
            print(f"   [ATTEMPT {attempt}/{MAX_RETRIES}] GET {FAO_DATA_ENDPOINT} "
                  f"params={params}", flush=True)
            try:
                r = requests.get(FAO_DATA_ENDPOINT, headers=headers,
                                 params=params, stream=True, timeout=25)
                log_response(r)

                if r.status_code == 200 and len(r.content) > 100:
                    if not looks_like_csv(r):
                        print("   [WARN] 200 OK but body is not CSV — skipping.",
                              flush=True)
                        time.sleep(2)
                        continue
                    with open(file_path, "wb") as f:
                        for chunk in r.iter_content(chunk_size=8192):
                            f.write(chunk)
                    print(f"[SUCCESS] {file_name} ({file_path.stat().st_size} bytes)",
                          flush=True)
                    success = True
                    break
                else:
                    print(f"   [WARN] Attempt {attempt}/{MAX_RETRIES}: "
                          f"status {r.status_code}", flush=True)
                    time.sleep(2)
            except (ConnectionError, Timeout) as e:
                print(f"   [WARN] Attempt {attempt}/{MAX_RETRIES}: {e}", flush=True)
                time.sleep(2)
            except Exception as e:
                print(f"   [WARN] Attempt {attempt}/{MAX_RETRIES}: {e}", flush=True)
                time.sleep(2)

        if not success:
            print(f"   [ERROR] FAOSTAT {year} failed after {MAX_RETRIES} attempts.",
                  flush=True)


# ==================================================================
# NASA POWER (month-by-month per governorate)
# ==================================================================
def _nasa_file_path(gov_name: str, year: int, month: int) -> Path:
    slug = gov_name.lower().replace(" ", "_").replace("-", "_")
    return RAW_DATA_DIR / f"nasa_daily_{slug}_{year}_{month:02d}.csv"


def fetch_nasa_month(gov: dict, year: int, month: int,
                     force: bool) -> Path | None:
    """Download one governorate-month of NASA POWER daily data."""
    file_path = _nasa_file_path(gov["name"], year, month)

    if file_path.exists() and not force:
        print(f"  [SKIP] {file_path.name} exists.", flush=True)
        return file_path

    start, end = month_range_dates(year, month)
    if start > date.today().strftime("%Y%m%d"):
        print(f"  [SKIP] {file_path.name} is in the future.", flush=True)
        return None

    params = {
        "latitude":   gov["lat"],
        "longitude":  gov["lon"],
        "start":      start,
        "end":        end,
        "parameters": ",".join(NASA_PARAMETERS),
        "community":  "AG",
        "format":     "JSON",
    }

    print(f"  [GET] {gov['name']} {year}-{month:02d} ({start}..{end})", flush=True)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(NASA_URL, params=params, timeout=120)
            log_response(r)
            if r.status_code != 200:
                print(f"   [WARN] attempt {attempt}/{MAX_RETRIES}: "
                      f"status {r.status_code}", flush=True)
                time.sleep(2)
                continue

            data = r.json()
            param_data = data.get("properties", {}).get("parameter", {})
            if not param_data:
                print("   [WARN] NASA response has no 'parameter' data.", flush=True)
                time.sleep(2)
                continue

            df = pd.DataFrame()
            for api_name, col_name in NASA_COLUMN_MAPPING.items():
                if api_name in param_data:
                    df[col_name] = pd.Series(param_data[api_name])

            if df.empty:
                print("   [WARN] No matching parameters in response.", flush=True)
                time.sleep(2)
                continue

            df.index = pd.to_datetime(df.index, format="%Y%m%d")
            df.index.name = "date"
            df = df.replace(-999.0, pd.NA)

            df = df.reset_index()
            df.insert(0, "governorate", gov["name"])
            df.insert(1, "governorate_ar", gov["name_ar"])
            df.insert(2, "latitude", gov["lat"])
            df.insert(3, "longitude", gov["lon"])

            df.to_csv(file_path, index=False)
            print(f"  [SAVED] {file_path.name} ({len(df)} rows)", flush=True)
            return file_path

        except (ConnectionError, Timeout) as e:
            print(f"   [WARN] attempt {attempt}/{MAX_RETRIES}: {e}", flush=True)
            time.sleep(2)
        except Exception as e:
            print(f"   [WARN] attempt {attempt}/{MAX_RETRIES}: {e}", flush=True)
            time.sleep(2)

    print(f"  [ERROR] {gov['name']} {year}-{month:02d} failed.", flush=True)
    return None


def download_nasa(start_year: int, end_year: int, force: bool) -> None:
    """Download NASA POWER daily data per governorate, saved by month."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    ref_path = RAW_DATA_DIR / "syria_governorates_reference.csv"
    if not ref_path.exists() or force:
        pd.DataFrame(GOVERNORATES).to_csv(ref_path, index=False)
        print(f"[SAVED] {ref_path}", flush=True)

    print(f"\n[INFO] NASA POWER: {len(GOVERNORATES)} governorates, "
          f"months {start_year}-01 .. {end_year}-12\n", flush=True)

    saved = skipped = failed = 0
    for gov in GOVERNORATES:
        print(f"\n[GOV] {gov['name']}", flush=True)
        for y, m in month_iter(start_year, end_year):
            result = fetch_nasa_month(gov, y, m, force=force)
            if result is None:
                if _nasa_file_path(gov["name"], y, m).exists():
                    skipped += 1
                else:
                    failed += 1
            else:
                saved += 1

    print(f"\n[SUMMARY] NASA: saved={saved} skipped={skipped} failed={failed}",
          flush=True)


# ==================================================================
# Source registry
# ==================================================================
SOURCE_REGISTRY = {
    "fao": download_fao,
    "nasa": download_nasa,
}
ALL_SOURCES = list(SOURCE_REGISTRY.keys())


# ==================================================================
# CLI
# ==================================================================
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Ingest climate/agriculture datasets into data/raw."
    )
    parser.add_argument(
        "--source", "-s",
        choices=ALL_SOURCES + ["all"],
        default="all",
        help="Which upstream dataset to download (default: all).",
    )
    parser.add_argument("-y", "--year", type=int,
                        help="Download a single specific year (e.g. 2020).")
    parser.add_argument("--start-year", type=int, default=DEFAULT_START_YEAR,
                        help=f"Start year for range (default: {DEFAULT_START_YEAR}).")
    parser.add_argument("--end-year", type=int, default=DEFAULT_CURRENT_YEAR,
                        help=f"End year for range (default: {DEFAULT_CURRENT_YEAR}).")
    parser.add_argument("-f", "--force", action="store_true",
                        help="Force re-download and overwrite existing files.")
    return parser


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    start_year, end_year, _ = resolve_year_range(args)
    sources = ALL_SOURCES if args.source == "all" else [args.source]

    print("============================================================", flush=True)
    print(f" FTL Syria Ingest — sources={sources} ({start_year}..{end_year})",
          flush=True)
    print(f" Output dir: {RAW_DATA_DIR}", flush=True)
    print("============================================================", flush=True)

    for src in sources:
        print(f"\n>>> Running source: {src}", flush=True)
        SOURCE_REGISTRY[src](
            start_year=start_year,
            end_year=end_year,
            force=args.force,
        )

    print("\n============================================================", flush=True)
    print(" [DONE]", flush=True)
    print("============================================================", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())