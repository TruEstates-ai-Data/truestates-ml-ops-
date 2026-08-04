import ast
import logging
import os
import sys
from math import atan2, cos, radians, sin, sqrt
from pathlib import Path
from typing import Optional
import pandas as pd
import s3fs
from pandas.tseries.offsets import MonthEnd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
CODE_DIR = Path(__file__).resolve().parent
if str(CODE_DIR) not in sys.path:
    sys.path.append(str(CODE_DIR))
from price_predictor_pipeline_h import predict_property_price
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
CONFIG_PATH = CODE_DIR / "config.yaml"


def load_config():
    import yaml
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"config.yaml not found at {CONFIG_PATH}")
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def _setup_r2_env(cfg):
    storage = cfg.get("cloud_storage", {})
    if storage.get("endpoint_url"):
        os.environ["AWS_ENDPOINT_URL"] = storage["endpoint_url"]
        endpoint_root = storage["endpoint_url"].split("/truestates")[0]
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = endpoint_root
    if storage.get("aws_access_key_id"):
        os.environ["AWS_ACCESS_KEY_ID"] = storage["aws_access_key_id"]
    if storage.get("aws_secret_access_key"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = storage["aws_secret_access_key"]


def _resolve_path(base, path_value):
    path_value = str(path_value)
    if path_value.startswith("s3://"):
        return path_value
    return os.path.join(str(base), path_value)


def _s3_join(base, *parts):
    base = str(base).rstrip("/")
    if base.startswith("s3://"):
        return "/".join([base] + [str(p).strip("/") for p in parts])
    return str(Path(base).joinpath(*parts))


def _is_s3(path):
    return str(path).startswith("s3://")


def _file_name(path):
    return str(path).rsplit("/", 1)[-1]


config = load_config()
_setup_r2_env(config)
_paths = config["paths"]
_base = _paths.get("base_dir", "s3://dubai")
_model_req_dir = _s3_join(_base, "data/model_requirements")
_processed_dir = _s3_join(_base, "data/processed")
INPUT_RANGES_CSV = _s3_join(_model_req_dir, "input_ranges.csv")
FORECAST_PATH = _s3_join(_model_req_dir, "final_chronos_forecasts.csv")
HISTORIC_PATH = _s3_join(_processed_dir, "historic_df.csv")
NEWS_PATH = _s3_join(_model_req_dir, "adjusted_macro_forecast.csv")
LATEST_COMBINED_PATH = _s3_join(_processed_dir, "latest_combined_data.parquet")
OLD_DIR = _s3_join(_base, config.get("archive", {}).get("folder_name", "old_files"))
_fs = s3fs.S3FileSystem()
EARTH_RADIUS_KM = 6371.0

app = FastAPI(
    title="TruEstates API v3.0 - orchestrated",
    description="Dubai Real Estate API with regression + forecast integration",
    version="v3.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AREA_COORDS = {
    "Al Barsha First": (25.1150, 55.2050),
    "Al Barshaa South Second": (25.0921, 55.2410),
    "Al Barshaa South Third": (25.0760, 55.2150),
    "Al Barsha South Fourth": (25.0610, 55.2360),
    "Al Barsha South Fifth": (25.0485, 55.2341),
    "Al Hebiah First": (25.0334, 55.2205),
    "Al Hebiah Second": (25.0420, 55.2450),
    "Al Hebiah Third": (25.0160, 55.2380),
    "Al Hebiah Fourth": (25.0210, 55.2150),
    "Al Hebiah Sixth": (24.9950, 55.2320),
    "Al Khairan First": (25.1850, 55.3550),
    "Al Kifaf": (25.2350, 55.2950),
    "Al Merkadh": (25.1480, 55.3050),
    "Al Thanyah Third": (25.0850, 55.1550),
    "Al Thanyah Fifth": (25.0685, 55.1450),
    "Al Warsan First": (25.1650, 55.4150),
    "Al Yelayiss 2": (24.9650, 55.2650),
    "Bukadra": (25.1850, 55.3350),
    "Burj Khalifa": (25.1972, 55.2744),
    "Business Bay": (25.1850, 55.2750),
    "Hadaeq Sheikh Mohammed Bin Rashid": (25.1150, 55.2950),
    "Jabal Ali": (25.0000, 55.0500),
    "Jabal Ali First": (25.0220, 55.1050),
    "Jabal Ali Industrial Second": (24.9850, 55.1250),
    "Madinat Al Mataar": (24.8950, 55.1550),
    "Madinat Dubai Almelaheyah": (25.2650, 55.2750),
    "Madinat Hind 4": (25.0250, 55.4550),
    "Marsa Dubai": (25.0780, 55.1350),
    "Me'Aisem First": (25.0350, 55.1950),
    "Nadd Hessa": (25.1250, 55.3850),
    "Palm Deira": (25.3150, 55.3050),
    "Palm Jumeirah": (25.1124, 55.1390),
    "Ras Al Khor Industrial First": (25.1950, 55.3650),
    "Wadi Al Safa 2": (25.1200, 55.3700),
    "Wadi Al Safa 3": (25.0850, 55.3250),
    "Wadi Al Safa 4": (25.1450, 55.3050),
    "Wadi Al Safa 5": (25.0950, 55.3650),
    "Warsan Fourth": (25.1550, 55.4250),
    "Zaabeel First": (25.2200, 55.2850),
    "Zaabeel Second": (25.2050, 55.2950),
}

# Areas that don't have their own trained model / enough data. When an area is
# requested that has no individual data, we fall back to using its proxy /
# group area's data & model instead.
AREA_PROXY_MAP = {
    "Al Barsha First": "proxy1",
    "Al Hebiah Second": "proxy1",
    "Al Hebiah Sixth": "proxy1",
    "Al Hebiah Third": "proxy1",
    "Madinat Hind 4": "proxy1",
    "Wadi Al Safa 3": "proxy1",
    "Wadi Al Safa 4": "proxy1",
    "Wadi Al Safa 7": "proxy1",
    "Bukadra": "proxy2",
    "Ras Al Khor Industrial First": "proxy2",
    "Jumeirah First": "proxy2",
    "Palm Deira": "proxy2",
    "Al Thanyah Third": "proxy3",
    "Jabal Ali Industrial Second": "proxy3",
    "Al Kifaf": "G1",
    "Warsan Fourth": "G3",
    "Jabal Ali": "G3",
    "Zaabeel Second": "G4",
    "Zaabeel First": "G4",
}

PROXY_REVERSE_MAP = {}
_mm = config.get("market_mappings", {})
for g_name, areas in _mm.get("groups", {}).items():
    PROXY_REVERSE_MAP[g_name.lower()] = [str(a).strip() for a in areas]
for p_name, areas in _mm.get("proxies", {}).items():
    PROXY_REVERSE_MAP[p_name.lower()] = [str(a).strip() for a in areas]
def _path_exists(path):
    path = str(path)
    if _is_s3(path):
        return _fs.exists(path)
    return Path(path).exists()


def _read_csv(path):
    path = str(path)
    if _is_s3(path):
        with _fs.open(path, "rb") as f:
            return pd.read_csv(f)
    return pd.read_csv(path)


def _read_parquet(path):
    path = str(path)
    if _is_s3(path):
        with _fs.open(path, "rb") as f:
            return pd.read_parquet(f)
    return pd.read_parquet(path)


def load_csv_with_fallback(primary_path, parse_month=True):
    primary_path = str(primary_path)
    name = _file_name(primary_path)
    try:
        df = _read_csv(primary_path)
    except Exception as e:
        logger.warning(f"Primary load failed for {name}: {e}")
        fallback_path = _s3_join(OLD_DIR, f"old_{name}")
        if _path_exists(fallback_path):
            df = _read_csv(fallback_path)
        else:
            return pd.DataFrame()
    if parse_month and "month" in df.columns:
        df["month"] = pd.to_datetime(df["month"], errors="coerce")
    return df


def load_parquet_with_fallback(primary_path):
    primary_path = str(primary_path)
    name = _file_name(primary_path)
    try:
        return _read_parquet(primary_path)
    except Exception as e:
        logger.warning(f"Primary parquet load failed for {name}: {e}")
        fallback_path = _s3_join(OLD_DIR, f"old_{name}")
        if _path_exists(fallback_path):
            return _read_parquet(fallback_path)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Startup data loads
# ---------------------------------------------------------------------------
logger.info(f"Startup: loading forecast data from {FORECAST_PATH}")
forecast_df = load_csv_with_fallback(FORECAST_PATH, parse_month=True)
if forecast_df.empty:
    logger.warning("Startup: forecast_df is EMPTY after load (check FORECAST_PATH / permissions).")
else:
    logger.info(f"Startup: forecast_df loaded with {len(forecast_df)} rows.")
if not forecast_df.empty:
    if "area_name" in forecast_df.columns and "area" not in forecast_df.columns:
        forecast_df = forecast_df.rename(columns={"area_name": "area"})
    if "predicted_monthly_price" in forecast_df.columns and "predictions" not in forecast_df.columns:
        forecast_df = forecast_df.rename(columns={"predicted_monthly_price": "predictions"})

logger.info(f"Startup: loading historic data from {HISTORIC_PATH}")
historic_df = load_csv_with_fallback(HISTORIC_PATH, parse_month=True)
if historic_df.empty:
    logger.warning("Startup: historic_df is EMPTY after load (check HISTORIC_PATH / permissions).")
else:
    logger.info(f"Startup: historic_df loaded with {len(historic_df)} rows.")

logger.info(f"Startup: loading news/macro data from {NEWS_PATH}")
news_df = load_csv_with_fallback(NEWS_PATH, parse_month=True)
if news_df.empty:
    logger.warning("Startup: news_df is EMPTY after load (check NEWS_PATH / permissions).")
else:
    logger.info(f"Startup: news_df loaded with {len(news_df)} rows.")

logger.info(f"Startup: loading latest combined data from {LATEST_COMBINED_PATH}")
latest_combined_df = load_parquet_with_fallback(LATEST_COMBINED_PATH)
if latest_combined_df.empty:
    logger.warning("Startup: latest_combined_df is EMPTY after load (check LATEST_COMBINED_PATH / permissions).")
else:
    logger.info(f"Startup: latest_combined_df loaded with {len(latest_combined_df)} rows.")

# input_ranges.csv -> one row per area (including proxy/group rows like
# "Proxy1", "G3"). Every column except area_name_en/pre_encode_columns/
# procedure_area_min/procedure_area_max holds a Python-list-literal string of
# allowed values for that area; the FIRST element is that area's default.
input_ranges_df = load_csv_with_fallback(INPUT_RANGES_CSV, parse_month=False)

_INPUT_RANGES_NON_LIST_COLS = {"area_name_en", "pre_encode_columns", "procedure_area_min", "procedure_area_max"}


def _parse_list_cell(val):
    """Parse a cell like "['A', 'B', 'C']" into a real python list. Falls back
    to a single-element list if it isn't a valid list literal."""
    if pd.isna(val):
        return []
    if isinstance(val, list):
        return val
    try:
        parsed = ast.literal_eval(str(val).strip())
        return parsed if isinstance(parsed, list) else [parsed]
    except Exception:
        return [val]


def _parse_float_cell(val):
    try:
        if pd.isna(val):
            return None
        return float(val)
    except Exception:
        return None


# area_name_en (lowercased) -> {column: [allowed values...]} plus
# "_procedure_area_min" / "_procedure_area_max" scalars.
INPUT_RANGES_BY_AREA = {}
if not input_ranges_df.empty and "area_name_en" in input_ranges_df.columns:
    for _, row in input_ranges_df.iterrows():
        area_key = str(row["area_name_en"]).strip().lower()
        entry = {}
        for col in input_ranges_df.columns:
            if col in _INPUT_RANGES_NON_LIST_COLS:
                continue
            entry[col] = _parse_list_cell(row[col])
        entry["_procedure_area_min"] = _parse_float_cell(row.get("procedure_area_min"))
        entry["_procedure_area_max"] = _parse_float_cell(row.get("procedure_area_max"))
        INPUT_RANGES_BY_AREA[area_key] = entry
logger.info(f"Startup: loaded input_ranges.csv defaults for {len(INPUT_RANGES_BY_AREA)} areas.")

# Global (all-areas) first-value fallback, used only if a requested area/proxy
# is somehow missing from input_ranges.csv entirely.
INPUT_RANGE_DEFAULTS_GLOBAL = {}
if not input_ranges_df.empty:
    for col in input_ranges_df.columns:
        if col in _INPUT_RANGES_NON_LIST_COLS:
            continue
        for _, row in input_ranges_df.iterrows():
            vals = _parse_list_cell(row[col])
            if vals:
                INPUT_RANGE_DEFAULTS_GLOBAL[col] = vals[0]
                break


def get_area_input_ranges(area_name: str) -> dict:
    return INPUT_RANGES_BY_AREA.get(str(area_name).strip().lower(), {})


# Fields that are NEVER accepted as request input anymore -- always resolved
# from input_ranges.csv (first value for the area).
DEFAULT_ONLY_FIELDS = {
    "Locality Zone",
    "Price Tier",
    "Developer Tier",
    "Reputation",
    "Score",
}

if not news_df.empty and "area_name_en" in news_df.columns:
    news_df["area_name_en"] = news_df["area_name_en"].astype(str).str.strip()
if not latest_combined_df.empty:
    latest_combined_df.columns = [c.strip() for c in latest_combined_df.columns]
    if "area_name_en" in latest_combined_df.columns:
        latest_combined_df["area_name_en"] = latest_combined_df["area_name_en"].astype(str).str.strip()
    if "instance_date" in latest_combined_df.columns:
        latest_combined_df["instance_date"] = pd.to_datetime(latest_combined_df["instance_date"], errors="coerce")
        latest_combined_df["month"] = latest_combined_df["instance_date"] + MonthEnd(0)
    elif "month" in latest_combined_df.columns:
        latest_combined_df["month"] = pd.to_datetime(latest_combined_df["month"], errors="coerce") + MonthEnd(0)

if not latest_combined_df.empty and "area_name_en" in latest_combined_df.columns:
    VALID_AREAS = sorted(latest_combined_df["area_name_en"].dropna().unique().tolist())
else:
    VALID_AREAS = []
VALID_AREAS_SET = set(VALID_AREAS)
logger.info(f"Startup: {len(VALID_AREAS)} areas loaded.")


def compute_latest_month_end(df: pd.DataFrame):
    """Return the last (max) month-end timestamp found in latest_combined_df.
    Raises ValueError if it can't be determined (empty df, missing column,
    or no valid dates) so the caller can log the exact reason and fall back."""
    if df.empty:
        raise ValueError("latest_combined_df is empty; cannot determine latest month.")
    if "month" not in df.columns:
        raise ValueError("latest_combined_df has no 'month' column; cannot determine latest month.")
    max_month = df["month"].dropna().max()
    if pd.isna(max_month):
        raise ValueError("latest_combined_df has no valid (non-null) 'month' values.")
    return pd.to_datetime(max_month) + MonthEnd(0)


try:
    _latest_month_end = compute_latest_month_end(latest_combined_df)
    PIVOT_DATE = _latest_month_end.strftime("%Y-%m-%d")
    DEFAULT_YEAR = int(_latest_month_end.year)
    DEFAULT_MONTH = int(_latest_month_end.month)
    logger.info(
        f"Startup: pivot date dynamically set to {PIVOT_DATE} "
        f"(latest month in latest_combined_data.parquet). "
        f"Default year/month for requests -> {DEFAULT_YEAR}/{DEFAULT_MONTH}."
    )
except Exception as e:
    _now_month_end = pd.Timestamp.now() + MonthEnd(0)
    PIVOT_DATE = _now_month_end.strftime("%Y-%m-%d")
    DEFAULT_YEAR = int(pd.Timestamp.now().year)
    DEFAULT_MONTH = int(pd.Timestamp.now().month)
    logger.warning(
        f"Startup: could not derive pivot date / default year-month from "
        f"latest_combined_data.parquet ({type(e).__name__}: {e}); "
        f"falling back to current date-based values -> "
        f"PIVOT_DATE={PIVOT_DATE}, DEFAULT_YEAR={DEFAULT_YEAR}, DEFAULT_MONTH={DEFAULT_MONTH}."
    )

# Diagnostic only (no logic change): confirm historic_df.csv / final_chronos_forecasts.csv
# actually have data reaching PIVOT_DATE. If latest_combined_data.parquet is more
# up-to-date than these two files, every /forecast request will fail at the
# pivot-month lookup with "Model did not return a prediction for pivot month...".
try:
    _pivot_check_dt = pd.to_datetime(PIVOT_DATE) + MonthEnd(0)
    if not historic_df.empty and "month" in historic_df.columns:
        _hist_max = historic_df["month"].dropna().max()
        if pd.notna(_hist_max):
            _hist_max = pd.to_datetime(_hist_max) + MonthEnd(0)
            logger.info(f"Startup: historic_df.csv latest month = {_hist_max.strftime('%Y-%m-%d')}")
            if _hist_max < _pivot_check_dt:
                logger.warning(
                    f"Startup: DATA FRESHNESS MISMATCH -> PIVOT_DATE={PIVOT_DATE} (derived from "
                    f"latest_combined_data.parquet) is AHEAD of historic_df.csv's latest month "
                    f"({_hist_max.strftime('%Y-%m-%d')}). Every /forecast request will fail with "
                    f"'Model did not return a prediction for pivot month' until historic_df.csv "
                    f"(and likely final_chronos_forecasts.csv) are refreshed to cover {PIVOT_DATE}, "
                    f"or until these files are regenerated on the same cadence as latest_combined_data.parquet."
                )
        else:
            logger.warning("Startup: historic_df.csv has no valid 'month' values to check freshness against.")
    if not forecast_df.empty and "month" in forecast_df.columns:
        _fc_min = forecast_df["month"].dropna().min()
        _fc_max = forecast_df["month"].dropna().max()
        logger.info(f"Startup: final_chronos_forecasts.csv covers {_fc_min} to {_fc_max}.")
except Exception as e:
    logger.warning(f"Startup: could not run historic/forecast freshness check ({type(e).__name__}): {e}")


def get_lookup_area(area_name: str) -> str:
    """
    Return the area name to use for data/model lookups.
    If `area_name` is explicitly mapped in AREA_PROXY_MAP, use its proxy/group area.
    Otherwise, if it has its own historical data, use it directly.
    If neither exists, return the original name unchanged (will likely 404 downstream).
    """
    proxy = AREA_PROXY_MAP.get(area_name)
    if proxy:
        logger.info(f"Area '{area_name}' has no individual model; using proxy/group '{proxy}'")
        return proxy
    if area_name in VALID_AREAS_SET:
        logger.info(f"Area '{area_name}' has its own historical data; using it directly for lookups.")
        return area_name
    logger.warning(
        f"Area '{area_name}' not found in VALID_AREAS and has no entry in AREA_PROXY_MAP; "
        f"lookups will proceed with the original name and may fail downstream."
    )
    return area_name


def haversine_km(lat1, lon1, lat2, lon2):
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return EARTH_RADIUS_KM * c


def map_latlon_to_area(lat: float, lon: float) -> str:
    distances = []
    for area, (alat, alon) in AREA_COORDS.items():
        dist = haversine_km(lat, lon, alat, alon)
        distances.append((dist, area))
    distances.sort(key=lambda x: x[0])
    
    thresholds = [3.0, 5.0, 8.0]
    for threshold in thresholds:
        for dist, area in distances:
            if dist <= threshold:
                return area
                
    raise ValueError(f"Coordinates ({lat}, {lon}) are too far (>{thresholds[-1]}km) from any known area.")


def get_ref_row(area_name: str):
    if not latest_combined_df.empty and "area_name_en" in latest_combined_df.columns:
        area_df = latest_combined_df[latest_combined_df["area_name_en"] == area_name]
        if not area_df.empty:
            return area_df.iloc[-1]
    raise ValueError(f"Area {area_name} not found in historical data.")


def get_validated_payload(area_name: str, params: dict) -> dict:
    logger.info(f"Validating payload for area='{area_name}' with request params={params}")
    # ref_row is only used as a last-resort fallback if the area is missing
    # from input_ranges.csv entirely; never fatal here.
    try:
        ref_row = get_ref_row(area_name)
    except Exception as e:
        logger.warning(f"No historical ref_row found for area '{area_name}' ({type(e).__name__}: {e}); continuing without it.")
        ref_row = None

    area_ranges = get_area_input_ranges(area_name)
    if not area_ranges:
        logger.warning(f"Area '{area_name}' has no entry in input_ranges.csv; falling back to global/hardcoded defaults where needed.")

    def area_default(key):
        vals = area_ranges.get(key)
        return vals[0] if vals else None

    def value_allowed(key, value):
        vals = area_ranges.get(key)
        if not vals:
            return True  # no range info for this area/column -> can't validate, accept
        return str(value) in [str(v) for v in vals]

    def ref_row_value(key, castfunc):
        if ref_row is None or key not in ref_row:
            return None
        v = ref_row.get(key)
        if pd.isna(v):
            return None
        if isinstance(v, pd.Timestamp) or hasattr(v, "month"):
            if key == "month":
                return v.month
            elif key == "year":
                return v.year
        try:
            return castfunc(v)
        except Exception:
            return None

    def resolve(key, castfunc, hardcoded_fallback, allow_input=True):
        # Tier 1: explicit request param, only if allowed for this area/column
        if allow_input:
            raw = params.get(key)
            if raw is not None:
                try:
                    cast_val = castfunc(raw)
                    if value_allowed(key, cast_val):
                        return cast_val
                except Exception:
                    pass
        # Tier 2: this area's first value in input_ranges.csv
        dv = area_default(key)
        if dv is not None:
            try:
                return castfunc(dv)
            except Exception:
                pass
        # Tier 3: most recent historical row for this area
        rv = ref_row_value(key, castfunc)
        if rv is not None:
            return rv
        # Tier 4: global first value across all areas in input_ranges.csv
        gv = INPUT_RANGE_DEFAULTS_GLOBAL.get(key)
        if gv is not None:
            try:
                return castfunc(gv)
            except Exception:
                pass
        # Tier 5: hardcoded literal fallback
        return hardcoded_fallback

    def resolve_procedure_area():
        raw = params.get("procedure_area")
        base_val = None
        
        if raw is not None:
            try:
                base_val = float(raw)
            except Exception:
                pass
                
        if base_val is None:
            dv = area_default("procedure_area")
            if dv is not None:
                try:
                    base_val = float(dv)
                except Exception:
                    pass
                    
        if base_val is None:
            base_val = 55.0
            
        # Apply strict clamping based on room type
        r_en = resolve("rooms_en", str, "1 B/R")
        if r_en == "Studio":
            min_area, max_area = 25.0, 45.0
        elif r_en == "1 B/R":
            min_area, max_area = 45.0, 70.0
        elif r_en == "2 B/R":
            min_area, max_area = 65.0, 90.0
        elif r_en == "3 B/R":
            min_area, max_area = 80.0, 110.0
        else:
            min_area, max_area = 110.0, 145.0
            
        area_limits = get_area_input_ranges(area_name)
        a_min = area_limits.get("_procedure_area_min")
        a_max = area_limits.get("_procedure_area_max")
        
        if a_min is not None and a_min > min_area:
            min_area = a_min
        if a_max is not None and a_max < max_area:
            max_area = a_max
            
        if min_area > max_area:
            min_area = max_area

        if base_val < min_area:
            return min_area
        elif base_val > max_area:
            return max_area
        return base_val

    payload = {
        "area_name": area_name,
        "rooms_en": resolve("rooms_en", str, "1 B/R"),
        "reg_type_en": resolve("reg_type_en", str, "Off-Plan Properties"),
        "floor_bin": resolve("floor_bin", str, "1-10"),
        "Grade": resolve("Grade", str, "B"),
        "project_grade": resolve("project_grade", str, "B"),
        "Developer_grade": resolve("Developer_grade", str, "B"),
        # Default-only fields: never read from request params, always input_ranges.csv
        "Developer Reputation Tier": resolve("Developer Reputation Tier", str, "Tier 2 - Established Premium", allow_input=False),
        "trans_group_en": resolve("trans_group_en", str, "Sales", allow_input=False),
        "Locality Zone": resolve("Locality Zone", str, "Suburban 🌳",allow_input=False),
        "Price Tier": resolve("Price Tier", str, "Mid-Range ●", allow_input=False),
        "Developer Tier": resolve("Developer Tier", str, "Tier 2 – Established", allow_input=False),
        "Reputation": resolve("Reputation", str, "Good ✓", allow_input=False),
        "Score": resolve("Score", float, 7.0, allow_input=False),
        "has_parking": resolve("has_parking", int, 1),
        "swimming_pool": resolve("swimming_pool", int, 1),
        "balcony": resolve("balcony", int, 1),
        "elevators": resolve("elevators", int, 1),
        "metro": resolve("metro", int, 1),
        "procedure_area": resolve_procedure_area(),
        # year/month are no longer accepted as request input -- always derived
        # from the latest month in latest_combined_data.parquet (see startup logs),
        # falling back to the current date if that couldn't be determined.
        "year": DEFAULT_YEAR,
        "month": DEFAULT_MONTH,
    }
    logger.info(f"Resolved payload for area='{area_name}': year={DEFAULT_YEAR}, month={DEFAULT_MONTH}")
    return payload


FILTER_COLS = ["Developer_grade", "reg_type_en", "rooms_en", "trans_group_en"]
PRICE_COL_CANDIDATES = ["meter_sale_price"]


def get_price_column(df: pd.DataFrame) -> str:
    for col in PRICE_COL_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError("No price column found.")


def remove_outliers_5_95(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return df
    x = pd.to_numeric(df[col], errors="coerce")
    low = x.quantile(0.05)
    high = x.quantile(0.95)
    return df[(x >= low) & (x <= high)].copy()


def normalize_monthly_series(df: pd.DataFrame, pivot_date: str, price_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["month", "median_price"])
    start_dt = pd.Timestamp("2020-01-01")
    pivot_dt = pd.to_datetime(pivot_date) + MonthEnd(0)
    monthly = (
        df.groupby("month", as_index=False)[price_col]
        .median()
        .rename(columns={price_col: "median_price"})
        .sort_values("month")
    )
    full_months = pd.date_range(start=start_dt, end=pivot_dt, freq="ME")
    monthly = (
        monthly.set_index("month")
        .reindex(full_months)
        .rename_axis("month")
        .reset_index()
    )
    monthly["median_price"] = pd.to_numeric(monthly["median_price"], errors="coerce")
    monthly["median_price"] = monthly["median_price"].interpolate(limit_direction="both").ffill().bfill()
    return monthly


def get_monthly_historic_series(area_name: str, params: dict, pivot_date: str) -> pd.DataFrame:
    if latest_combined_df.empty:
        return pd.DataFrame(columns=["month", "median_price", "area"])
    required_cols = {"area_name_en", "month"}
    if not required_cols.issubset(set(latest_combined_df.columns)):
        return pd.DataFrame(columns=["month", "median_price", "area"])
    price_col = get_price_column(latest_combined_df)
    pivot_dt = pd.to_datetime(pivot_date) + MonthEnd(0)
    start_dt = pd.Timestamp("2020-01-01")
    base = latest_combined_df.copy()
    
    lower_area = area_name.lower()
    if lower_area in PROXY_REVERSE_MAP:
        combined_areas = PROXY_REVERSE_MAP[lower_area]
        base = base[
            (base["area_name_en"].isin(combined_areas)) &
            (base["month"] >= start_dt) &
            (base["month"] <= pivot_dt)
        ].copy()
    else:
        base = base[
            (base["area_name_en"] == area_name) &
            (base["month"] >= start_dt) &
            (base["month"] <= pivot_dt)
        ].copy()
    if base.empty:
        return pd.DataFrame(columns=["month", "median_price", "area"])
    for col in FILTER_COLS:
        if col in base.columns:
            base[col] = base[col].astype(str).str.strip()
    wanted = {col: str(params[col]).strip() for col in FILTER_COLS if params.get(col) is not None}
    exact_df = base.copy()
    for col, val in wanted.items():
        if col in exact_df.columns:
            exact_df = exact_df[exact_df[col] == val]
    if not exact_df.empty:
        exact_df = remove_outliers_5_95(exact_df, price_col)
        out = normalize_monthly_series(exact_df, pivot_date, price_col)
        out["area"] = area_name
        return out
    candidates = base.copy()
    available_cols = [c for c in FILTER_COLS if c in candidates.columns and c in wanted]
    if available_cols:
        candidates["mismatch_score"] = 0
        for col in available_cols:
            candidates["mismatch_score"] += (candidates[col] != wanted[col]).astype(int)
        min_score = candidates["mismatch_score"].min()
        nearest_df = candidates[candidates["mismatch_score"] == min_score].copy()
        if not nearest_df.empty:
            nearest_df = remove_outliers_5_95(nearest_df, price_col)
            out = normalize_monthly_series(nearest_df, pivot_date, price_col)
            out["area"] = area_name
            return out
    base = remove_outliers_5_95(base, price_col)
    out = normalize_monthly_series(base, pivot_date, price_col)
    out["area"] = area_name
    return out


def build_news_adjusted_forecast(area_name: str, pivot_price: float, pivot_date: str) -> pd.DataFrame:
    if news_df.empty:
        return pd.DataFrame(columns=["month", "median_price"])
    required_cols = {"area", "date", "adjusted_pred"}
    if not required_cols.issubset(set(news_df.columns)):
        return pd.DataFrame(columns=["month", "median_price"])
    
    pivot_dt = pd.to_datetime(pivot_date) + MonthEnd(0)
    area_news = news_df[
        (news_df["area"].astype(str).str.strip() == area_name) &
        (pd.to_datetime(news_df["date"], errors="coerce") >= pivot_dt)
    ].copy()
    if area_news.empty:
        return pd.DataFrame(columns=["month", "median_price"])
    
    area_news["month"] = pd.to_datetime(area_news["date"], errors="coerce")
    area_news = area_news.sort_values("month")
    
    rows = []
    prev_api_price = float(pivot_price)
    prev_news_price = None
    
    for _, row in area_news.iterrows():
        curr_month = row["month"]
        curr_news_price = float(row["adjusted_pred"]) if pd.notna(row["adjusted_pred"]) else 0.0
        
        if curr_month <= pivot_dt:
            prev_news_price = curr_news_price
            continue
            
        if prev_news_price and prev_news_price > 0:
            growth = (curr_news_price - prev_news_price) / prev_news_price
        else:
            growth = 0.0
            
        adjusted_price = prev_api_price * (1 + growth)
        rows.append({
            "month": curr_month.strftime("%Y-%m-%d"),
            "median_price": round(adjusted_price, 2)
        })
        
        prev_api_price = adjusted_price
        prev_news_price = curr_news_price
        
    return pd.DataFrame(rows)


def find_fallback_pivot(res_df: pd.DataFrame, primary_pivot_dt: pd.Timestamp):
    """
    Fallback pivot resolution -- ONLY called when no valid (non-null
    median_price) row exists in res_df at the primary, dynamically-computed
    pivot date (PIVOT_DATE, derived from latest_combined_data.parquet).

    Attempts, in order:
      1. The last available month in res_df at/before the primary pivot with
         a non-null median_price ("last month of historic df").
      2. If historic data doesn't reach that far (or res_df is empty),
         the current calendar month, using whatever row (historic or
         forecast) already exists there.
      3. If that's also missing, forward-fill the whole median_price series
         and take the value as-of the current month.

    Returns (effective_pivot_dt, pivot_price, reason_str). If nothing at all
    could be resolved, effective_pivot_dt and pivot_price are both None and
    reason_str explains why.
    """
    current_month_end = pd.Timestamp.now() + MonthEnd(0)

    if res_df.empty:
        return None, None, "res_df is completely empty (no historic or forecast rows for this area)."

    df = res_df.dropna(subset=["month"]).sort_values("month").copy()

    # 1) last historic-range month at/before the primary pivot, with a real price
    hist_candidates = df[(df["month"] <= primary_pivot_dt) & (df["median_price"].notna())]
    if not hist_candidates.empty:
        row = hist_candidates.iloc[-1]
        return row["month"], float(row["median_price"]), "last available historic month at/before pivot"

    # 2) current month, using whatever row already exists there (usually forecast data)
    cur_candidates = df[(df["month"] == current_month_end) & (df["median_price"].notna())]
    if not cur_candidates.empty:
        row = cur_candidates.iloc[-1]
        return row["month"], float(row["median_price"]), "current month (historic pivot not found)"

    # 3) forward-fill the whole series and take the value as-of the current month
    series = df.set_index("month")["median_price"].sort_index()
    full_index = sorted(set(series.index.tolist() + [current_month_end]))
    filled = series.reindex(full_index).ffill()
    if current_month_end in filled.index and pd.notna(filled.loc[current_month_end]):
        return current_month_end, float(filled.loc[current_month_end]), "current month via forward-filled series"

    return None, None, "no non-null median_price found anywhere in res_df, even after forward-fill."


@app.get("/areas")
async def list_areas():
    return {
        "areas": VALID_AREAS,
        "count": len(VALID_AREAS),
        "coords_available": list(AREA_COORDS.keys())
    }

@app.get("/areas/{area_name}")
async def area_info(area_name: str):
    if area_name not in VALID_AREAS_SET:
        raise HTTPException(404, "Area not found")
    
    try:
        row = get_ref_row(area_name)
    except ValueError:
        raise HTTPException(404, "Area data not available")
        
    coords = AREA_COORDS.get(area_name, {"lat": None, "lon": None})
    
    return {
        "area_name": area_name,
        "default_combinations": {
            "reg_type_en": row.get("reg_type_en"),
            "land_type_en": row.get("land_type_en"),
            "elevator": row.get("elevator") if "elevator" in row else row.get("elevators"),
            "procedure_area": row.get("procedure_area"),
        },
        "coordinates": coords
    }


@app.get("/forecast")
async def unified_forecast(
    area_name: Optional[str] = Query(None, description="Area Name"),
    lat: Optional[float] = Query(None, description="Latitude"),
    lon: Optional[float] = Query(None, description="Longitude"),
    rooms_en: Optional[str] = Query(None),
    reg_type_en: Optional[str] = Query(None),
    floor_bin: Optional[str] = Query(None),
    Grade: Optional[str] = Query(None),
    project_grade: Optional[str] = Query(None),
    Developer_grade: Optional[str] = Query(None),
    has_parking: Optional[int] = Query(None, ge=0, le=1),
    swimming_pool: Optional[int] = Query(None, ge=0, le=1),
    balcony: Optional[int] = Query(None, ge=0, le=1),
    elevators: Optional[int] = Query(None, ge=0, le=1),
    metro: Optional[int] = Query(None, ge=0, le=1),
    procedure_area: Optional[float] = Query(None),
):
    try:
        resolved_area = area_name or (map_latlon_to_area(lat, lon) if lat is not None and lon is not None else None)
        if not resolved_area:
            raise ValueError("Must provide either area_name or lat/lon coordinates.")
        logger.info(f"/forecast request: resolved_area='{resolved_area}' (area_name={area_name}, lat={lat}, lon={lon})")

        # If this area has no individual model/data, use its proxy/group instead.
        lookup_area = get_lookup_area(resolved_area)
        if lookup_area != resolved_area:
            logger.info(f"/forecast request: Area '{resolved_area}' is mapped to '{lookup_area}'. Using '{lookup_area}' for all data fetching and output.")
            resolved_area = lookup_area

        if resolved_area != area_name and area_name is not None:
            logger.info(f"Area name fuzzy matched: '{area_name}' -> '{resolved_area}'")

        params = {
            "rooms_en": rooms_en,
            "reg_type_en": reg_type_en,
            "floor_bin": floor_bin,
            "Grade": Grade,
            "project_grade": project_grade,
            "Developer_grade": Developer_grade,
            "has_parking": has_parking,
            "swimming_pool": swimming_pool,
            "balcony": balcony,
            "elevators": elevators,
            "metro": metro,
            "procedure_area": procedure_area
        }
        final_input = get_validated_payload(resolved_area, params)
        final_input["model_source_area"] = resolved_area
        logger.info(f"/forecast request: final_input resolved for model call: {final_input}")
        history_params = {
            "project_grade": final_input.get("project_grade"),
            "reg_type_en": final_input.get("reg_type_en"),
            "rooms_en": final_input.get("rooms_en"),
            "trans_group_en": final_input.get("trans_group_en"),
        }

        historic_series = get_monthly_historic_series(resolved_area, history_params, PIVOT_DATE)
        logger.info(f"/forecast request: calling predict_property_price for '{resolved_area}'")
        res_df = predict_property_price(final_input, forecast_df.copy(), historic_series.copy())
        pivot_dt = pd.to_datetime(PIVOT_DATE) + MonthEnd(0)

        model_point_df = pd.DataFrame()
        if not res_df.empty:
            res_df = res_df.copy()
            res_df["month"] = pd.to_datetime(res_df["month"], errors="coerce") + MonthEnd(0)
            res_df = res_df.sort_values("month")
            exact_match = res_df[(res_df["month"] == pivot_dt) & (res_df["median_price"].notna())]
            if not exact_match.empty:
                model_point_df = exact_match.copy()

        effective_pivot_dt = pivot_dt
        if model_point_df.empty:
            # Primary pivot lookup failed -- fall back: last historic month
            logger.warning(
                f"/forecast request: no valid prediction at primary pivot {PIVOT_DATE} for "
                f"'{lookup_area}'; attempting fallback pivot resolution."
            )
            if not historic_series.empty:
                last_hist = historic_series.iloc[-1]
                fb_month = pd.to_datetime(last_hist["month"])
                fb_price = float(last_hist["median_price"])
                fb_reason = "last month of historic_series (latest_combined_data)"
            else:
                fb_month, fb_price, fb_reason = find_fallback_pivot(res_df, pivot_dt)
                
            if fb_month is None:
                raise ValueError(
                    f"Model did not return a prediction for pivot month {PIVOT_DATE} "
                    f"(area='{lookup_area}'), and fallback pivot resolution also failed: {fb_reason}"
                )
            effective_pivot_dt = fb_month
            model_point_df = pd.DataFrame([{"month": fb_month, "median_price": fb_price, "area": resolved_area}])
            logger.info(
                f"/forecast request: using FALLBACK pivot for '{lookup_area}' -> "
                f"month={fb_month.strftime('%Y-%m-%d')}, price={fb_price} (reason: {fb_reason})"
            )

        pivot_price = float(model_point_df.iloc[0]["median_price"])
        effective_pivot_date_str = effective_pivot_dt.strftime("%Y-%m-%d")
        logger.info(f"/forecast request: effective pivot for '{resolved_area}' = {effective_pivot_date_str}, price = {pivot_price}")
        if not res_df.empty:
            res_df = res_df.copy()
            res_df["month"] = pd.to_datetime(res_df["month"], errors="coerce") + MonthEnd(0)
            res_df = res_df.sort_values("month").reset_index(drop=True)
            
        area_news = pd.DataFrame()
        news_available = False
        if not news_df.empty and "area" in news_df.columns:
            area_news = news_df[news_df["area"].astype(str).str.strip() == resolved_area].copy()
            if area_news.empty and lookup_area != resolved_area:
                area_news = news_df[news_df["area"].astype(str).str.strip() == lookup_area].copy()
            
            if not area_news.empty:
                news_available = True
                if "date" in area_news.columns:
                    area_news["month"] = pd.to_datetime(area_news["date"], errors="coerce") + MonthEnd(0)
                else:
                    area_news["month"] = pd.to_datetime(area_news["month"], errors="coerce") + MonthEnd(0)
                
                area_news = area_news.drop_duplicates(subset=["month"], keep="last")

        if news_available:
            merge_cols = ["month"]
            if "adjusted_pred" in area_news.columns: merge_cols.append("adjusted_pred")
            if "predicted_mom_growth_pct" in area_news.columns: merge_cols.append("predicted_mom_growth_pct")
            if "narrative" in area_news.columns: merge_cols.append("narrative")
            elif "news_narrative" in area_news.columns:
                area_news["narrative"] = area_news["news_narrative"]
                merge_cols.append("narrative")

            res_df = res_df.merge(
                area_news[merge_cols],
                on='month', how='left'
            )

            cutoff = effective_pivot_dt
            idx_jan = res_df.index[res_df["month"] == cutoff].tolist()
            if idx_jan:
                idx_jan = idx_jan[0]
                res_df.at[idx_jan, "macro_forecast"] = res_df.loc[idx_jan, "median_price"]
                res_df.at[idx_jan, "news_adjusted_forecast"] = res_df.loc[idx_jan, "median_price"]

                for i in range(idx_jan + 1, len(res_df)):
                    prev_baseline = res_df.loc[i-1, "median_price"]
                    raw_prev = res_df.loc[i-1, "median_price"]
                    raw_curr = res_df.loc[i, "median_price"]
                    m_rate = (raw_curr / raw_prev) - 1 if (pd.notna(raw_prev) and pd.notna(raw_curr) and raw_prev != 0) else 0.0

                    a_rate = 0.0
                    if "adjusted_pred" in res_df.columns:
                        news_prev = res_df.loc[i-1, "adjusted_pred"]
                        news_curr = res_df.loc[i, "adjusted_pred"]
                        if pd.notna(news_prev) and pd.notna(news_curr) and news_prev != 0:
                            a_rate = (news_curr / news_prev) - 1

                    current_macro = prev_baseline * (1 + m_rate)
                    res_df.at[i, "macro_forecast"] = current_macro
                    res_df.at[i, "news_adjusted_forecast"] = current_macro * (1 + a_rate)
                    res_df.at[i, "median_price"] = current_macro
        else:
            res_df["news_adjusted_forecast"] = None
            res_df["narrative"] = None

        def fmt_series(df, col="median_price"):
            if df.empty: return []
            df = df.sort_values("month")
            return [
                {
                    "timestamp": pd.to_datetime(r["month"]).strftime("%Y-%m-%d"),
                    "value": round(float(r[col]), 2) if pd.notna(r[col]) else None
                }
                for _, r in df.iterrows()
            ]

        before = historic_series[pd.to_datetime(historic_series["month"]) < pd.to_datetime(effective_pivot_dt)].copy()
        if not before.empty:
            before["month"] = pd.to_datetime(before["month"], errors="coerce")
        at_cutoff = res_df[res_df["month"] == effective_pivot_dt]
        after = res_df[res_df["month"] > effective_pivot_dt]

        narrative_val = None
        if news_available and not at_cutoff.empty and "narrative" in at_cutoff.columns:
            raw_narr = at_cutoff.iloc[0].get("narrative")
            if pd.notna(raw_narr):
                narrative_val = str(raw_narr)

        news_forecast_series = []
        if news_available and "news_adjusted_forecast" in res_df.columns:
            news_forecast_series = fmt_series(res_df[res_df["month"] > effective_pivot_dt], col="news_adjusted_forecast")

        logger.info(
            f"/forecast request: completed successfully for area='{resolved_area}' "
            f"(model_source_area='{lookup_area}', news_available={news_available})"
        )
        return {
            "news_available": news_available,
            "before_prediction": fmt_series(before),
            "prediction_point": fmt_series(at_cutoff),
            "forecast": fmt_series(after),
            "news_adjusted_forecast": news_forecast_series,
            "narrative": narrative_val,
        }
    except ValueError as ve:
        logger.warning(f"/forecast request failed with a validation error ({type(ve).__name__}): {ve}")
        raise HTTPException(status_code=400, detail=f"{type(ve).__name__}: {ve}")
    except Exception as e:
        logger.error(f"/forecast request failed with an internal error ({type(e).__name__}): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal Prediction Error ({type(e).__name__}): {e}")