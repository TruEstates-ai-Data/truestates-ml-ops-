import os
import pandas as pd
import ast
import logging
import math
import random
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException, Query
from math import radians, cos, sin, sqrt, atan2


from fastapi.middleware.cors import CORSMiddleware






# ============================================================
# 1. SETUP & CONFIGURATION
# ============================================================
try:
    from price_predictor_v_2_3 import predict_property_price 
except ImportError:
    logging.error("Could not find price_predictor_v_2_3.py. Ensure the file is in the workspace.")

BASE_PATH = Path(__file__).parent 


# Files
COMBINATIONS_CSV = BASE_PATH / "V_2.3_combinations.csv"
FORECAST_PATH = BASE_PATH / "forecast_df_v_2_3.csv"
HISTORIC_PATH = BASE_PATH / "historic_df_v_2_3.csv"

PIVOT_DATE = "2026-07-31" 
EARTH_RADIUS_KM = 6371.0

app = FastAPI(
    title="TruEstates API v2.3",
    description="Dubai Real Estate API - Combinations-based Defaults & Volatility Clamping",
    version="2.3"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["https://truestates.com", "http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)

# --- Coordinate Reference Data ---
# "Jumeriah First": (25.2250, 55.2550),"Um Suqaim Third": (25.1450, 55.1950),
AREA_COORDS = {
    "Al Barsha First": (25.1150, 55.2050), "Al Barshaa South Second": (25.0921, 55.2410),
    "Al Barshaa South Third": (25.0760, 55.2150), "Al Barsha South Fourth": (25.0610, 55.2360),
    "Al Barsha South Fifth": (25.0485, 55.2341), "Al Hebiah First": (25.0334, 55.2205),
    "Al Hebiah Second": (25.0420, 55.2450), "Al Hebiah Third": (25.0160, 55.2380),
    "Al Hebiah Fourth": (25.0210, 55.2150), "Al Hebiah Sixth": (24.9950, 55.2320),
    "Al Khairan First": (25.1850, 55.3550), "Al Kifaf": (25.2350, 55.2950),
    "Al Merkadh": (25.1480, 55.3050), "Al Thanyah Third": (25.0850, 55.1550),
    "Al Thanyah Fifth": (25.0685, 55.1450), "Al Warsan First": (25.1650, 55.4150),
    "Al Yelayiss 2": (24.9650, 55.2650), "Bukadra": (25.1850, 55.3350),
    "Burj Khalifa": (25.1972, 55.2744), "Business Bay": (25.1850, 55.2750),
    "Hadaeq Sheikh Mohammed Bin Rashid": (25.1150, 55.2950), "Jabal Ali": (25.0000, 55.0500),
    "Jabal Ali First": (25.0220, 55.1050), "Jabal Ali Industrial Second": (24.9850, 55.1250),
     "Madinat Al Mataar": (24.8950, 55.1550),
    "Madinat Dubai Almelaheyah": (25.2650, 55.2750), "Madinat Hind 4": (25.0250, 55.4550),
    "Marsa Dubai": (25.0780, 55.1350), "Me'Aisem First": (25.0350, 55.1950),
    "Nadd Hessa": (25.1250, 55.3850), "Palm Deira": (25.3150, 55.3050),
    "Palm Jumeirah": (25.1124, 55.1390), "Ras Al Khor Industrial First": (25.1950, 55.3650),
     "Wadi Al Safa 2": (25.1200, 55.3700),
    "Wadi Al Safa 3": (25.0850, 55.3250), "Wadi Al Safa 4": (25.1450, 55.3050),
    "Wadi Al Safa 5": (25.0950, 55.3650), "Warsan Fourth": (25.1550, 55.4250),
    "Zaabeel First": (25.2200, 55.2850), "Zaabeel Second": (25.2050, 55.2950)
}

# # --- Load Datasets ---
# try:
#     combinations_df = pd.read_csv(COMBINATIONS_CSV)
#     forecast_df = pd.read_csv(FORECAST_PATH)
#     historic_df = pd.read_csv(HISTORIC_PATH)
    
#     forecast_df["month"] = pd.to_datetime(forecast_df["month"], dayfirst=True, errors='coerce')
#     historic_df["month"] = pd.to_datetime(historic_df["month"], dayfirst=True, errors='coerce')
#     combinations_df["area_name_en"] = combinations_df["area_name_en"].astype(str).str.strip()
    
#     VALID_AREAS = sorted(combinations_df["area_name_en"].unique())
#     logging.info(f"Startup: {len(VALID_AREAS)} areas loaded from combinations file.")
# except Exception as e:
#     logging.error(f"Startup Error: {e}")
#     VALID_AREAS = []


# --- Load Datasets ---
try:
    combinations_df = pd.read_csv(COMBINATIONS_CSV)
    print("combinations df:", combinations_df.head())
    forecast_df = pd.read_csv(FORECAST_PATH)
    print("forecast df:", forecast_df.head())
    historic_df = pd.read_csv(HISTORIC_PATH)
    print("historic df:", historic_df.head())
    news_df = pd.read_csv(BASE_PATH / "news_preds_v3_fixed.csv")  # ← ADD

    forecast_df["month"] = pd.to_datetime(forecast_df["month"], dayfirst=True, errors='coerce')
    historic_df["month"] = pd.to_datetime(historic_df["month"], dayfirst=True, errors='coerce')
    news_df["month"] = pd.to_datetime(news_df["month"], dayfirst=True, errors='coerce')  # ← ADD
    news_df["area_name_en"] = news_df["area_name_en"].astype(str).str.strip()              # ← ADD

    combinations_df["area_name_en"] = combinations_df["area_name_en"].astype(str).str.strip()
    VALID_AREAS = sorted(combinations_df["area_name_en"].unique())
    logging.info(f"Startup: {len(VALID_AREAS)} areas loaded.")
except Exception as e:
    logging.error(f"Startup Error: {e}")
    VALID_AREAS = []
    news_df = pd.DataFrame()



# ============================================================
# 2. LOGIC HELPERS
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    dlat, dlon = radians(lat2 - lat1), radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return EARTH_RADIUS_KM * c

def map_latlon_to_area(lat: float, lon: float) -> str:
    distances = []
    for area, (alat, alon) in AREA_COORDS.items():
        dist = haversine_km(lat, lon, alat, alon)
        distances.append((dist, area))
    distances.sort(key=lambda x: x[0])
    return distances[0][1]

def apply_historic_clamp(res_df: pd.DataFrame, pivot_date: str):
    """
    Compares the last month of history against the previous month.
    Clamps change to 15-18% if jump exceeds 18%.
    """
    pivot_dt = pd.to_datetime(pivot_date)
    hist_part = res_df[res_df["month"] <= pivot_dt].sort_values("month").reset_index(drop=True)

    if len(hist_part) < 2:
        return res_df

    idx_last, idx_prev = hist_part.index[-1], hist_part.index[-2]
    price_last, price_prev = hist_part.loc[idx_last, "median_price"], hist_part.loc[idx_prev, "median_price"]

    if price_prev == 0: return res_df
    perc_change = (price_last - price_prev) / price_prev

    if abs(perc_change) > 0.05:
        clamp_val = random.uniform(0.035, 0.05)
        direction = 1 if perc_change > 0 else -1
        new_price = price_prev * (1 + (direction * clamp_val))
        scaling_factor = new_price / price_last
        target_date = hist_part.loc[idx_last, "month"]
        res_df.loc[res_df["month"] >= target_date, "median_price"] *= scaling_factor
        logging.info(f"Clamped historic spike: {perc_change:.2%} -> {direction * clamp_val:.2%}")

    return res_df

def get_validated_payload(area_name: str, params: dict) -> dict:
    print("Area name is: ", area_name)
    print("VALID AREAS LIST IS: ", VALID_AREAS)
    if area_name not in VALID_AREAS:
        raise ValueError(f"Area '{area_name}' not found in combinations dataset.")

    # Get the reference default row for this area from the combinations file
    ref_row = combinations_df[combinations_df["area_name_en"] == area_name].iloc[0]

    def resolve(key, ref_key, cast_func, force_default=False):
        # If force_default is True, we ignore user input (for removed columns)
        val = None if force_default else params.get(key)
        
        if val is not None: 
            try: return cast_func(val)
            except: pass
        
        raw_ref = ref_row[ref_key]
        # Handle cases where CSV contains string representations of lists
        if isinstance(raw_ref, str) and "[" in raw_ref:
            try:
                actual_list = ast.literal_eval(raw_ref)
                return actual_list[0] if actual_list else ""
            except:
                return raw_ref
        return cast_func(raw_ref)

    # Note: procedure_area, land_type, reg_type, and elevator are now STRICTLY from the file.
    payload = {
        "area_name":      area_name,
        "reg_type_en":    resolve(None, "reg_type_en", str, force_default=True),
        "rooms_en":       resolve("rooms_en", "rooms_en", str),
        "land_type_en":   resolve(None, "land_type_en", str, force_default=True),
        "floor_bin":      resolve("floor_bin", "floor_bin", str),
        "developer_cat":  resolve("developer_type", "developer_cat", str),
        "project_cat":    resolve("project_type", "project_cat", str),
        "has_parking":    resolve("has_parking", "has_parking", int),
        "swimming_pool":  resolve("swimming_pool", "swimming_pool", int),
        "balcony":        resolve("balcony", "balcony", int),
        "elevator":       resolve(None, "elevator", int, force_default=True),
        "metro":          resolve("metro", "metro", int),
        "procedure_area": resolve(None, "procedure_area", float, force_default=True)
    }
    
    return payload

# ============================================================
# 3. ENDPOINT
# ============================================================
@app.get("/areas")
async def list_areas():
    return {
        "areas": VALID_AREAS,
        "count": len(VALID_AREAS),
        "coords_available": list(AREA_COORDS.keys())
    }

@app.get("/areas/{area_name}")
async def area_info(area_name: str):
    if area_name not in VALID_AREAS:
        raise HTTPException(404, "Area not found")
    
    row = combinations_df[combinations_df["area_name_en"] == area_name].iloc[0]
    coords = AREA_COORDS.get(area_name, {"lat": None, "lon": None})
    
    return {
        "area_name": area_name,
        "default_combinations": {
            "reg_type_en": row["reg_type_en"],
            "land_type_en": row["land_type_en"],
            "elevator": row["elevator"],
            "procedure_area": row["procedure_area"],
        },
        "coordinates": coords
    }


@app.get("/unified_forecast")
async def unified_forecast(
    area_name: Optional[str] = Query(None, description="Area Name"),
    lat: Optional[float] = Query(None, description="Latitude"),
    lon: Optional[float] = Query(None, description="Longitude"),
    # Removed: reg_type, procedure_area, land_type, elevator (now automated)
    rooms_en: Optional[str] = Query(None),
    floor_bin: Optional[str] = Query(None),
    developer_type: Optional[str] = Query(None),
    project_type: Optional[str] = Query(None),
    has_parking: Optional[int] = Query(None, ge=0, le=1),
    swimming_pool: Optional[int] = Query(None, ge=0, le=1),
    balcony: Optional[int] = Query(None, ge=0, le=1),
    metro: Optional[int] = Query(None, ge=0, le=1)
):
    try:
        # 1. Resolve Area (Name or Coordinates)
        resolved_area = area_name
        if not resolved_area:
            if lat is not None and lon is not None:
                resolved_area = map_latlon_to_area(lat, lon)
            else:
                raise ValueError("Must provide either area_name or lat/lon coordinates.")

        # 2. Bundle user inputs (Automated fields are handled inside get_validated_payload)
        user_params = {
            "rooms_en": rooms_en, "floor_bin": floor_bin, "developer_type": developer_type,
            "project_type": project_type, "has_parking": has_parking,
            "swimming_pool": swimming_pool, "balcony": balcony, "metro": metro
        }
        
        # 3. Resolve defaults from combinations file
        final_input = get_validated_payload(resolved_area, user_params)
        print("final input: ",final_input)
        # 4. Run Prediction
        res_df = predict_property_price(final_input, forecast_df, historic_df)
        print("res df before: ",res_df.head())
        # 5. Apply Volatility Clamp (18% logic)
        res_df = apply_historic_clamp(res_df, PIVOT_DATE)
        print("res df after: ",res_df.head())
                # 4. Run Prediction
        res_df = predict_property_price(final_input, forecast_df, historic_df)

        # 5. Apply Volatility Clamp
        res_df = apply_historic_clamp(res_df, PIVOT_DATE)

        # 6. Merge news adjustments (same as Streamlit)
        area_news = news_df[news_df["area_name_en"] == resolved_area].copy()
        news_available = not area_news.empty

        if news_available:
            res_df = res_df.merge(
                area_news[['month', 'predictions_mom_growth', 'adjusted_price_real_mom_growth', 'narrative']],
                on='month', how='left'
            )

            cutoff = pd.to_datetime(PIVOT_DATE)
            idx_jan = res_df.index[res_df["month"] == cutoff].tolist()
            if idx_jan:
                idx_jan = idx_jan[0]
                res_df.at[idx_jan, "macro_forecast"] = res_df.loc[idx_jan, "median_price"]
                res_df.at[idx_jan, "news_adjusted_forecast"] = res_df.loc[idx_jan, "median_price"]

                for i in range(idx_jan + 1, len(res_df)):
                    prev_baseline = res_df.loc[i-1, "median_price"]
                    m_rate = res_df.loc[i, "predictions_mom_growth"]
                    if pd.isna(m_rate):
                        raw_prev = res_df.loc[i-1, "median_price"]
                        raw_curr = res_df.loc[i, "median_price"]
                        m_rate = (raw_curr / raw_prev) - 1 if raw_prev != 0 else 0

                    a_rate = res_df.loc[i, "adjusted_price_real_mom_growth"] \
                             if pd.notna(res_df.loc[i, "adjusted_price_real_mom_growth"]) else 0

                    current_macro = prev_baseline * (1 + m_rate)
                    res_df.at[i, "macro_forecast"] = current_macro
                    res_df.at[i, "news_adjusted_forecast"] = current_macro * (1 + a_rate)
                    res_df.at[i, "median_price"] = current_macro
        else:
            res_df["news_adjusted_forecast"] = None
            res_df["narrative"] = None

        # 7. Format Response
        cutoff = pd.to_datetime(PIVOT_DATE)

        def fmt_series(df, col="median_price"):
            if df.empty: return []
            df = df.sort_values("month")
            return [
                {
                    "timestamp": r["month"].strftime("%Y-%m-%d"),
                    "value": round(float(r[col]), 2) if pd.notna(r[col]) else None
                }
                for _, r in df.iterrows()
            ]

        before = res_df[res_df["month"] < cutoff]
        at_cutoff = res_df[res_df["month"] == cutoff]
        after = res_df[res_df["month"] > cutoff]

        # Narrative: take from prediction_point row (Jan 2026)
        narrative_val = None
        if news_available and not at_cutoff.empty:
            raw_narr = at_cutoff.iloc[0].get("narrative")
            narrative_val = str(raw_narr) if pd.notna(raw_narr) else None

        # news_adjusted_forecast: full series (before + cutoff + after)
        news_forecast_series = []
        if news_available and "news_adjusted_forecast" in res_df.columns:
            news_forecast_series = fmt_series(res_df[res_df["month"]>=cutoff], col="news_adjusted_forecast")

        return {
            "news_available": news_available,
            "before_prediction": fmt_series(before),
            "prediction_point": fmt_series(at_cutoff),
            "forecast": fmt_series(after),
            "news_adjusted_forecast": news_forecast_series,
            "narrative": narrative_val,
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"Internal API Error: {e}")
        raise HTTPException(status_code=500, detail="Internal Prediction Error")

@app.get("/")
def health_check():
    return {
        "status": "online", 
        "areas_supported": len(VALID_AREAS),
        "pivot_date": PIVOT_DATE
    }

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("Api_v_2_3:app", host="0.0.0.0", port=8504, reload=True)
