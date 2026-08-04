import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VERSION = "v_2_3"
BASE_PATH = Path(__file__).parent
MODEL_DIR = BASE_PATH / "models"
COL_DIR = BASE_PATH / "trained_columns"
ARCHIVE_DIR = BASE_PATH / "old_files"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
COL_DIR.mkdir(parents=True, exist_ok=True)
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)

DIRECT_MODEL_AREAS = {
    "Al Barsha South Fourth", "Business Bay", "Al Merkadh", "Burj Khalifa",
    "Hadaeq Sheikh Mohammed Bin Rashid", "Al Khairan First", "Wadi Al Safa 5",
    "Al Thanyah Fifth", "Al Barshaa South Third", "Jabal Ali First",
    "Madinat Al Mataar", "Madinat Dubai Almelaheyah", "Me'Aisem First",
    "Al Hebiah Fourth", "Al Barsha South Fifth", "Al Hebiah First",
    "Nadd Hessa", "Palm Jumeirah", "Al Barshaa South Second", "Al Barsha South Second",
    "Al Barsha South Third", "Al Yelayiss 2", "Al Warsan First", "Marsa Dubai", "Bukadra"
}

PROXY_MAPPING = {
    "Al Kifaf": "G1", "Wadi Al Safa 4": "G1",
    "Warsan Fourth": "G3", "Jabal Ali": "G3",
    "Zaabeel Second": "G4", "Zaabeel First": "G4",
    "Al Barsha First": "Proxy1", "Al Hebiah Second": "Proxy1",
    "Al Hebiah Sixth": "Proxy1", "Al Hebiah Third": "Proxy1",
    "Madinat Hind 4": "Proxy1", "Wadi Al Safa 3": "Proxy1",
    "Wadi Al Safa 7": "Proxy1",
    "Ras Al Khor Industrial First": "Proxy2",
    "Jumeirah First": "Proxy2", "Palm Deira": "Proxy2",
    "Al Thanyah Third": "Proxy3", "Jabal Ali Industrial Second": "Proxy3"
}

CATEGORICAL_FEATURES = [
    "trans_group_en",
    "rooms_en",
    "reg_type_en",
    "floor_bin",
    "Grade",
    "project_grade",
    "Developer_grade",
    "Developer Reputation Tier",
    "Locality Zone",
    "Price Tier",
    "Developer Tier",
    "Reputation",
]

NUMERIC_FEATURES = [
    "has_parking",
    "swimming_pool",
    "balcony",
    "elevators",
    "metro",
    "procedure_area",
    "Score",
    "year",
    "month",
]

def get_slug(name):
    return str(name).replace(" ", "_").replace("'", "").lower()

def archive_existing_file(file_path: Path, prefix="old_"):
    if file_path.exists():
        archived_path = ARCHIVE_DIR / f"{prefix}{file_path.name}"
        if archived_path.exists():
            archived_path.unlink()
        shutil.move(str(file_path), str(archived_path))
        logger.info(f"Archived {file_path.name} -> {archived_path.name}")

def safe_load(path: Path):
    try:
        return joblib.load(path)
    except Exception as e:
        logger.warning(f"Failed to load {path.name}: {e}")
        return None

def load_assets(model_key):
    slug = get_slug(model_key)
    model_path = MODEL_DIR / f"model_{VERSION}_{slug}.joblib"
    col_path = COL_DIR / f"trained_columns_{VERSION}_{slug}.joblib"

    model = safe_load(model_path)
    cols = safe_load(col_path)

    if model is None or cols is None:
        old_model_path = ARCHIVE_DIR / f"old_{model_path.name}"
        old_col_path = ARCHIVE_DIR / f"old_{col_path.name}"
        if model is None and old_model_path.exists():
            model = safe_load(old_model_path)
        if cols is None and old_col_path.exists():
            cols = safe_load(old_col_path)

    if model is None:
        raise FileNotFoundError(f"Missing model for {model_key}")
    if cols is None:
        raise FileNotFoundError(f"Missing trained columns for {model_key}")

    return model, cols

def save_assets(model_key, model_obj, train_columns):
    slug = get_slug(model_key)
    model_path = MODEL_DIR / f"model_{VERSION}_{slug}.joblib"
    col_path = COL_DIR / f"trained_columns_{VERSION}_{slug}.joblib"

    archive_existing_file(model_path)
    archive_existing_file(col_path)

    joblib.dump(model_obj, model_path)
    joblib.dump(list(train_columns), col_path)
    logger.info(f"Saved model + columns for {model_key}")

def _build_input_frame(input_data, train_columns):
    raw = pd.DataFrame([input_data])
    raw = raw.reindex(columns=CATEGORICAL_FEATURES + NUMERIC_FEATURES, fill_value=np.nan)

    for c in CATEGORICAL_FEATURES:
        if c in raw.columns:
            raw[c] = raw[c].fillna("Unknown").astype(str)

    for c in NUMERIC_FEATURES:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce").fillna(0)

    encoded = pd.get_dummies(raw, columns=[c for c in CATEGORICAL_FEATURES if c in raw.columns], drop_first=False).astype(float)
    encoded = encoded.reindex(columns=train_columns, fill_value=0)
    return raw, encoded

def predict_property_price(input_data, forecast_df, historic_df):
    area = input_data["area_name"]

    if area in DIRECT_MODEL_AREAS:
        model_key = area
    elif area in PROXY_MAPPING:
        model_key = PROXY_MAPPING[area]
    else:
        raise ValueError(f"No model/proxy mapping found for: {area}")

    model, train_columns = load_assets(model_key)

    _, final_input_enc = _build_input_frame(input_data, train_columns)
    raw_row = pd.DataFrame([input_data])

    try:
        n_features = getattr(model, "n_features_in_", None)
        if n_features == len(train_columns):
            prediction_input = final_input_enc
        else:
            prediction_input = raw_row.reindex(columns=CATEGORICAL_FEATURES + NUMERIC_FEATURES, fill_value=np.nan)
            for c in CATEGORICAL_FEATURES:
                if c in prediction_input.columns:
                    prediction_input[c] = prediction_input[c].fillna("Unknown").astype(str)
            for c in NUMERIC_FEATURES:
                if c in prediction_input.columns:
                    prediction_input[c] = pd.to_numeric(prediction_input[c], errors="coerce").fillna(0)
    except Exception:
        prediction_input = raw_row.reindex(columns=CATEGORICAL_FEATURES + NUMERIC_FEATURES, fill_value=np.nan)

    base_prediction = float(model.predict(prediction_input)[0])

    forecast_df["month"] = pd.to_datetime(forecast_df["month"], errors="coerce")
    historic_df["month"] = pd.to_datetime(historic_df["month"], errors="coerce")

    gf_slice = forecast_df[forecast_df["area"] == model_key].copy() if "area" in forecast_df.columns else pd.DataFrame()
    hist_slice = historic_df[historic_df["area"] == model_key].copy() if "area" in historic_df.columns else pd.DataFrame()

    if gf_slice.empty and hist_slice.empty:
        return pd.DataFrame({
            "month": [pd.Timestamp.now()],
            "median_price": [base_prediction],
            "area": [area]
        })

    if not gf_slice.empty:
        gf_slice = gf_slice.sort_values("month").copy()
        if "growth_factor" in gf_slice.columns:
            gf_slice["median_price"] = base_prediction * gf_slice["growth_factor"]
        elif "predictions" in gf_slice.columns:
            gf_slice["median_price"] = gf_slice["predictions"]
        else:
            gf_slice["median_price"] = base_prediction

    # if not hist_slice.empty:
    #     hist_slice = hist_slice.sort_values("month").copy()
    #     hist_slice.loc[hist_slice.index[-1], "median_price"] = base_prediction

    combined = pd.concat([hist_slice, gf_slice], ignore_index=True)
    combined["area"] = area

    if "median_price" not in combined.columns:
        combined["median_price"] = base_prediction

    return combined[["month", "median_price", "area"]].sort_values("month")

def train_model_and_save(model_key, model_obj, train_columns):
    save_assets(model_key, model_obj, train_columns)

if __name__ == "__main__":
    logger.info("price_predictor_pipeline.py loaded")