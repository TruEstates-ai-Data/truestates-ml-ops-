import logging
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from pandas.tseries.offsets import MonthEnd
import s3fs

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CODE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = CODE_DIR / 'config.yaml'


def load_config():
    import yaml
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f'config.yaml not found at {CONFIG_PATH}')
    with open(CONFIG_PATH, 'r') as f:
        return yaml.safe_load(f)


def _setup_r2_env(cfg):
    storage = cfg.get('cloud_storage', {})
    if storage.get('endpoint_url'):
        os.environ['AWS_ENDPOINT_URL'] = storage['endpoint_url']
        endpoint_root = storage['endpoint_url'].split('/truestates')[0]
        os.environ['MLFLOW_S3_ENDPOINT_URL'] = endpoint_root
    if storage.get('aws_access_key_id'):
        os.environ['AWS_ACCESS_KEY_ID'] = storage['aws_access_key_id']
    if storage.get('aws_secret_access_key'):
        os.environ['AWS_SECRET_ACCESS_KEY'] = storage['aws_secret_access_key']


def _resolve_path(base, path_value):
    path_value = str(path_value)
    if path_value.startswith('s3://'):
        return path_value
    return os.path.join(str(base), path_value)


def _s3_join(base, *parts):
    base = str(base).rstrip('/')
    if base.startswith('s3://'):
        return '/'.join([base] + [str(p).strip('/') for p in parts])
    return str(Path(base).joinpath(*parts))


def _is_s3(path):
    return str(path).startswith('s3://')


config = load_config()
_setup_r2_env(config)

_paths = config['paths']
_base = _paths.get('base_dir', 's3://dubai')
MODEL_DIR = _s3_join(_base, 'data/model_requirements/models')
COL_DIR = _s3_join(_base, 'data/model_requirements/trained_columns')
ARCHIVE_DIR = _s3_join(_base, config.get('archive', {}).get('folder_name', 'old_files'))

_fs = s3fs.S3FileSystem()

# NOTE: Area -> direct/proxy/group model-key mapping is intentionally NOT
# read from config.yaml here. The caller (Api_pipeline_h.py, via
# AREA_PROXY_MAP / get_lookup_area) is the single source of truth for that
# mapping and is expected to pass in `area_name` already resolved to whatever
# model key should be loaded (the direct area name itself, or its proxy/group
# name, e.g. "proxy1", "G3"). This module simply loads whatever model key it
# is given -- it does not re-derive or second-guess that mapping.

CATEGORICAL_FEATURES = config.get('training_columns', {}).get('cat_cols', [])
NUMERIC_FEATURES = config.get('training_columns', {}).get('num_cols', [])


def get_slug(name):
    return str(name).replace(' ', '_').replace("'", '')


def _path_exists(path):
    path = str(path)
    if _is_s3(path):
        return _fs.exists(path)
    return Path(path).exists()


def safe_load(path):
    path = str(path)
    name = path.rsplit('/', 1)[-1]
    try:
        if _is_s3(path):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.joblib') as tmp:
                local_path = tmp.name
            _fs.get(path, local_path)
            obj = joblib.load(local_path)
            os.remove(local_path)
            return obj
        return joblib.load(path)
    except Exception as e:
        logger.warning(f'Failed to load {name} ({type(e).__name__}): {e}')
        return None


def load_assets(model_key):
    slug = get_slug(model_key)
    candidates = [slug, slug.lower(), slug.upper(), slug.title()]
    logger.info(f"load_assets: searching model/columns for model_key='{model_key}' (candidates={candidates})")
    model = None
    cols = None

    for s in candidates:
        model_path = _s3_join(MODEL_DIR, f'best_model_{s}.joblib')
        col_path = _s3_join(COL_DIR, f'trained_columns_{s}.joblib')
        if model is None and _path_exists(model_path):
            model = safe_load(model_path)
        if cols is None and _path_exists(col_path):
            cols = safe_load(col_path)
        if model is not None and cols is not None:
            break

    if model is None or cols is None:
        logger.warning(f"load_assets: model/columns not found in primary dirs for '{model_key}', checking archive folder.")
        for s in candidates:
            model_path = _s3_join(ARCHIVE_DIR, f'old_best_model_{s}.joblib')
            col_path = _s3_join(ARCHIVE_DIR, f'old_trained_columns_{s}.joblib')
            if model is None and _path_exists(model_path):
                model = safe_load(model_path)
            if cols is None and _path_exists(col_path):
                cols = safe_load(col_path)

    if model is None:
        raise FileNotFoundError(f'Missing model for model_key={model_key!r} (checked candidates: {candidates})')
    if cols is None:
        raise FileNotFoundError(f'Missing trained columns for model_key={model_key!r} (checked candidates: {candidates})')
    logger.info(f"load_assets: successfully loaded model + columns for model_key='{model_key}'")
    return model, cols


def _build_input_frame(input_data, train_columns):
    raw = pd.DataFrame([input_data])
    raw = raw.reindex(columns=['area_name'] + CATEGORICAL_FEATURES + NUMERIC_FEATURES, fill_value=np.nan)
    for c in CATEGORICAL_FEATURES:
        if c in raw.columns:
            raw[c] = raw[c].fillna('Unknown').astype(str)
    for c in NUMERIC_FEATURES:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors='coerce').fillna(0)
    encoded = pd.get_dummies(raw.drop(columns=['area_name'], errors='ignore'), columns=[c for c in CATEGORICAL_FEATURES if c in raw.columns], drop_first=False).astype(float)
    encoded = encoded.reindex(columns=train_columns, fill_value=0)
    return raw, encoded


def predict_property_price(input_data, forecast_df=None, historic_df=None):
    area = input_data['area_name']
    model_key = input_data.get('model_source_area', area)
    logger.info(f"predict_property_price: using model_key='{model_key}' (for load) and area='{area}' (for data).")

    try:
        model, train_columns = load_assets(model_key)
    except Exception as e:
        logger.error(f"predict_property_price: failed to load model assets for model_key='{model_key}' ({type(e).__name__}): {e}")
        raise

    _, encoded_input = _build_input_frame(input_data, train_columns)
    raw_input = pd.DataFrame([input_data]).drop(columns=['area_name'], errors='ignore')

    # Attempt to predict using direct/raw values first (for CatBoost natively)
    try:
        prediction_input = raw_input.reindex(columns=train_columns, fill_value=np.nan)
        for c in CATEGORICAL_FEATURES:
            if c in prediction_input.columns:
                prediction_input[c] = prediction_input[c].fillna('Unknown').astype(str)
        for c in NUMERIC_FEATURES:
            if c in prediction_input.columns:
                prediction_input[c] = pd.to_numeric(prediction_input[c], errors='coerce').fillna(0)
        
        # This will fail for models expecting dummy arrays (RandomForest) because of type mismatch or NaNs
        base_prediction = float(model.predict(prediction_input)[0])
        logger.info("predict_property_price: Successfully used direct raw values for prediction (CatBoost mode).")
    except Exception as raw_e:
        logger.info(f"predict_property_price: Direct raw prediction failed ({type(raw_e).__name__}: {raw_e}), falling back to encoded dummy values.")
        base_prediction = float(model.predict(encoded_input)[0])

    if forecast_df is None:
        forecast_df = pd.DataFrame()
    if historic_df is None:
        historic_df = pd.DataFrame()

    if not forecast_df.empty and 'month' in forecast_df.columns:
        forecast_df = forecast_df.copy()
        forecast_df['month'] = pd.to_datetime(forecast_df['month'], errors='coerce')
    if not historic_df.empty and 'month' in historic_df.columns:
        historic_df = historic_df.copy()
        historic_df['month'] = pd.to_datetime(historic_df['month'], errors='coerce')

    if forecast_df.empty and historic_df.empty:
        return pd.DataFrame({'month': [pd.Timestamp.now()], 'median_price': [base_prediction], 'area': [area]})

    if not historic_df.empty:
        if 'area' in historic_df.columns:
            if area in historic_df['area'].astype(str).unique():
                historic_df = historic_df[historic_df['area'].astype(str) == area].copy()
            else:
                historic_df = historic_df[historic_df['area'].astype(str) == model_key].copy()
        historic_df = historic_df.sort_values('month')
        if 'median_price' not in historic_df.columns:
            historic_df['median_price'] = np.nan

    if not forecast_df.empty:
        if 'area' in forecast_df.columns:
            if area in forecast_df['area'].astype(str).unique():
                forecast_df = forecast_df[forecast_df['area'].astype(str) == area].copy()
            else:
                forecast_df = forecast_df[forecast_df['area'].astype(str) == model_key].copy()
        forecast_df = forecast_df.sort_values('month')
        if 'growth_factor' in forecast_df.columns:
            forecast_df['median_price'] = base_prediction * forecast_df['growth_factor']
        elif 'predictions' in forecast_df.columns:
            forecast_df['median_price'] = forecast_df['predictions']
        else:
            forecast_df['median_price'] = base_prediction

    combined = pd.concat([historic_df, forecast_df], ignore_index=True)
    if combined.empty:
        return pd.DataFrame({'month': [pd.Timestamp.now() + MonthEnd(0)], 'median_price': [base_prediction], 'area': [area]})

    combined['area'] = area
    if 'median_price' not in combined.columns:
        combined['median_price'] = base_prediction
    return combined[['month', 'median_price', 'area']].sort_values('month')


if __name__ == '__main__':
    logger.info('price_predictor_pipeline.py loaded')