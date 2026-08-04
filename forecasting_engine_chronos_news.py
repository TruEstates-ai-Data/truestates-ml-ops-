import os
import pandas as pd
import numpy as np
import requests
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import datetime
import json
import logging
from tqdm import tqdm
import yaml
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

CONFIG_PATH = "config.yaml"
def resolve_path(base, path_val):
    return path_val if str(path_val).startswith("s3://") else os.path.join(base, path_val)

def check_exists(file_path):
    if str(file_path).startswith("s3://"):
        import s3fs
        fs = s3fs.S3FileSystem()
        return fs.exists(file_path)
    else:
        import os
        return os.path.exists(file_path)
    
RSS_FEEDS = {
    "Al Jazeera": "https://www.aljazeera.com/xml/rss/all.xml",
    "Arab Times": "https://www.arabtimesonline.com/rssfeed/47/",
    "Khaleej Times": "https://www.khaleejtimes.com/stories.rss?botrequest=true",
    "BBC News": "http://feeds.bbci.co.uk/news/world/middle_east/rss.xml",
    "Dubai Chronicle": "https://www.dubaichronicle.com/feed/",
    "Abu Dhabi IPR": "https://www.abudhabipr.com/rss/rss.xml",
    "Dubai Confidential": "https://www.dubaiconfidential.ae/feed/",
}

SYSTEM_PROMPT = """You are an information extraction system for macroeconomic, local, regional and geopolitical news.

Your task is to read a news article and extract structured real-world events.
You must strictly follow the output JSON schema provided.

Rules:
- Extract only factual events, not opinions or speculation.
- Use only the provided event taxonomy.
- Do not assign numeric scores, weights, sentiment, or impact.
- Do not infer prices, markets, or economic effects.
- If no relevant event exists, set has_relevant_event = false.
- Your output must be valid JSON and nothing else.
"""

USER_PROMPT_TEMPLATE = """
Extract macro-relevant events from the following news article.

Event taxonomy (allowed event_code values):
- TOURISM_MEGA_PROJECT
- TOURISM_FOOTFALL_SURGE
- GLOBAL_TOURISM_SHOCK
- AIR_CAPACITY_EXPANSION
- MAJOR_INFRA_PROJECT
- URBAN_MASTERPLAN_UPDATE
- PUBLIC_TRANSPORT_EXPANSION
- INFRA_DELAY_OR_CANCEL
- JOB_CREATION_SURGE
- CORPORATE_RELOCATION
- VISA_POLICY_EASING
- LABOR_MARKET_WEAKNESS
- OIL_PRICE_SHOCK_UP
- OIL_PRICE_SHOCK_DOWN
- ENERGY_POLICY_SHIFT
- GLOBAL_RATE_CUT
- GLOBAL_RATE_HIKE
- LIQUIDITY_INJECTION
- CREDIT_TIGHTENING
- GEO_RISK_ESCALATION
- GEO_RISK_DEESCALATION
- NEW_SUPPLY_ANNOUNCEMENT
- SUPPLY_DELAY
- REGULATORY_TIGHTENING_RE
- REGULATORY_EASING_RE
- ECONOMIC_EXPANSION
- ECONOMIC_CONTRACTION
- PANDEMIC_OUTBREAK

Article:
Title: {title}
Published date: {published_date}
Description: {description}

Output JSON schema:
{{
  "article_id": "string",
  "source": "BBC",
  "title": "title",
  "published_date": "YYYY-MM-DD",
  "has_relevant_event": true | false,
  "events": [
    {{
      "event_id": "string",
      "event_code": "string",
      "event_subtype": "string | null",
      "primary_entity": {{
        "name": "string",
        "type": "government | company | institution | person | unknown"
      }},
      "secondary_entities": [
        {{
          "name": "string",
          "type": "government | company | institution | person | unknown"
        }}
      ],
      "countries": ["string"],
      "regions": ["string"],
      "event_nature": {{
        "is_policy": false,
        "is_conflict": false,
        "is_economic": false,
        "is_regulatory": false,
        "is_infrastructure": false
      }},
      "time_horizon": "immediate | short_term | medium_term | long_term | unknown",
      "raw_facts": {{
        "announced_numbers": [],
        "keywords": [],
        "quoted_phrases": []
      }},
      "confidence": 0.0
    }}
  ]
}}
"""

WEIGHTS_PROMPT = """You are an expert analyst specializing in assessing the strength of real-world events based on their characteristics.
Given the news titles, event code, assign the the weights for the event from the JSON based on the respectively assigned entity tiers, event scales and geo relevance else assign the default weight present in each of the JSONs if the event does not fit into any of the categories in the JSONs.

News titles:
{title_map}

Event code:
{event_code_map}

Entity Tiers:
{entity_tiers}

Event Scales:
{event_scales}

Geo Relevance:
{geo_relevance}

Output a JSON with the following format:
{{
  "event_id": "string",
  "event_code": "string",
  "titles": ["string"],
  "assigned_entity_tiers": ["string"],
  "assigned_event_scales": ["string"],
  "assigned_geo_relevance": ["string"],
  "assigned_weights": {{
    "entity_tier_weight": float,
    "event_scale_weight": float,
    "geo_relevance_weight": float
  }}
}}
"""


def fill_missing_with_column_name(df: pd.DataFrame) -> pd.DataFrame:
    df_clean = df.copy()
    for col in df_clean.columns:
        if df_clean[col].dtype == "object" or pd.api.types.is_string_dtype(df_clean[col]):
            df_clean[col] = df_clean[col].fillna(f"missing_{col}")
        else:
            df_clean[col] = df_clean[col].fillna(0)
    return df_clean


def load_json_config(filepath: str, default_val: Any = None) -> Any:
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Failed to load {filepath}. Using default. Error: {e}")
        return default_val if default_val is not None else {}


def load_yaml_config(filepath: str) -> dict:
    try:
        with open(filepath, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logger.warning(f"Failed to load yaml config from {filepath}. Error: {e}")
        return {}


def normalize_area_id(val):
    if isinstance(val, pd.Series):
        return val.apply(normalize_area_id)
    if isinstance(val, np.ndarray):
        return pd.Series(val).apply(normalize_area_id)
    if val is None:
        return None
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    sval = str(val).strip()
    if sval.endswith(".0"):
        sval = sval[:-2]
    return sval


def drop_duplicate_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[:, ~df.columns.duplicated()].copy()


app_config = load_yaml_config(CONFIG_PATH)
macro_news_settings = app_config.get("macro_news_settings", {})
paths_config = app_config.get("paths", {})

OLLAMA_URL = macro_news_settings.get("llm_api_url", "http://localhost:11434/api/generate")
MODEL = macro_news_settings.get("llm_model", "gpt-oss:120b-cloud")
PROJECTION_MONTHS_AHEAD = macro_news_settings.get("projection_months_ahead", 6)


def call_llm(prompt: str, system_prompt: str = "", model: str = MODEL) -> str:
    payload = {
        "model": model,
        "prompt": prompt,
        "system": system_prompt,
        "temperature": 0.0,
        "stream": False
    }
    try:
        r = requests.post(OLLAMA_URL, json=payload, timeout=120)
        r.raise_for_status()
        return r.json()["response"]
    except requests.exceptions.RequestException as e:
        logger.error(f"LLM API call failed: {e}")
        raise


class NewsFetcher:
    @staticmethod
    def fetch_xml(url: str) -> str:
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.text
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to fetch RSS from {url}: {e}")
            return ""

    @staticmethod
    def get_geo_tags(title: str, desc: str, source: str):
        text = f"{title} {desc}".lower()

        if any(k in text for k in ["dubai", "abu dhabi", "sharjah", "ajman", "rak", "fujairah"]):
            return "GCC", "UAE"
        if "saudi" in text or "riyadh" in text:
            return "GCC", "Saudi Arabia"
        if "kuwait" in text:
            return "GCC", "Kuwait"
        if "bahrain" in text:
            return "GCC", "Bahrain"
        if "oman" in text or "muscat" in text:
            return "GCC", "Oman"
        if "qatar" in text or "doha" in text:
            return "GCC", "Qatar"
        if "iran" in text:
            return "MENA", "Iran"
        if "nigeria" in text:
            return "Africa", "Nigeria"
        if "philippines" in text:
            return "Asia", "Philippines"

        if source in ["Dubai Chronicle", "Abu Dhabi IPR", "Dubai Confidential", "Khaleej Times"]:
            return "GCC", "UAE"

        return "Global", "Global"

    @classmethod
    def parse_standard_feed(cls, source_name: str) -> List[Dict[str, Any]]:
        url = RSS_FEEDS.get(source_name)
        if not url:
            return []

        xml_str = cls.fetch_xml(url)
        if not xml_str:
            return []

        try:
            root = ET.fromstring(xml_str)
            channel = root.find("channel")
            if channel is None:
                return []

            items = channel.findall("item")
            out = []

            for it in items:
                title = (it.findtext("title") or "").strip()
                desc = (it.findtext("description") or "").strip()
                cat = (it.findtext("category") or "").strip()
                region, country = cls.get_geo_tags(title, desc, source_name)

                out.append({
                    "title": title,
                    "link": (it.findtext("link") or "").strip(),
                    "source": source_name,
                    "snippet": desc,
                    "published_date": (it.findtext("pubDate") or "").strip(),
                    "category": cat,
                    "region": region,
                    "country": country,
                })
            return out
        except ET.ParseError as e:
            logger.error(f"XML Parsing failed for {source_name}: {e}")
            return []

    @classmethod
    def build_all_results_from_rss(cls) -> pd.DataFrame:
        sources = ["Arab Times", "Al Jazeera", "Abu Dhabi IPR", "Dubai Confidential"]
        all_items = []

        for source in sources:
            all_items.extend(cls.parse_standard_feed(source))

        df = pd.DataFrame(all_items)
        if not df.empty:
            df = fill_missing_with_column_name(df)
            df.drop_duplicates(["title", "published_date"], inplace=True)
            df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce", utc=True).dt.strftime("%Y-%m-%d")

        return df


class EventProcessor:
    def __init__(self, entity_tiers: dict, event_scales: dict, geo_relevance: dict):
        self.entity_tiers = entity_tiers
        self.event_scales = event_scales
        self.geo_relevance = geo_relevance

    def extract_events(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []

        if df.empty:
            return pd.DataFrame()

        for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting Events"):
            prompt = USER_PROMPT_TEMPLATE.format(
                title=row.get("title", ""),
                published_date=row.get("published_date", ""),
                description=row.get("snippet", "")
            )
            try:
                raw_output = call_llm(prompt, system_prompt=SYSTEM_PROMPT)
                parsed = json.loads(raw_output)
                results.append({
                    "article_id": parsed.get("article_id", f"idx_{i}"),
                    "published_date": parsed.get("published_date", row.get("published_date")),
                    "title": parsed.get("title", row.get("title")),
                    "has_relevant_event": parsed.get("has_relevant_event", False),
                    "events": parsed.get("events", [])
                })
            except Exception as e:
                logger.error(f"Event extraction failed for index {i}: {e}")

        out_df = pd.DataFrame(results)
        return out_df[out_df["has_relevant_event"] == True].copy() if not out_df.empty else out_df

    def score_events(self, events_df: pd.DataFrame) -> pd.DataFrame:
        if events_df.empty:
            return events_df

        weight_responses = []
        for i, row in tqdm(events_df.iterrows(), total=len(events_df), desc="Scoring Events"):
            event_code_map = {
                row.get("article_id", i): row["events"][0].get("event_code", "unknown") if row["events"] else "unknown"
            }
            title_map = {
                row.get("article_id", i): str(row["title"])
            }

            prompt = WEIGHTS_PROMPT.format(
                event_code_map=event_code_map,
                title_map=title_map,
                entity_tiers=json.dumps(self.entity_tiers),
                event_scales=json.dumps(self.event_scales),
                geo_relevance=json.dumps(self.geo_relevance)
            )

            try:
                raw_output = call_llm(prompt)
                parsed = json.loads(raw_output)
                weight_responses.append(parsed)
            except Exception as e:
                logger.warning(f"Scoring failed for {row['title']}, using defaults. {e}")
                weight_responses.append({
                    "event_id": list(event_code_map.keys())[0],
                    "assigned_weights": {
                        "entity_tier_weight": 0.5,
                        "event_scale_weight": 0.5,
                        "geo_relevance_weight": 0.5
                    }
                })

        weights_df = pd.DataFrame(weight_responses)
        extract_w = lambda x, key: x.get(key, 0.5) if isinstance(x, dict) else 0.5

        events_df["entity_tier_weight"] = weights_df["assigned_weights"].apply(lambda x: extract_w(x, "entity_tier_weight"))
        events_df["event_scale_weight"] = weights_df["assigned_weights"].apply(lambda x: extract_w(x, "event_scale_weight"))
        events_df["geo_relevance_weight"] = weights_df["assigned_weights"].apply(lambda x: extract_w(x, "geo_relevance_weight"))

        events_df["strength"] = events_df.apply(
            lambda r: np.clip(
                r["entity_tier_weight"] * r["event_scale_weight"] * r["geo_relevance_weight"],
                0.2,
                1.5
            ),
            axis=1
        )
        events_df["event_code"] = events_df["events"].apply(lambda x: x[0].get("event_code", "unknown") if x else "unknown")
        return events_df


class MacroImpactModeler:
    def __init__(self, event_to_channel: dict, area_sens: dict, decay_params: dict):
        self.event_to_channel = event_to_channel
        self.area_sens = area_sens
        self.decay_params = decay_params

    def aggregate_monthly_factors(self, scored_df: pd.DataFrame) -> pd.DataFrame:
        channel_rows = []
        for _, row in scored_df.iterrows():
            event_code = row["event_code"]
            if event_code not in self.event_to_channel:
                continue

            for channel, weight in self.event_to_channel[event_code].items():
                channel_rows.append({
                    "published_date": row["published_date"],
                    "channel": channel,
                    "title": row["title"],
                    "contribution": row["strength"] * weight
                })

        channel_df = pd.DataFrame(channel_rows)
        if channel_df.empty:
            return pd.DataFrame()

        channel_df["published_date"] = pd.to_datetime(channel_df["published_date"], errors="coerce")
        channel_df = channel_df.dropna(subset=["published_date"])
        channel_df.drop_duplicates(["title", "channel"], inplace=True)

        monthly_channels = channel_df.groupby(["published_date", "channel"], as_index=False).agg({
            "contribution": "sum",
            "title": lambda x: list(set(x))
        })

        contrib_pivot = monthly_channels.pivot_table(
            index="published_date",
            columns="channel",
            values="contribution",
            aggfunc="sum"
        ).fillna(0)

        title_list = monthly_channels.groupby("published_date")["title"].apply(
            lambda x: list(set([item for sublist in x for item in sublist]))
        )

        monthly_pivot = contrib_pivot.join(title_list).reset_index()

        channels = [c for c in monthly_pivot.columns if c not in ["published_date", "title"]]
        for c in channels:
            med, q1, q3 = monthly_pivot[c].median(), monthly_pivot[c].quantile(0), monthly_pivot[c].quantile(1)
            iqr = max(q3 - q1, 1e-6)
            monthly_pivot[c] = np.tanh((monthly_pivot[c] - med) / iqr)

        return monthly_pivot

    def calculate_area_scores(self, monthly_aggregated: pd.DataFrame) -> pd.DataFrame:
        if monthly_aggregated.empty:
            return pd.DataFrame(columns=["published_date", "area", "macro_news_factor", "drivers", "month"])

        channels = [c for c in monthly_aggregated.columns if c not in ["month", "titles_list", "published_date", "title"]]
        rows = []

        for _, r in monthly_aggregated.iterrows():
            pub_date = r["published_date"]
            for area, sens in self.area_sens.items():
                area_score = 0.0
                drivers = {}

                for c in channels:
                    w = sens.get(c, 0.0)
                    contrib = r.get(c, 0) * w
                    area_score += contrib
                    if abs(contrib) > 0.01:
                        drivers[c] = round(contrib, 4)

                rows.append({
                    "published_date": pub_date,
                    "area": area.title().replace("_", " "),
                    "macro_news_factor": area_score,
                    "drivers": json.dumps(drivers)
                })

        area_df = pd.DataFrame(rows)
        if not area_df.empty:
            area_df["month"] = pd.to_datetime(area_df["published_date"]).dt.strftime("%Y-%m")
        return area_df


class DecayModeler:
    EVENT_DECAY_PARAMS = {
        "GEO_RISK_ESCALATION": {"s_halflife": 1.5, "p_peak": 4, "p_sensitivity": 0.020, "direction": -1},
        "GEO_RISK_DEESCALATION": {"s_halflife": 1.5, "p_peak": 2, "p_sensitivity": 0.010, "direction": +1},
        "INFRA_DELAY_OR_CANCEL": {"s_halflife": 4.0, "p_peak": 6, "p_sensitivity": 0.012, "direction": -1},
        "REGULATORY_TIGHTENING_RE": {"s_halflife": 5.0, "p_peak": 5, "p_sensitivity": 0.015, "direction": -1},
        "REGULATORY_EASING_RE": {"s_halflife": 3.0, "p_peak": 3, "p_sensitivity": 0.012, "direction": +1},
        "SUPPLY_DELAY": {"s_halflife": 3.0, "p_peak": 5, "p_sensitivity": 0.008, "direction": -1},
        "OIL_PRICE_SHOCK_UP": {"s_halflife": 2.0, "p_peak": 3, "p_sensitivity": 0.010, "direction": +1},
        "ENERGY_POLICY_SHIFT": {"s_halflife": 6.0, "p_peak": 8, "p_sensitivity": 0.008, "direction": -1},
        "GLOBAL_TOURISM_SHOCK": {"s_halflife": 3.0, "p_peak": 4, "p_sensitivity": 0.018, "direction": -1},
        "NEW_SUPPLY_ANNOUNCEMENT": {"s_halflife": 4.0, "p_peak": 6, "p_sensitivity": 0.010, "direction": -1},
        "PANDEMIC_OUTBREAK": {"s_halflife": 1.0, "p_peak": 3, "p_sensitivity": 0.025, "direction": -1},
        "DEFAULT": {"s_halflife": 3.0, "p_peak": 4, "p_sensitivity": 0.015, "direction": -1},
    }

    SEVERITY_RANK = {
        "GEO_RISK_ESCALATION": 10,
        "PANDEMIC_OUTBREAK": 9,
        "GLOBAL_TOURISM_SHOCK": 8,
        "REGULATORY_TIGHTENING_RE": 7,
        "OIL_PRICE_SHOCK_UP": 6,
        "INFRA_DELAY_OR_CANCEL": 5,
        "ENERGY_POLICY_SHIFT": 4,
        "SUPPLY_DELAY": 5,
        "NEW_SUPPLY_ANNOUNCEMENT": 5,
        "REGULATORY_EASING_RE": 6,
        "GEO_RISK_DEESCALATION": 7,
    }

    @classmethod
    def get_dominant_event(cls, event_df: pd.DataFrame, strategy="severity") -> str:
        if event_df.empty:
            return "DEFAULT"
        if strategy == "frequency":
            return event_df["event_code"].value_counts().idxmax()
        elif strategy == "severity":
            present_events = event_df["event_code"].unique()
            return max(present_events, key=lambda e: cls.SEVERITY_RANK.get(e, 0))
        return "DEFAULT"

    @classmethod
    def dubai_re_decay_dynamic(cls, months_since_event: int, event_code: str):
        params = cls.EVENT_DECAY_PARAMS.get(event_code, cls.EVENT_DECAY_PARAMS["DEFAULT"])
        s_halflife, p_peak = params["s_halflife"], params["p_peak"]

        decay_rate = np.log(2) / s_halflife
        sentiment_decay = np.exp(-decay_rate * months_since_event)

        k = p_peak * decay_rate
        price_impact = (months_since_event ** k) * np.exp(-decay_rate * months_since_event)

        peak_val = (k / decay_rate) ** k * np.exp(-k)
        price_impact_norm = price_impact / (peak_val + 1e-9)

        return sentiment_decay, price_impact_norm, params

    @classmethod
    def apply_decay_to_pipeline(cls, totest_df: pd.DataFrame, scored_df: pd.DataFrame, shock_month: str) -> pd.DataFrame:
        dominant_event = cls.get_dominant_event(scored_df, strategy="severity")
        event_date = pd.to_datetime(shock_month)

        v0_lookup = {}
        for area_id, group in totest_df.groupby("area_id"):
            non_zero = group[group["macro_news_factor"] != 0]
            if not non_zero.empty:
                v0_lookup[area_id] = non_zero.iloc[-1]["macro_news_factor"]
            else:
                v0_lookup[area_id] = 0.0

        def _apply_row(row):
            month_dt = pd.to_datetime(row["month"])
            t = (month_dt.year - event_date.year) * 12 + (month_dt.month - event_date.month)

            v0 = v0_lookup.get(row["area_id"], 0)

            if t < 0:
                return row["predictions"], row.get("macro_news_factor", 0), 1.0, 0.0

            sentiment_decay, price_impact_norm, params = cls.dubai_re_decay_dynamic(t, dominant_event)

            current_news_factor = v0 * sentiment_decay

            if abs(v0) > 0.0001:
                structural_impact = price_impact_norm * params["p_sensitivity"] * abs(params["direction"])
            else:
                structural_impact = 0.0

            total_adj = 1 + current_news_factor - structural_impact
            adj_price = row["predictions"] * total_adj

            return adj_price, current_news_factor, sentiment_decay, price_impact_norm

        totest_df = totest_df.copy()
        totest_df[["adjusted_pred", "macro_news_factor", "sentiment_decay", "price_impact"]] = totest_df.apply(
            lambda r: pd.Series(_apply_row(r)), axis=1
        )
        return totest_df


class NarrativeSynthesizer:
    @staticmethod
    def load_previous_context(filepath: str) -> dict:
        try:
            with open(filepath, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            logger.info("No valid previous context found. Proceeding with None.")
            return {}

    @staticmethod
    def get_dynamic_date_range(months_ahead: int = 6) -> tuple[str, str]:
        start_date = pd.Timestamp.now()
        end_date = start_date + pd.DateOffset(months=months_ahead)
        return start_date.strftime("%Y-%m"), end_date.strftime("%Y-%m")

    @classmethod
    def generate_narratives(
        cls,
        totest_df: pd.DataFrame,
        monthly_aggregated: pd.DataFrame,
        area_sens_config: dict,
        prev_context: dict,
        months_ahead: int = 6,
    ) -> Dict[str, str]:
        logger_local = logging.getLogger(__name__)

        start_month, end_month = cls.get_dynamic_date_range(months_ahead)

        tot = totest_df.copy()
        if "month" in tot.columns:
            tot["month"] = pd.to_datetime(tot["month"]).dt.strftime("%Y-%m")

        mon = monthly_aggregated.copy()

        if "published_date" in mon.columns:
            mon.rename(columns={"published_date": "month", "title": "titles_list"}, inplace=True)

        if "month" in mon.columns:
            mon["month"] = pd.to_datetime(mon["month"]).dt.strftime("%Y-%m")

        channels = [c for c in mon.columns if c not in ["month", "titles_list"]]
        per_area_narratives = {}

        if tot.empty:
            return per_area_narratives

        for area_name in tot["area"].dropna().unique():
            area_df = tot[tot["area"] == area_name]
            if area_df.empty:
                continue

            area_id = area_df["area_id"].iloc[0]

            available_months = sorted([m for m in mon["month"].unique() if m < start_month]) if not mon.empty else []
            last_month_with_drivers = None
            drivers_snapshot = {}

            for m in reversed(available_months):
                row = mon[mon["month"] == m]
                if not row.empty:
                    r = row.iloc[0]
                    t = r.get("titles_list", [])

                    if t and ((isinstance(t, list) and len(t) > 0) or isinstance(t, str)):
                        last_month_with_drivers = m
                        specific_sens = area_sens_config.get(area_name, {})
                        for c in channels:
                            mval = r.get(c, 0)
                            w = specific_sens.get(c, 0.0)
                            contrib = float(mval or 0) * float(w or 0)
                            if abs(contrib) > 0.01:
                                drivers_snapshot[c] = round(contrib, 4)
                        break

            has_news = last_month_with_drivers is not None

            if not has_news:
                logger_local.info(f"No historical news found for {area_name}. Generating forecast-only narrative.")
                last_month_str = "N/A (No recent news data)"
                drivers_text = "{}\n*Note: No major macroeconomic catalysts recorded recently.*"
                tense_rules_text = (
                    "- Use FUTURE/CONDITIONAL tense ONLY for all forward-looking forecasted months "
                    "(e.g., 'prices are forecast to recover', 'market is expected to stabilise')."
                )
                structure_text = (
                    "BASELINE TREND- Explain the forecasted trajectory for the area based purely on structural "
                    "momentum and predictions, noting the absence of recent external macro shocks."
                )
            else:
                last_month_str = last_month_with_drivers
                drivers_text = (
                    f"{json.dumps(drivers_snapshot)}\n"
                    "*Note: Interpret these as 'Impact Shares'. If a value is high relative to others, "
                    "it is the primary driver.*"
                )
                tense_rules_text = f""" - If the month is ON OR BEFORE {last_month_with_drivers}: use PAST TENSE.
(e.g., "confidence stood at 67%", "the market fell", "prices declined")
- If the month is AFTER {last_month_with_drivers}: use FUTURE/CONDITIONAL tense ONLY.
(e.g., "confidence is expected to reach 67%", "prices are forecast to recover",
"the market will likely stabilise")
- NEVER write past-tense statements about months that have not yet occurred."""
                structure_text = f"""TRIGGER (past tense)- Which specific headlines from {last_month_with_drivers} caused the shift 
and why — name the actual drivers, not generic categories.
AMPLIFIER (past tense)- How {area_name}'s reliance on {json.dumps(area_sens_config.get(area_name, {}))} made the impact 
worse than a typical area would experience."""

            price_trajectory = []
            for m in pd.date_range(start=start_month, end=end_month, freq="MS").strftime("%Y-%m"):
                row = tot[(tot["area_id"] == area_id) & (tot["month"] == m)]
                if not row.empty:
                    r = row.iloc[0]
                    baseline = r.get("predictions", 0)
                    adjusted = r.get("adjusted_pred", 0)
                    sentiment_decay = r.get("sentiment_decay", None)

                    if baseline != 0:
                        delta_pct = ((adjusted - baseline) / baseline * 100)
                        price_trajectory.append({
                            "month": m,
                            "baseline": round(float(baseline), 2),
                            "adjusted": round(float(adjusted), 2),
                            "change_pct": round(delta_pct, 2),
                            "sentiment_decay": round(float(sentiment_decay), 3) if pd.notnull(sentiment_decay) else None
                        })

            if not price_trajectory:
                per_area_narratives[area_name] = "Narrative unavailable: No upcoming prediction window available."
                continue

            prompt = f"""You are a senior Real Estate Strategist. Your goal is to explain market movements to an investor in plain English.

CONTEXT:
Area: {area_name}
Today's Date: {pd.Timestamp.now().strftime('%B %Y')}
Analysis Period: {start_month} to {end_month}
Last Known News Month: {last_month_str}

MARKET DRIVERS:
{drivers_text}

AREA SENSITIVITIES:
{json.dumps(area_sens_config.get(area_name, {}))}
*Note: This explains why this area is more/less vulnerable to certain economic elements.*

PRICE & RECOVERY DATA:
{json.dumps(price_trajectory)}
*Note: 'sentiment_decay' is the Recovery Strength (1.0 = Full Confidence, 0.0 = Panic).*

PREVIOUS ITERATION CONTEXT:
{json.dumps(prev_context) if prev_context else "None"}
*Note: This is the raw data from the previous iteration, which may help you understand how news effects evolved.*

INSTRUCTIONS:
1. NO RAW SCORES: Never mention numbers like '0.5' or '-0.9'. 
Use percentages of impact only. (e.g., 'Geopolitical risk accounted for 70% of the downward pressure').

2. STRICT TENSE RULES — apply this decision for every month you mention:
{tense_rules_text}

3. CONFIDENCE TRANSLATION: Translate 'sentiment_decay' into 'Market Confidence Index'.
(e.g., decay of 0.67 → "Market Confidence Index at 67%")

4. BANNED METRIC — 'price lag': Do NOT use the phrase 'price lag' or quote any price lag 
percentage in the narrative. Instead describe price movement qualitatively:
- High lag → "prices have been slow to respond" / "the market remains under pressure"
- Recovering lag → "pricing is beginning to stabilise" / "recovery is gaining traction"
- Low lag → "prices have largely held firm"

5. PREVIOUS ITERATION CONTEXT: Refer to the previous iteration's raw data to understand how news effects evolved. Use this to add depth to your analysis, especially in the Recovery Outlook section.

SYNTHESIS TASK:
Write a 100-word narrative for an investor audience. Structure it as follows:

{structure_text}
RECOVERY OUTLOOK (future tense ONLY — these months have not happened yet)-
Describe how confidence is expected to evolve month by month. 
Use ONLY phrases like "is expected to", "is forecast to", "will likely".
PREVIOUS ITERATION INSIGHTS:
How previous iteration context informs your outlook.
VERDICT: Is this a temporary shock or a permanent repricing? Keep it one sentence.
*Note: you do not have to add separate sections as TRIGGER, AMPLIFIER, etc. Just make sure to cover all these points in a cohesive narrative.*
"""
            try:
                llm_response = call_llm(prompt, model=MODEL)
                per_area_narratives[area_name] = llm_response.strip()
            except Exception as e:
                logger.error(f"LLM synthesis call failed for area {area_name}: {e}")
                per_area_narratives[area_name] = "Narrative unavailable due to system processing error."

        return per_area_narratives


def run_market_forecasting_pipeline(chronos_df: pd.DataFrame, area_to_id: dict) -> pd.DataFrame:
    logger.info("Initializing configuration and fetching news...")

    base_dir = paths_config.get("base_dir", ".")

    # area_sens_path = os.path.join(base_dir, paths_config.get("area_sens_path", "utils/area_sensitivity.json"))
    # entity_tiers_path = os.path.join(base_dir, paths_config.get("entity_tiers_path", "utils/entity_tiers.json"))
    # event_scales_path = os.path.join(base_dir, paths_config.get("event_scales_path", "utils/event_scale_bins.json"))
    # event_to_channel_path = os.path.join(base_dir, paths_config.get("event_to_channel_path", "utils/event_to_channel.json"))
    # geo_relevance_path = os.path.join(base_dir, paths_config.get("geo_relevance_path", "utils/geo_relevance.json"))
    # news_context_file = os.path.join(base_dir, paths_config.get("news_context_file", "news_context.json"))
    # area_sens_path = resolve_path(base_dir, paths_config.get("area_sens_path", "utils/area_sensitivity.json"))
    # entity_tiers_path = resolve_path(base_dir, paths_config.get("entity_tiers_path", "utils/entity_tiers.json"))
    # event_scales_path = resolve_path(base_dir, paths_config.get("event_scales_path", "utils/event_scale_bins.json"))
    # event_to_channel_path = resolve_path(base_dir, paths_config.get("event_to_channel_path", "utils/event_to_channel.json"))
    # geo_relevance_path = resolve_path(base_dir, paths_config.get("geo_relevance_path", "utils/geo_relevance.json"))
    # news_context_file = resolve_path(base_dir, paths_config.get("news_context_file", "news_context.json"))

    # The lines you just pasted:
    entity_tiers_path = "s3://dubai/data/utils/entity_tiers.json"
    event_scales_path = "s3://dubai/data/utils/event_scale_bins.json"
    geo_relevance_path = "s3://dubai/data/utils/geo_relevance.json"
    event_to_channel_path = "s3://dubai/data/utils/event_to_channel.json"
    area_sens_path = "s3://dubai/data/utils/area_sensitivity.json"
    news_context_file = "s3://dubai/data/utils/news_context.json"

    # Custom R2 JSON Loader
    # Custom R2 JSON Loader (Updated for Cloudflare R2 Auth)
    def load_r2_json(file_path):
        import s3fs, json, os
        try:
            fs = s3fs.S3FileSystem(
                key=os.environ.get("AWS_ACCESS_KEY_ID", "c198c85bd01da0931eae24009fb2100b"),
                secret=os.environ.get("AWS_SECRET_ACCESS_KEY", "826187ffaee4742816f65ca4ebe149902db75ac52dbb81606bb34fe8bae4a57c"),
                client_kwargs={
                    'endpoint_url': os.environ.get("AWS_ENDPOINT_URL_S3", "https://ef8eef61229ee8854b4237f6949e50d8.r2.cloudflarestorage.com/truestates-re-analytics"),
                    'region_name': 'auto'
                },
                config_kwargs={'signature_version': 's3v4'}
            )
            with fs.open(file_path, 'rb') as f:
                return json.load(f)
        except Exception as e:
            print(f"Failed to load {file_path} from R2: {e}")
            return {}

    # Load the files directly using the credentials!
    entity_tiers = load_r2_json(entity_tiers_path)
    event_scales = load_r2_json(event_scales_path)
    geo_relevance = load_r2_json(geo_relevance_path)
    event_to_channel = load_r2_json(event_to_channel_path)
    area_sens = load_r2_json(area_sens_path)


    news_df = NewsFetcher.build_all_results_from_rss()

    processor = EventProcessor(entity_tiers, event_scales, geo_relevance)
    events_df = processor.extract_events(news_df)
    scored_df = processor.score_events(events_df)

    modeler = MacroImpactModeler(event_to_channel, area_sens, {})
    monthly_factors = modeler.aggregate_monthly_factors(scored_df)
    area_scores_df = modeler.calculate_area_scores(monthly_factors)

    area_scores_df = area_scores_df.copy()
    area_scores_df["area_id"] = area_scores_df["area"].map(area_to_id)
    area_scores_df["area_id"] = area_scores_df["area_id"].apply(normalize_area_id)

    id_to_area = {normalize_area_id(v): k for k, v in area_to_id.items()}

    chronos_df = drop_duplicate_columns(chronos_df)
    chronos_df = chronos_df.copy()
    chronos_df["month"] = pd.to_datetime(chronos_df["month"], errors="coerce").dt.strftime("%Y-%m")
    chronos_df["area_id"] = chronos_df["area_id"].apply(normalize_area_id)
    chronos_df["predictions"] = pd.to_numeric(chronos_df["predictions"], errors="coerce")

    totest = chronos_df.merge(
        area_scores_df[["month", "area_id", "drivers", "macro_news_factor"]],
        on=["month", "area_id"],
        how="left"
    )

    if "area_name" in totest.columns:
        totest["area"] = totest["area_name"]
    else:
        totest["area"] = totest["area_id"].map(id_to_area)

    totest["predictions"] = pd.to_numeric(totest["predictions"], errors="coerce").fillna(0)
    totest["macro_news_factor"] = pd.to_numeric(totest["macro_news_factor"], errors="coerce").fillna(0)

    if not area_scores_df.empty and area_scores_df["month"].notna().any():
        shock_month_str = area_scores_df["month"].dropna().max()
    else:
        shock_month_str = pd.Timestamp.now().strftime("%Y-%m")

    totest = DecayModeler.apply_decay_to_pipeline(totest, scored_df, shock_month_str)
    totest = fill_missing_with_column_name(totest)

    prev_context = NarrativeSynthesizer.load_previous_context(news_context_file)

    narrative_dict = NarrativeSynthesizer.generate_narratives(
        totest_df=totest,
        monthly_aggregated=monthly_factors,
        area_sens_config=area_sens,
        prev_context=prev_context,
        months_ahead=PROJECTION_MONTHS_AHEAD
    )

    totest["narrative"] = totest["area"].map(narrative_dict)

    totest.rename(columns={"month": "date"}, inplace=True)
    out_cols = ["area", "date", "predictions", "macro_news_factor", "adjusted_pred", "narrative", "area_id"]
    if "predicted_mom_growth_pct" in totest.columns:
        out_cols.append("predicted_mom_growth_pct")
    if "model_area_id" in totest.columns:
        out_cols.append("model_area_id")
    final_output = totest[out_cols].copy()

    logger.info("Pipeline execution complete.")
    return final_output


def execute_pipeline_entry(config=None):
    global app_config, macro_news_settings, paths_config, OLLAMA_URL, MODEL, PROJECTION_MONTHS_AHEAD

    if config is not None:
        app_config = config
        macro_news_settings = app_config.get("macro_news_settings", {})
        paths_config = app_config.get("paths", {})
        OLLAMA_URL = macro_news_settings.get("llm_api_url", "http://51.38.112.237:11434/api/generate")
        MODEL = macro_news_settings.get("llm_model", "gpt-oss:120b-cloud")
        PROJECTION_MONTHS_AHEAD = macro_news_settings.get("projection_months_ahead", 6)

    base_dir = paths_config.get("base_dir", ".")
    # forecast_file = os.path.join(base_dir, paths_config.get("chronos_output", "final_chronos_forecasts.csv"))
    # output_file = os.path.join(base_dir, paths_config.get("adjusted_forecast_output", "adjusted_macro_forecast.csv"))
    forecast_file = resolve_path(base_dir, paths_config.get("chronos_output", "final_chronos_forecasts.csv"))
    output_file = resolve_path(base_dir, paths_config.get("adjusted_forecast_output", "adjusted_macro_forecast.csv"))

    raw_df = pd.read_csv(forecast_file)
    raw_df = drop_duplicate_columns(raw_df)

    logger.info(f"Forecast file columns: {list(raw_df.columns)}")

    if "area_id" not in raw_df.columns:
        raise KeyError("Expected 'area_id' column not found in forecast file.")
    if "predicted_monthly_price" not in raw_df.columns:
        raise KeyError("Expected 'predicted_monthly_price' column not found in forecast file.")
    if "area_name" not in raw_df.columns:
        raise KeyError("Expected 'area_name' column not found in forecast file.")

    test_df = raw_df.copy()
    test_df = drop_duplicate_columns(test_df)

    test_df["month"] = pd.to_datetime(test_df["month"], errors="coerce").dt.strftime("%Y-%m")
    test_df["area_id"] = pd.Series(test_df["area_id"]).apply(normalize_area_id)
    test_df["predictions"] = pd.to_numeric(test_df["predicted_monthly_price"], errors="coerce")
    test_df["area_name"] = test_df["area_name"].astype(str)

    keep_cols = ["area_id", "month", "predictions", "area_name"]
    if "predicted_mom_growth_pct" in test_df.columns:
        keep_cols.append("predicted_mom_growth_pct")
    if "model_area_id" in test_df.columns:
        keep_cols.append("model_area_id")
    test_df = test_df[keep_cols].dropna(subset=["area_id", "month", "predictions", "area_name"]).drop_duplicates()

    area_to_id_mapping = (
        test_df[["area_name", "area_id"]]
        .drop_duplicates()
        .rename(columns={"area_name": "area"})
    )
    area_to_id_mapping = dict(zip(area_to_id_mapping["area"], area_to_id_mapping["area_id"]))

    final_df = run_market_forecasting_pipeline(test_df, area_to_id_mapping)
    # final_df.to_csv(output_file, index=False)
    if str(output_file).startswith("s3://"):
        import s3fs, tempfile, os
        fs = s3fs.S3FileSystem()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            local_path = tmp.name
        final_df.to_csv(local_path, index=False)  # <-- Changed here!
        fs.put(local_path, output_file)
        os.remove(local_path)
    else:
        final_df.to_csv(output_file, index=False) # <-- Changed here!

    logger.info(f"Adjusted macro forecast saved to: {output_file}")

    # Log forecasting news metrics to MLflow
    try:
        mlflow.log_metric("adjusted_forecast_rows", len(final_df) if final_df is not None else 0)
        mlflow.log_param("llm_model", config.get('macro_news_settings', {}).get('llm_model', 'unknown') if config is not None else MODEL)
    except Exception as e:
        logger.warning(f"MLflow forecasting news logging failed: {e}")

    return final_df


if __name__ == "__main__":
    execute_pipeline_entry()