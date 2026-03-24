import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

_REGION_TYPES = {"city", "town", "neighborhood"}
_DEFAULT_REQUIRED_COLUMNS = ["RegionName", "RegionType", "Metro"]


@dataclass(frozen=True)
class DataSourceDecision:
    mode: str
    source: str
    reason: str


_cache_store: Dict[str, Tuple[float, pd.DataFrame]] = {}


def normalize_key(text: str) -> str:
    """Lowercase, strip non-alphanumerics so 'St. Louis' → 'stlouis'."""
    return re.sub(r"[^a-z0-9]", "", (text or "").lower())


def _get_data_source_mode() -> str:
    mode = (os.getenv("DATA_SOURCE_MODE", "auto") or "auto").strip().lower()
    if mode not in {"auto", "bigquery", "csv"}:
        logger.warning("Invalid DATA_SOURCE_MODE '%s'; defaulting to 'auto'", mode)
        return "auto"
    return mode


def _get_city_match_mode() -> str:
    mode = (os.getenv("CITY_MATCH_MODE", "contains") or "contains").strip().lower()
    if mode not in {"contains", "exact", "prefix"}:
        logger.warning("Invalid CITY_MATCH_MODE '%s'; defaulting to 'contains'", mode)
        return "contains"
    return mode


def _cache_ttl_seconds() -> int:
    raw = (os.getenv("CACHE_TTL_SECONDS", "600") or "600").strip()
    try:
        ttl = int(raw)
    except ValueError:
        logger.warning("Invalid CACHE_TTL_SECONDS '%s'; defaulting to 600", raw)
        return 600
    return max(ttl, 0)


def _log_load_event(**kwargs: Any) -> None:
    payload = {"event": "load_city_dataset", **kwargs}
    logger.info("%s", payload)


def _cache_get(city_key: str) -> Optional[pd.DataFrame]:
    ttl = _cache_ttl_seconds()
    if ttl == 0:
        return None

    item = _cache_store.get(city_key)
    if not item:
        return None

    cached_at, df = item
    if time.time() - cached_at > ttl:
        _cache_store.pop(city_key, None)
        return None

    return df.copy()


def _cache_set(city_key: str, df: pd.DataFrame) -> None:
    ttl = _cache_ttl_seconds()
    if ttl == 0:
        return
    _cache_store[city_key] = (time.time(), df.copy())


def _bigquery_config() -> Optional[Tuple[str, str, str]]:
    project = (
        os.getenv("BIGQUERY_PROJECT")
        or os.getenv("BQ_PROJECT")
        or os.getenv("PROJECT_ID")
    )
    dataset = os.getenv("BIGQUERY_DATASET") or os.getenv("BQ_DATASET")
    table = os.getenv("BIGQUERY_TABLE") or os.getenv("BQ_TABLE")
    if not (project and dataset and table):
        return None
    return project, dataset, table


def _is_allowed_identifier(name: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name or ""))


def _query_columns() -> Tuple[str, ...]:
    raw_extra_cols = os.getenv("BIGQUERY_DATE_COLUMNS", "")
    extras = [c.strip() for c in raw_extra_cols.split(",") if c.strip()]
    requested = _DEFAULT_REQUIRED_COLUMNS + extras

    clean = []
    for col in requested:
        if _is_allowed_identifier(col):
            clean.append(col)
        else:
            logger.warning("Skipping unsafe BigQuery column identifier: '%s'", col)

    # preserve order and dedupe
    unique_cols = list(dict.fromkeys(clean))
    if not unique_cols:
        unique_cols = _DEFAULT_REQUIRED_COLUMNS

    return tuple(unique_cols)


def _build_match_condition(city_match_mode: str) -> str:
    if city_match_mode == "exact":
        return "LOWER(REGEXP_REPLACE(COALESCE(Metro, ''), r'[^a-z0-9]', '')) = @city_key"
    if city_match_mode == "prefix":
        return "LOWER(REGEXP_REPLACE(COALESCE(Metro, ''), r'[^a-z0-9]', '')) LIKE CONCAT(@city_key, '%')"
    return "LOWER(REGEXP_REPLACE(COALESCE(Metro, ''), r'[^a-z0-9]', '')) LIKE CONCAT('%', @city_key, '%')"


def _load_from_bigquery(city_key: str) -> pd.DataFrame:
    from google.cloud import bigquery

    config = _bigquery_config()
    if not config:
        return pd.DataFrame()

    project, dataset, table = config
    table_id = f"{project}.{dataset}.{table}"
    city_match_mode = _get_city_match_mode()
    match_condition = _build_match_condition(city_match_mode)
    columns_sql = ", ".join(f"`{col}`" for col in _query_columns())

    query = f"""
        SELECT {columns_sql}
        FROM `{table_id}`
        WHERE {match_condition}
          AND LOWER(RegionType) IN ('city', 'town', 'neighborhood')
    """

    client = bigquery.Client(project=project)
    job_config = bigquery.QueryJobConfig(
        query_parameters=[
            bigquery.ScalarQueryParameter("city_key", "STRING", city_key),
        ]
    )

    try:
        df = client.query(query, job_config=job_config).result().to_dataframe()
    except Exception as exc:  # noqa: BLE001 - surfaced as log message for Streamlit UI
        logger.exception("BigQuery lookup failed for city '%s': %s", city_key, exc)
        return pd.DataFrame()

    return df


def _apply_local_filters(df: pd.DataFrame, city_key: str) -> pd.DataFrame:
    if "Metro" in df.columns:
        df["Metro_norm"] = df["Metro"].fillna("").apply(normalize_key)
        city_match_mode = _get_city_match_mode()
        if city_match_mode == "exact":
            df = df[df["Metro_norm"] == city_key]
        elif city_match_mode == "prefix":
            df = df[df["Metro_norm"].str.startswith(city_key, na=False)]
        else:
            df = df[df["Metro_norm"].str.contains(city_key, na=False)]

    if "RegionType" in df.columns:
        df = df[df["RegionType"].str.lower().isin(_REGION_TYPES)]

    if "RegionName" in df.columns:
        df["RegionName"] = df["RegionName"].fillna("").str.strip()

    return df.reset_index(drop=True)


def _decide_source(mode: str, has_bigquery_config: bool, rag_data_path: Optional[str]) -> DataSourceDecision:
    has_csv = bool(rag_data_path and os.path.exists(rag_data_path))

    if mode == "bigquery":
        if has_bigquery_config:
            return DataSourceDecision(mode=mode, source="bigquery", reason="mode_forced")
        return DataSourceDecision(mode=mode, source="none", reason="missing_bigquery_config")

    if mode == "csv":
        if has_csv:
            return DataSourceDecision(mode=mode, source="csv", reason="mode_forced")
        return DataSourceDecision(mode=mode, source="none", reason="missing_csv_path")

    # mode == auto
    if has_bigquery_config:
        return DataSourceDecision(mode=mode, source="bigquery", reason="bigquery_config_present")
    if has_csv:
        return DataSourceDecision(mode=mode, source="csv", reason="bigquery_config_missing")
    return DataSourceDecision(mode=mode, source="none", reason="no_source_available")


def load_city_dataset(city: str, rag_data_path: Optional[str] = None) -> Optional[pd.DataFrame]:
    city = (city or "").strip()
    if not city:
        _log_load_event(city=city, source="none", reason="empty_city", rows=0, cache_hit=False)
        return None

    city_key = normalize_key(city)
    cached = _cache_get(city_key)
    if cached is not None:
        _log_load_event(city=city, source="cache", reason="ttl_cache_hit", rows=len(cached), cache_hit=True)
        return cached

    mode = _get_data_source_mode()
    decision = _decide_source(mode, has_bigquery_config=bool(_bigquery_config()), rag_data_path=rag_data_path)

    if decision.source == "bigquery":
        df = _load_from_bigquery(city_key)
        if not df.empty:
            df = _apply_local_filters(df.copy(), city_key)
            if "RegionName" in df.columns and not df.empty:
                _cache_set(city_key, df)
                _log_load_event(
                    city=city,
                    source="bigquery",
                    reason=decision.reason,
                    rows=len(df),
                    cache_hit=False,
                    mode=mode,
                )
                return df
        if mode == "bigquery":
            _log_load_event(city=city, source="none", reason="bigquery_no_rows_or_failed", rows=0, cache_hit=False, mode=mode)
            return None

        # auto mode fallback
        if rag_data_path and os.path.exists(rag_data_path):
            csv_df = pd.read_csv(rag_data_path)
            csv_df = _apply_local_filters(csv_df, city_key)
            _cache_set(city_key, csv_df)
            _log_load_event(city=city, source="csv", reason="bigquery_empty_fallback_csv", rows=len(csv_df), cache_hit=False, mode=mode)
            return csv_df

        _log_load_event(city=city, source="none", reason="bigquery_empty_no_csv", rows=0, cache_hit=False, mode=mode)
        return None

    if decision.source == "csv":
        df = pd.read_csv(rag_data_path)
        df = _apply_local_filters(df, city_key)
        _cache_set(city_key, df)
        _log_load_event(city=city, source="csv", reason=decision.reason, rows=len(df), cache_hit=False, mode=mode)
        return df

    _log_load_event(city=city, source="none", reason=decision.reason, rows=0, cache_hit=False, mode=mode)
    return None
