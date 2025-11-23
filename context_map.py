"""Utility module for the Neighborhood Context Map view."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import pydeck as pdk
import importlib
import importlib.util
import re
import streamlit as st

DEFAULT_WINDOW_DAYS = 30

# Tier thresholds for serious incidents in the time window
TIER_THRESHOLDS = {
    "calm_max": 3,    # ≤ 3  -> green
    "amber_max": 9,   # 4-9  -> amber
    # else             >=10   -> red
}

# Color palette (RGB lists for pydeck)
PALETTE = {
    "green": [46, 204, 113],
    "amber": [243, 156, 18],
    "red":   [231, 76,  60],
    "label": [52, 73,   94],    # text color (dark slate)
    "ring":  [41, 128, 185, 80] # amenities ring (semi-transparent)
}

# Minimal neighborhood → centroid reference to anchor the CompStat upload.
NEIGHBORHOOD_COORDS = {
    "academy": (38.6621, -90.2446),
    "baden": (38.7027, -90.2250),
    "benton park": (38.6006, -90.2109),
    "benton park west": (38.6039, -90.2258),
    "bevo mill": (38.5761, -90.2876),
}

# Bundled CompStat PDF stored in the repo so users can load it without uploading.
DEFAULT_COMPSTAT_PATH = Path("data/compstat_sample.pdf")

# Expected downstream schema for the map
BASE_COLUMNS = [
    "neighborhood_id",
    "lat",
    "lon",
    "incidents_serious_30d",
    "gunshots_30d",
    "amenities_score",
    "price_change_pct",
    "events_30d",
    "listings_30d",
]


@dataclass
class ContextMapOptions:
    """Configuration toggles for the map layer construction."""

    show_labels: bool = True
    show_amenity_ring: bool = True


def _normalize_columns(columns: Iterable[str]) -> dict:
    """Return a mapping from normalized name → original column name."""

    mapping = {}
    for col in columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", col.lower()).strip("_")
        mapping[normalized] = col
    return mapping


def _lookup_first(mapping: dict, candidates: Iterable[str]) -> str | None:
    """Find the first candidate key present in ``mapping`` and return the original column name."""

    for candidate in candidates:
        if candidate in mapping:
            return mapping[candidate]
    return None


def _numeric_from_row(row: pd.Series, col_name: str) -> int:
    """Safely pull an integer-like value from a row."""

    try:
        return int(pd.to_numeric(row.get(col_name, 0), errors="coerce") or 0)
    except Exception:
        return 0


def _extract_text_from_pdf(file_obj) -> str:
    """Extract text from a PDF using ``pdfplumber`` when available.

    Falls back to a naive stream parser for uncompressed text-only PDFs so the
    bundled sample still works even if the dependency is missing.
    """

    file_obj.seek(0)
    spec = importlib.util.find_spec("pdfplumber")
    if spec:
        pdfplumber = importlib.import_module("pdfplumber")
        with pdfplumber.open(file_obj) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)

    raw = file_obj.read()
    segments = re.findall(rb"BT(.*?)ET", raw, flags=re.DOTALL)
    text_segments = []
    for seg in segments:
        try:
            text_segments.append(seg.decode("latin-1", errors="ignore"))
        except Exception:
            continue
    return "\n".join(text_segments)


def _extract_metric_from_block(lines: list[str], label: str) -> int | None:
    """Find ``label`` in ``lines`` and return the 28-day 2025 value (3rd number)."""

    for line in lines:
        if not line.lower().startswith(label.lower()):
            continue
        sanitized = re.sub(r"-?\d+%", "", line)
        numbers = [int(n) for n in re.findall(r"-?\d+", sanitized)]
        if len(numbers) >= 3:
            return numbers[2]
    return None


def _parse_compstat_text(text: str) -> pd.DataFrame:
    """Parse text from a CompStat PDF into the normalized map schema."""

    if not text:
        return pd.DataFrame(columns=BASE_COLUMNS)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    neighborhood_keys = set(NEIGHBORHOOD_COORDS.keys())

    rows = []
    idx = 0
    while idx < len(lines):
        current = lines[idx]
        normalized = current.lower()
        if normalized in neighborhood_keys:
            block = []
            idx += 1
            while idx < len(lines) and lines[idx].lower() not in neighborhood_keys:
                block.append(lines[idx])
                idx += 1

            total_28 = _extract_metric_from_block(block, "TOTAL")
            shootings_28 = _extract_metric_from_block(block, "SHOOTING INCIDENTS")

            base = {
                "neighborhood_id": current,
                "incidents_serious_30d": total_28 or 0,
                "gunshots_30d": shootings_28 or 0,
                "amenities_score": 0.0,
                "price_change_pct": 0.0,
                "events_30d": 0,
                "listings_30d": 0,
            }
            rows.append(base)
            continue
        idx += 1

    df = pd.DataFrame(rows)
    df = _fill_coordinates(df)
    df = df.dropna(subset=["lat", "lon", "neighborhood_id"]).copy()
    for col in ["lat", "lon"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["lat", "lon"]).copy()
    return df.reindex(columns=BASE_COLUMNS)


def _load_compstat_pdf(file_obj) -> pd.DataFrame:
    """Read a CompStat PDF into the normalized map schema."""

    text = _extract_text_from_pdf(file_obj)
    df = _parse_compstat_text(text)
    if df.empty:
        st.warning("No recognizable CompStat rows were found in the PDF.")
    return df


def _fill_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ``lat``/``lon`` exist, using reference centroids when missing."""

    df = df.copy()
    has_lat_lon = {"lat", "lon"}.issubset({c.lower() for c in df.columns})
    if has_lat_lon:
        df = df.rename(columns={"Lat": "lat", "Lon": "lon", "LAT": "lat", "LON": "lon"})
        return df

    # derive coordinates from the lookup table
    lats = []
    lons = []
    for _, row in df.iterrows():
        key = str(row.get("neighborhood_id", "")).strip().lower()
        lat, lon = NEIGHBORHOOD_COORDS.get(key, (None, None))
        lats.append(lat)
        lons.append(lon)

    df["lat"] = lats
    df["lon"] = lons
    return df


def _normalize_compstat_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Transform a CompStat-style CSV into the columns expected by the map."""

    if df.empty:
        return pd.DataFrame(columns=BASE_COLUMNS)

    colmap = _normalize_columns(df.columns)
    neighborhood_col = _lookup_first(colmap, ["neighborhood", "neighborhood_id", "area"])
    if neighborhood_col is None:
        st.error("Upload must include a neighborhood column.")
        return pd.DataFrame(columns=BASE_COLUMNS)

    total_col = _lookup_first(
        colmap,
        [
            "total_last_28_days",
            "last_28_days_total",
            "total_28_days",
            "last_28_days",
            "total",
        ],
    )
    shooting_col = _lookup_first(
        colmap,
        [
            "shooting_incidents_last_28_days",
            "last_28_days_shooting_incidents",
            "shooting_incidents",
        ],
    )

    violent_candidates = [
        "murder_last_28_days",
        "sexual_assault_last_28_days",
        "robbery_last_28_days",
        "aggravated_assault_last_28_days",
        "burglary_last_28_days",
        "felony_theft_last_28_days",
        "auto_theft_last_28_days",
    ]

    rows = []
    for _, row in df.iterrows():
        base = {
            "neighborhood_id": str(row[neighborhood_col]).strip(),
            "amenities_score": 0.0,
            "price_change_pct": 0.0,
            "events_30d": 0,
            "listings_30d": 0,
        }

        total_val = _numeric_from_row(row, total_col) if total_col else 0

        if total_val == 0:
            total_val = 0
            for candidate in violent_candidates:
                col_name = _lookup_first(colmap, [candidate])
                if col_name:
                    total_val += _numeric_from_row(row, col_name)

        base["incidents_serious_30d"] = total_val

        if shooting_col:
            base["gunshots_30d"] = _numeric_from_row(row, shooting_col)
        else:
            base["gunshots_30d"] = 0

        rows.append(base)

    normalized = pd.DataFrame(rows)
    normalized = _fill_coordinates(normalized)
    normalized = normalized.dropna(subset=["lat", "lon", "neighborhood_id"])

    for col in ["lat", "lon"]:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")

    normalized = normalized.dropna(subset=["lat", "lon"]).copy()
    return normalized.reindex(columns=BASE_COLUMNS)


def load_uploaded(uploaded_file) -> pd.DataFrame:
    """Load a PDF/CSV/JSON into the normalized DataFrame used by the map."""

    if uploaded_file is None:
        return pd.DataFrame(columns=BASE_COLUMNS)

    close_after = False
    file_obj = uploaded_file
    name = getattr(uploaded_file, "name", "")

    if isinstance(uploaded_file, (str, Path)):
        path = Path(uploaded_file)
        name = path.name
        file_obj = path.open("rb")
        close_after = True

    name = (name or "").lower()

    try:
        if name.endswith(".pdf"):
            normalized = _load_compstat_pdf(file_obj)
        elif name.endswith(".csv"):
            normalized = _normalize_compstat_schema(pd.read_csv(file_obj))
        elif name.endswith(".json"):
            normalized = _normalize_compstat_schema(pd.read_json(file_obj))
        else:
            st.warning("Unsupported file type; please upload PDF, CSV, or JSON.")
            return pd.DataFrame(columns=BASE_COLUMNS)
    finally:
        if close_after:
            file_obj.close()

    if normalized.empty:
        st.warning("No valid neighborhood rows were found in the upload.")
    return normalized


def load_bundled_compstat() -> pd.DataFrame:
    """Load the baked-in CompStat PDF from the repository if present."""

    if DEFAULT_COMPSTAT_PATH.exists():
        return load_uploaded(DEFAULT_COMPSTAT_PATH)

    st.warning("Bundled CompStat PDF is missing from the data directory.")
    return pd.DataFrame(columns=BASE_COLUMNS)


def color_for_incidents(incidents: int) -> List[int]:
    """Map incident count to a tier color."""

    if incidents <= TIER_THRESHOLDS["calm_max"]:
        return PALETTE["green"]
    if incidents <= TIER_THRESHOLDS["amber_max"]:
        return PALETTE["amber"]
    return PALETTE["red"]


def radius_for_amenities(score: float) -> float:
    """Visual radius for the amenities ring: simple, monotonic scale."""

    score = 0 if np.isnan(score) else max(0.0, min(1.0, score))
    return 120 + 280 * score  # 120..400 meters visual ring


def prepare_map_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with derived columns for the map view."""

    if df.empty:
        return df.copy()

    prepared = df.copy()
    prepared["tier_color"] = prepared["incidents_serious_30d"].apply(color_for_incidents)
    return prepared


def build_layers(df: pd.DataFrame, options: ContextMapOptions) -> List[pdk.Layer]:
    """Build pydeck layers using only built-in primitives."""

    layers: List[pdk.Layer] = []

    dot_layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position='[lon, lat]',
        get_radius=160,
        get_fill_color="tier_color",
        pickable=True,
    )
    layers.append(dot_layer)

    if options.show_amenity_ring:
        ring_df = df.copy()
        ring_df["ring_radius"] = ring_df["amenities_score"].apply(radius_for_amenities)
        ring_layer = pdk.Layer(
            "ScatterplotLayer",
            data=ring_df,
            get_position='[lon, lat]',
            get_radius="ring_radius",
            stroked=True,
            filled=False,
            get_line_color=PALETTE["ring"],
            get_line_width=2,
            pickable=False,
        )
        layers.append(ring_layer)

    if options.show_labels:
        text_layer = pdk.Layer(
            "TextLayer",
            data=df,
            get_position='[lon, lat]',
            get_text="neighborhood_id",
            get_color=PALETTE["label"],
            get_size=14,
            get_alignment_baseline="'bottom'",
        )
        layers.append(text_layer)

    return layers


def summarize_row(row: pd.Series, tone: str = "Factual", window_days: int = DEFAULT_WINDOW_DAYS) -> str:
    """Deterministic one-liner summaries with tone variants."""

    name = row["neighborhood_id"]
    inc = int(row["incidents_serious_30d"])
    gun = int(row["gunshots_30d"])
    amen = float(row["amenities_score"])
    pct = float(row["price_change_pct"])
    ev = int(row.get("events_30d", 0))
    li = int(row.get("listings_30d", 0))

    if tone.lower() == "empathetic":
        return (
            f"{name} stays grounded with {inc} serious incidents ({gun} gunshots) in {window_days}d; "
            f"amenities {amen:.2f} and {pct:+.1f}% price trend suggest steady, livable momentum."
        )
    if tone.lower() == "energetic":
        return (
            f"{name} pops with amenities {amen:.2f} and {pct:+.1f}% price momentum, plus {ev} events and {li} listings; "
            f"{inc} serious incidents ({gun} gunshots) in {window_days}d."
        )
    return (
        f"{name}: {inc} serious incidents ({gun} gunshots) in {window_days}d, "
        f"amenities {amen:.2f}, price trend {pct:+.1f}%, {ev} events, {li} listings."
    )


def select_recommendations(df: pd.DataFrame, limit: int = 3) -> pd.DataFrame:
    """Return the top ``limit`` rows using a simple ranking heuristic."""

    if df.empty:
        return df

    sorted_df = df.sort_values(
        by=["incidents_serious_30d", "gunshots_30d", "price_change_pct"],
        ascending=[True, True, False],
    )
    return sorted_df.head(limit)


def build_tooltip_fields(fields: Iterable[str]) -> str:
    """Construct a multiline tooltip body from the provided ``fields``."""

    lines = []
    for field in fields:
        label = field.replace("_", " ")
        lines.append(f"{label.title()}: {{{field}}}")
    return "\n".join(lines)


def compute_initial_view(df: pd.DataFrame) -> pdk.ViewState:
    """Return a view centred on the dataset, or a neutral fallback if empty."""

    if df.empty:
        return pdk.ViewState(latitude=0.0, longitude=0.0, zoom=1.5, pitch=0)

    mean_lat = float(df["lat"].mean())
    mean_lon = float(df["lon"].mean())
    if not np.isfinite(mean_lat) or not np.isfinite(mean_lon):
        return pdk.ViewState(latitude=0.0, longitude=0.0, zoom=1.5, pitch=0)

    return pdk.ViewState(latitude=mean_lat, longitude=mean_lon, zoom=11.0, pitch=0)


def build_deck(df: pd.DataFrame, options: ContextMapOptions) -> pdk.Deck:
    """Convenience helper to assemble a :class:`pydeck.Deck` instance."""

    layers = build_layers(df, options)
    tooltip_text = build_tooltip_fields(
        [
            "neighborhood_id",
            "incidents_serious_30d",
            "gunshots_30d",
            "amenities_score",
            "price_change_pct",
            "events_30d",
            "listings_30d",
        ]
    )
    tooltip = {"text": tooltip_text}
    return pdk.Deck(layers=layers, initial_view_state=compute_initial_view(df), tooltip=tooltip)
