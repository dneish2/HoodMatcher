"""Utility module for the Neighborhood Context Map view."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd
import pydeck as pdk
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


@dataclass
class ContextMapOptions:
    """Configuration toggles for the map layer construction."""

    show_labels: bool = True
    show_amenity_ring: bool = True


def get_demo_data() -> pd.DataFrame:
    """Return placeholder neighborhood rows until real data is wired in."""

    return pd.DataFrame(
        [
            {
                "neighborhood_id": "Soulard",
                "lat": 38.6108,
                "lon": -90.2136,
                "incidents_serious_30d": 2,
                "gunshots_30d": 0,
                "amenities_score": 0.78,
                "price_change_pct": 3.1,
                "events_30d": 5,
                "listings_30d": 9,
            },
            {
                "neighborhood_id": "Tower Grove",
                "lat": 38.6073,
                "lon": -90.2427,
                "incidents_serious_30d": 3,
                "gunshots_30d": 1,
                "amenities_score": 0.84,
                "price_change_pct": 2.4,
                "events_30d": 6,
                "listings_30d": 7,
            },
            {
                "neighborhood_id": "The Hill",
                "lat": 38.6176,
                "lon": -90.2729,
                "incidents_serious_30d": 1,
                "gunshots_30d": 0,
                "amenities_score": 0.65,
                "price_change_pct": 1.2,
                "events_30d": 2,
                "listings_30d": 4,
            },
        ]
    )


def load_uploaded(uploaded_file) -> pd.DataFrame:
    """Load an uploaded CSV/JSON to a normalized DataFrame."""

    if uploaded_file is None:
        return get_demo_data()

    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    elif name.endswith(".json"):
        df = pd.read_json(uploaded_file)
    else:
        st.warning("Unsupported file type; using demo data.")
        df = get_demo_data()

    required = [
        "neighborhood_id",
        "lat",
        "lon",
        "incidents_serious_30d",
        "gunshots_30d",
        "amenities_score",
        "price_change_pct",
    ]
    for col in required:
        if col not in df.columns:
            st.error(f"Missing required column: {col}. Using demo data instead.")
            return get_demo_data()

    df = df.dropna(subset=["neighborhood_id", "lat", "lon"]).copy()
    df["events_30d"] = df.get("events_30d", 0).fillna(0).astype(int)
    df["listings_30d"] = df.get("listings_30d", 0).fillna(0).astype(int)

    return df


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
