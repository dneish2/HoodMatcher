# Neighborhood Context Map Integration Plan

## Goals
- Surface a production-ready "Neighborhood Context Map" experience next to the existing LLM-powered matchmaker.
- Keep the map self-contained with placeholder data so real feeds can be swapped in later without touching the UI contract.
- Ensure the wider application continues to compile/run cleanly while introducing the new feature toggles and dependencies.

## Key Components
1. **`context_map.py` module**
   - Owns placeholder data loading (`get_demo_data`, `load_uploaded`).
   - Produces derived columns for tier colours (`prepare_map_dataframe`).
   - Encapsulates pydeck layer creation (`build_layers`, `build_deck`) and tooltips.
   - Provides deterministic card summaries so the UI can render without an LLM.

2. **Streamlit tab wiring (`NeighborhoodMatcher.py`)**
   - Adds a second tab (`"Neighborhood Context Map"`) alongside the matchmaker.
   - Exposes file upload + map display options (`labels`, `amenity ring`, `tone`).
   - Re-uses the deterministic summaries to render the recommendation cards below the map.

3. **Dependencies**
   - `pydeck` is added to `requirements.txt` so the map renders in production.

## Data Flow
1. User uploads CSV/JSON (or defaults to demo data).
2. `load_uploaded` returns a cleaned DataFrame with optional fields normalised.
3. `prepare_map_dataframe` derives the `tier_color` column.
4. `build_deck` assembles pydeck layers respecting `ContextMapOptions`.
5. Streamlit renders the map + legend, then cards summarise the top three neighbourhoods using the placeholder heuristic.

## Future Hooks
- Swap `get_demo_data` for a real warehouse/ETL feed that preserves the existing schema.
- Replace `summarize_context_row` with an LLM call while keeping the same API contract.
- Bind card selections to zoom/pan events (pydeck viewport updates) once two-way communication is needed.
- Extend `ContextMapOptions` with toggles for additional layers (events, historical incidents, etc.).

## Bug-Proofing Notes
- `load_uploaded` validates required columns and falls back gracefully to demo data.
- `compute_initial_view` guards against empty/invalid coordinates to avoid pydeck crashes.
- Recommendation section handles empty datasets so the UI does not raise when uploads are blank.

With these structures in place the feature can accept real data feeds later while keeping the UI and summarisation pipeline stable.
