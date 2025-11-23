import os
import re
import json
import ast
import logging
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI
import numpy as np
import faiss
from PIL import Image
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel
from io import BytesIO
from matplotlib.ticker import FuncFormatter
from google.cloud import storage
from google.api_core.exceptions import NotFound, PreconditionFailed

from context_map import (
    ContextMapOptions,
    DEFAULT_WINDOW_DAYS,
    build_deck,
    load_bundled_compstat,
    load_uploaded as load_context_dataset,
    prepare_map_dataframe,
    select_recommendations,
    summarize_row as summarize_context_row,
)

# -------------------------------
# Load Environment Variables
# -------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "imagen")
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "us-central1")
EMAIL_BUCKET = os.getenv("EMAIL_BUCKET")

# Leave GOOGLE_APPLICATION_CREDENTIALS alone when it's already configured;
# otherwise rely on the local default file so Cloud Run can keep using its
# service account when no key is mounted.
if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
    default_creds_path = os.path.join(os.getcwd(), "application_default_credentials.json")
    if os.path.exists(default_creds_path):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = default_creds_path
client = OpenAI(api_key=OPENAI_API_KEY)

# -------------------------------
# Helper: pricing timeseries
# -------------------------------
def normalize_key(text: str) -> str:
    """Lowercase, strip non-alphanumerics so 'St. Louis' → 'stlouis'."""
    return re.sub(r'[^a-z0-9]', '', text.lower() or "")

def get_pricing_timeseries(region_name: str, rag_data: pd.DataFrame) -> dict:
    if rag_data is None:
        return {"error": "Dataset not loaded."}

    # Strict lookup—nm_name came from FAISS so it should match exactly
    rec = rag_data[rag_data["RegionName"] == region_name]
    if rec.empty:
        return {"error": f"No data for '{region_name}'."}

    # Pull only the known date columns
    dates = [c for c in
             ["2024-01-31","2024-02-29","2024-03-31","2025-01-31","2025-02-28"]
             if c in rec.columns]
    row = rec.iloc[0]
    return {d: row[d] for d in dates if pd.notnull(row[d])}

# -------------------------------
# Core: NeighborhoodMatchmaker
# -------------------------------
class NeighborhoodMatchmaker:
    def __init__(self, city: str, rag_data_path: str = None):
        self.city = city.strip()
        # load & filter RAG CSV
        if rag_data_path and os.path.exists(rag_data_path):
            df = pd.read_csv(rag_data_path)

            # Normalize City vs. Metro for robust matching
            norm_city = normalize_key(self.city)
            df["Metro_norm"] = (
                df["Metro"]
                .fillna("")
                .apply(normalize_key)
            )

            df = df[df["Metro_norm"].str.contains(norm_city, na=False)]

            # Also strip whitespace from RegionName for downstream lookups
            df["RegionName"] = df["RegionName"].fillna("").str.strip()

            types = {"city", "town", "neighborhood"}
            self.rag_data = (
                df[df["RegionType"].str.lower().isin(types)]
                .reset_index(drop=True)
            )
        else:
            self.rag_data = None
        # build FAISS index
        if self.rag_data is not None and not self.rag_data.empty:
            names = self.rag_data["RegionName"].dropna().tolist()
            response = client.embeddings.create(input=names, model="text-embedding-3-small")
            embeds = [d.embedding for d in response.data]
            vecs = np.array(embeds, dtype='float32')
            self.faiss_index = faiss.IndexFlatL2(vecs.shape[1])
            self.faiss_index.add(vecs)
            self.region_names = names
        else:
            self.faiss_index = None
            self.region_names = []

        # system prompt

        self.system_prompt = (
            f"You are a local real-estate guide for {self.city}, writing from the perspective of "
            "a home-buyer exploring new neighborhoods.  \n"
            "Only choose from the provided list.  \n"
            "Output JSON with a single top-level key 'recommendations', whose value is an array of objects "
            "each containing:\n"
            "  • 'neighborhood': the exact name  \n"
            "  • 'explanation': a narrative of at least three sentences that:\n"
            "      1. Describes why this area fits the user’s criteria  \n"
            "      2. For each amenity (e.g., library, farmers market), gives a specific location or address "
            "(landmark, street intersection, etc.) so someone could look it up  \n"
            "Write in a friendly, informative tone as if guiding a walking tour."
        )
       
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def call_llm(self, messages: list) -> str:
        resp = client.chat.completions.create(model="gpt-4", messages=messages)
        return resp.choices[0].message.content

    def get_recommendation(self, details: str, amenities: list, proximity: str) -> list:
        # 1) start with the main instruction
        msgs = [{"role": "system", "content": self.system_prompt}]

        # 2) provide the full list of valid neighborhoods
        if self.region_names:
            list_content = "Available neighborhoods: " + ", ".join(self.region_names)
            msgs.append({"role": "system", "content": list_content})

        # 3) user’s query
        amen_str = ", ".join(amenities) if amenities else "amenities"
        user = (
            f"I want neighborhoods in {self.city}. "
            f"{details} "
            f"Amenities: {amen_str}. "
            f"Proximity: {proximity}."
        )
        msgs.append({"role": "user", "content": user})

        # 4) call and parse logs
        out = self.call_llm(msgs)
    
        with st.expander("💬 Raw LLM output", expanded=False):
            st.code(out, language="json")

        match = re.search(r"\{.*\}", out, re.DOTALL)
        if not match:
            return []
        json_str = match.group(0)
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError:
            try:
                data = ast.literal_eval(json_str)
            except (ValueError, SyntaxError):
                return []
        return data.get("recommendations", [])

    def match_with_faiss(self, name: str) -> str:
        if not self.faiss_index:
            return name
        response = client.embeddings.create(input=[name], model="text-embedding-3-small")
        q = response.data[0].embedding
        D, I = self.faiss_index.search(np.array([q], dtype='float32'), 1)
        return self.region_names[int(I[0][0])]
     
    def imagen(self, prompt: str) -> Image.Image:
        try:
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")

            # log the outgoing prompt for debugging
            st.write(f"🎨 Generating image with prompt: {prompt!r}")

            images = model.generate_images(prompt=prompt, number_of_images=1)

            if not images:
                st.warning("⚠️ No GeneratedImage objects returned.")
                return None

            # use the internal bytes buffer
            raw = images[0]._image_bytes
            if not raw:
                st.warning("⚠️ Imagen API returned empty bytes—likely blocked or filtered.")
                return None

            # show size for transparency
            st.write(f"📦 Received {len(raw)} bytes from Imagen API")

            return Image.open(BytesIO(raw))

        except Exception as e:
            # log the full exception for troubleshooting
            st.error(f"❌ Imagen image generation failed: {e!s}")
            return None

    def fetch_image(self, prompt: str, neighborhood: str) -> Image.Image:
        text = f"Artistic view of {neighborhood} in {self.city}. {prompt}"
        return self.imagen(text)

# -------------------------------
# Streamlit App
# -------------------------------
st.set_page_config(layout='wide')
st.title("🏡 Neighborhood Matchmaker")

tab_matchmaker, tab_context_map = st.tabs([
    "Neighborhood Matchmaker",
    "Neighborhood Context Map",
])

with tab_matchmaker:
    city = st.text_input("City", value="St. Louis")
    csv_path = (
        "data/housing-data-slim-2024.csv"
        if os.path.exists("data/housing-data-slim-2024.csv")
        else None
    )
    nm = NeighborhoodMatchmaker(city, csv_path)

    amenities = st.multiselect(
        "Amenities",
        [
            "Schools",
            "Parks",
            "Shopping",
            "Transit",
            "Restaurants",
            "Libraries",
            "Farmers Markets",
            "Community Centers",
            "Hospitals",
            "Gyms",
            "Cafes",
            "Art Galleries",
            "Theaters",
        ],
    )
    prox = st.selectbox("Proximity", ["Walking", "Short drive", "Far"])
    details = st.text_area("Details", placeholder="e.g., tree-lined streets.")

    if st.button("Find Neighborhood", key="run_matchmaker"):
        recs = nm.get_recommendation(details, amenities, prox)
        if not recs:
            st.error(
                "RAG Data for neighborhoods could not be found. View Raw LLM Output for details."
            )
        else:
            for r in recs:
                nm_raw = r["neighborhood"]
                nm_name = nm.match_with_faiss(nm_raw)

                st.header(nm_name)
                st.write(r.get("explanation", ""))

                tab_chart, tab_art = st.tabs(["Chart", "Artistic"])

                with tab_chart:
                    ts = get_pricing_timeseries(nm_name, nm.rag_data)
                    if "error" in ts:
                        st.error(ts["error"])
                    else:
                        st.caption("Data pulled via RAG from Zillow dataset")
                        df = pd.DataFrame(list(ts.items()), columns=["Date", "Price"])
                        df["Date"] = pd.to_datetime(df["Date"])
                        fig, ax = plt.subplots(figsize=(6, 3))
                        ax.plot(
                            df["Date"],
                            df["Price"],
                            marker="o",
                            markersize=4,
                            linewidth=1,
                        )
                        ax.yaxis.set_major_formatter(
                            FuncFormatter(lambda v, pos: f"${v:,.0f}")
                        )
                        ax.set_title("Historic Avg Home Prices", fontsize=10, pad=8)
                        ax.set_xlabel("Date", fontsize=8)
                        ax.set_ylabel("Price (USD)", fontsize=8)
                        ax.tick_params(axis="x", labelrotation=45, labelsize=7)
                        ax.tick_params(axis="y", labelsize=7)
                        fig.tight_layout(pad=2)
                        st.pyplot(fig, clear_figure=True)

                with tab_art:
                    img = nm.fetch_image(details, nm_name)
                    if img:
                        st.image(img, caption=f"{nm_name}, {city}")

            st.divider()
            with st.form("email_capture"):
                st.markdown("### Get neighborhood updates")
                email_input = st.text_input(
                    "Email address", placeholder="you@example.com"
                )
                opt_in = st.checkbox(
                    "I agree to receive follow-up emails about neighborhoods."
                )
                submit_form = st.form_submit_button("Notify me")

            if submit_form:
                email_pattern = r"^[^@\s]+@[^@\s]+\.[^@\s]+$"
                if not re.match(email_pattern, email_input or ""):
                    st.error("Please enter a valid email address before submitting.")
                elif not opt_in:
                    st.warning(
                        "Please check the opt-in box if you'd like us to reach out."
                    )
                elif not EMAIL_BUCKET:
                    err = "Email capture bucket is not configured."
                    st.error(err)
                    logging.error(err)
                else:
                    try:
                        storage_client = storage.Client()
                        bucket = storage_client.bucket(EMAIL_BUCKET)
                        blob_path = (
                            f"emails/{datetime.utcnow().strftime('%Y-%m-%d')}.jsonl"
                        )
                        blob = bucket.blob(blob_path)

                        payload = {
                            "email": email_input.strip(),
                            "opt_in": opt_in,
                            "timestamp": datetime.utcnow().isoformat() + "Z",
                            "city": city,
                            "details": details,
                            "amenities": amenities,
                            "proximity": prox,
                            "recommendations": recs,
                        }
                        line = json.dumps(payload, ensure_ascii=False) + "\n"

                        try:
                            blob.reload()
                            existing = blob.download_as_text()
                            generation = blob.generation
                            content = (existing or "") + line
                        except NotFound:
                            generation = 0
                            content = line

                        try:
                            blob.upload_from_string(
                                content, if_generation_match=generation
                            )
                            st.success("Thanks! We'll send updates to your inbox soon.")
                        except PreconditionFailed:
                            st.error(
                                "We couldn't save your request right now. Please try again."
                            )
                            logging.exception(
                                "GCS generation precondition failed while saving email capture"
                            )
                    except Exception:
                        st.error(
                            "Something went wrong while saving your email. Please try again later."
                        )
                        logging.exception(
                            "Unexpected failure while saving email capture"
                        )

with tab_context_map:
    st.subheader("Neighborhood Context Map")
    st.caption(
        "Load the bundled CompStat PDF or upload a new report to plot crime trends on the map."
    )

    upload_col, options_col = st.columns([3, 2])
    with upload_col:
        data_source = st.radio(
            "Choose a CompStat dataset",
            [
                "Bundled sample (11/3–11/9/2025)",
                "Upload a PDF/CSV/JSON",
            ],
            index=0,
            key="context_map_source",
        )
        uploaded_context = None
        if data_source.startswith("Bundled"):
            st.caption(
                "Using the bundled CompStat PDF stored at data/compstat_sample.pdf."
            )
        else:
            uploaded_context = st.file_uploader(
                "Upload neighborhoods PDF/CSV/JSON",
                type=["pdf", "csv", "json"],
                key="context_map_uploader",
            )
    with options_col:
        show_labels = st.checkbox(
            "Show neighborhood labels", value=True, key="context_map_labels"
        )
        show_amenity_ring = st.checkbox(
            "Show amenities ring", value=True, key="context_map_ring"
        )
        tone = st.selectbox(
            "Summary tone",
            ["Factual", "Empathetic", "Energetic"],
            index=0,
            key="context_map_tone",
        )

    if data_source.startswith("Bundled"):
        map_df = load_bundled_compstat()
    else:
        map_df = load_context_dataset(uploaded_context)
    prepared_map_df = prepare_map_dataframe(map_df)
    map_options = ContextMapOptions(
        show_labels=show_labels, show_amenity_ring=show_amenity_ring
    )

    if prepared_map_df.empty:
        st.info(
            "The map stays blank until you load the bundled CompStat PDF or upload your own report."
        )
    deck = build_deck(prepared_map_df, map_options)
    st.pydeck_chart(deck)

    st.markdown(
        """
**Legend**  
- <span style="color:#2ecc71">Green</span>: ≤3 serious incidents  
- <span style="color:#f39c12">Amber</span>: 4–9 serious incidents  
- <span style="color:#e74c3c">Red</span>: ≥10 serious incidents  
<small>Amenities ring scales with amenities score; gunshot reports called out when present.</small>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.subheader("Recommended neighborhoods")

    recs_df = select_recommendations(prepared_map_df, limit=3)
    if recs_df.empty:
        st.info("No neighborhoods available to recommend.")
    else:
        num_cards = min(3, len(recs_df))
        columns = st.columns(num_cards)
        for column, (_, rec_row) in zip(columns, recs_df.iterrows()):
            color = "#{:02x}{:02x}{:02x}".format(*rec_row["tier_color"])
            column.markdown(f"### {rec_row['neighborhood_id']}")
            column.markdown(
                f"""
**Snapshot**  
- <span style="color:{color}">Serious incidents (30d): {int(rec_row['incidents_serious_30d'])}</span>  
- Gunshots (30d): {int(rec_row['gunshots_30d'])}  
- Amenities score: {rec_row['amenities_score']:.2f}  
- Price trend: {rec_row['price_change_pct']:+.1f}%  
- Events: {int(rec_row.get('events_30d', 0))} • Listings: {int(rec_row.get('listings_30d', 0))}
                """,
                unsafe_allow_html=True,
            )
            summary = summarize_context_row(
                rec_row, tone=tone, window_days=DEFAULT_WINDOW_DAYS
            )
            column.markdown(f"**Summary ({tone}):** {summary}")

    st.caption(
        "Future: bind card click to update map view; plug real amenities/listings/events layers; swap the deterministic summaries with an LLM formatter."
    )
