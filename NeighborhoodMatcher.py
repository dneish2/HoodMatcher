import os
import re
import json
import ast
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

# -------------------------------
# Load Environment Variables
# -------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "imagen")
PROJECT_ID = os.getenv("PROJECT_ID")
LOCATION = os.getenv("LOCATION", "us-central1")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), "application_default_credentials.json")
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
city = st.text_input("City", value="St. Louis")
csv_path = "data/housing-data-slim-2024.csv" if os.path.exists("data/housing-data-slim-2024.csv") else None
nm = NeighborhoodMatchmaker(city, csv_path)

amenities = st.multiselect("Amenities", ["Schools","Parks","Shopping","Transit","Restaurants","Libraries","Farmers Markets","Community Centers","Hospitals","Gyms","Cafes","Art Galleries","Theaters"])
prox = st.selectbox("Proximity", ["Walking","Short drive","Far"])
details = st.text_area("Details", placeholder="e.g., tree-lined streets.")

if st.button("Find Neighborhood"):
    recs = nm.get_recommendation(details, amenities, prox)
    if not recs:
        st.error("RAG Data for neighborhoods could not be found. View Raw LLM Output for details.")
    else:
        for r in recs:
            nm_raw = r["neighborhood"]
            nm_name = nm.match_with_faiss(nm_raw)

            st.header(nm_name)
            st.write(r.get("explanation",""))

            # Create two tabs: Chart vs. Artistic
            tab_chart, tab_art = st.tabs(["Chart", "Artistic"])

            # Tab 1: RAG chart + caption
            with tab_chart:
                ts = get_pricing_timeseries(nm_name, nm.rag_data)
                if "error" in ts:
                    st.error(ts["error"])
                else:
                    st.caption("Data pulled via RAG from Zillow dataset")
                    df = pd.DataFrame(list(ts.items()), columns=["Date","Price"])
                    df["Date"] = pd.to_datetime(df["Date"])
                    fig, ax = plt.subplots(figsize=(6,3))
                    ax.plot(df["Date"], df["Price"], marker="o", markersize=4, linewidth=1)
                    ax.yaxis.set_major_formatter(
                        FuncFormatter(lambda v,pos: f"${v:,.0f}")
                    )
                    ax.set_title("Historic Avg Home Prices", fontsize=10, pad=8)
                    ax.set_xlabel("Date", fontsize=8)
                    ax.set_ylabel("Price (USD)", fontsize=8)
                    ax.tick_params(axis="x", labelrotation=45, labelsize=7)
                    ax.tick_params(axis="y", labelsize=7)
                    fig.tight_layout(pad=2)
                    st.pyplot(fig, clear_figure=True)

            # Tab 2: placeholder image
            with tab_art:
                # placeholder = Image.new("RGB", (400,300), color=(200,200,200))
                # st.image(placeholder, caption=f"Artistic view of {nm_name}", use_container_width=False)
                # # later swap in:
                img = nm.fetch_image(details, nm_name)
                if img: st.image(img, caption=f"{nm_name}, {city}")