import os
import re
import difflib
import requests
import json
import base64
import pandas as pd
import streamlit as st
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_fixed
from openai import OpenAI  # OpenAI's latest SDK

# For Imagen on Vertex AI
import vertexai
from vertexai.preview.vision_models import ImageGenerationModel

# -------------------------------
# Step 1: Load Environment Variables
# -------------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "imagen")  # Options: "gemini" or "imagen"
PROJECT_ID = os.getenv("PROJECT_ID")  # Required for Imagen
LOCATION = os.getenv("LOCATION", "us-central1")  # Optional; defaults to "us-central1"

client = OpenAI(api_key=OPENAI_API_KEY)
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getcwd(), "application_default_credentials.json")

# -------------------------------
# Step 2: Define Helper Functions
# -------------------------------
def normalize(text: str) -> str:
    """
    Normalize a string by converting to lowercase and removing all non-alphanumeric characters.
    """
    return re.sub(r'[^a-z0-9]', '', text.lower())

def normalize_city(text: str) -> str:
    """
    Custom normalization for city names.
    Converts abbreviations like "st" to "saint" when appropriate.
    """
    text = text.lower().strip()
    text = re.sub(r'[^a-z0-9]', '', text)
    # If the text starts with "st" but not "saint", convert it.
    if text.startswith("st") and not text.startswith("saint"):
        text = "saint" + text[2:]
    return text

def get_valid_neighborhood(lm_output: str, valid_names: list) -> str:
    """
    Use fuzzy matching on normalized names to find the closest valid neighborhood.
    """
    norm_output = normalize(lm_output)
    # Build a mapping: normalized neighborhood -> original neighborhood name.
    norm_mapping = {normalize(name): name for name in valid_names}
    best_match = difflib.get_close_matches(norm_output, norm_mapping.keys(), n=1, cutoff=0.6)
    if best_match:
        return norm_mapping[best_match[0]]
    return lm_output

def get_pricing_details(region_name: str, rag_data: pd.DataFrame) -> dict:
    """
    Retrieve pricing details for a given neighborhood.
    Uses normalized matching with a fallback fuzzy match if needed.
    """
    if rag_data is None:
        return {"error": "CSV data is not loaded."}
    
    norm_target = normalize(region_name)
    def match_region(val):
        return normalize(val) == norm_target
    record = rag_data[rag_data["RegionName"].apply(match_region)]
    
    # If not found, try fuzzy matching.
    if record.empty:
        valid_names = rag_data["RegionName"].dropna().tolist()
        closest = get_valid_neighborhood(region_name, valid_names)
        record = rag_data[rag_data["RegionName"] == closest]
    
    if record.empty:
        return {"error": "Neighborhood not found in the dataset."}
    
    price_columns = ["2024-01-31", "2024-02-29", "2024-03-31", "2025-01-31", "2025-02-28"]
    try:
        prices = record[price_columns].iloc[0].to_dict()
        valid_prices = [price for price in prices.values() if pd.notnull(price)]
        avg_price = sum(valid_prices) / len(valid_prices) if valid_prices else None
    except Exception as e:
        return {"error": f"Error retrieving pricing data: {str(e)}"}
    
    explanation = (
        f"We computed the average price for '{region_name}' by considering the prices from "
        f"{', '.join(price_columns)}. Based on the available data, the average home price is "
        f"${avg_price:,.2f}." if avg_price is not None else "Price data is incomplete."
    )
    
    return {
        "prices": prices,
        "average_price": avg_price,
        "explanation": explanation
    }

# -------------------------------
# Step 3: Define the Core Class
# -------------------------------
class NeighborhoodMatchmaker:
    def __init__(self, city: str, rag_data_path: str = None):
        """
        Initialize the matchmaker by loading and filtering the CSV based on the city.
        Build a cached lookup dictionary of normalized neighborhood names.
        """
        self.city = city.strip()
        self.normalized_city = normalize_city(self.city)
        
        if rag_data_path and os.path.exists(rag_data_path):
            data = pd.read_csv(rag_data_path)
            # Filter rows by matching the normalized Metro column with the user city.
            data = data[data["Metro"].fillna("").apply(lambda x: self.normalized_city in normalize_city(x))]
            valid_types = {"city", "town", "neighborhood"}
            self.rag_data = data[data["RegionType"].str.lower().isin(valid_types)]
            # Cache a mapping: normalized neighborhood -> original neighborhood name.
            self.normalized_lookup = {
                normalize(name): name for name in self.rag_data["RegionName"].dropna().unique()
            }
        else:
            self.rag_data = None
            self.normalized_lookup = {}

        # Minimal system prompt with a few key examples to guide the LLM.
        self.system_prompt = (
            "You are a friendly, knowledgeable local guide for neighborhoods in "
            f"{self.city}. Only suggest neighborhoods that are recognized communities (e.g., Ladue, Clayton, Des Peres, etc.). "
            "Return your answer in JSON format with a key 'recommendations' containing objects with keys 'neighborhood' and 'explanation'. "
            "After the JSON block, add: \"Don't forget to fill out the form at the bottom for more info!\""
        )

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
    def call_openai_api(self, messages: list) -> str:
        """Calls OpenAI's Chat API with a list of messages."""
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages
        )
        return response.choices[0].message.content

    def get_recommendation(self, user_details: str, amenities: list, amenity_proximity: str) -> str:
        """Generates neighborhood recommendations from the LLM."""
        messages = [{"role": "system", "content": self.system_prompt}]
        selected_amenities = ', '.join(amenities) if amenities else "various local amenities"
        user_prompt = (
            f"I'm interested in neighborhoods in {self.city}. {user_details}. "
            f"I'm looking for neighborhoods with these amenities: {selected_amenities}. "
            f"I want to be {amenity_proximity} to these amenities."
        )
        messages.append({"role": "user", "content": user_prompt})
        response = self.call_openai_api(messages)
        return self.validate_output(response)

    def validate_output(self, text: str) -> str:
        """Ensures the output ends with the required call-to-action."""
        required_suffix = "Don't forget to fill out the form at the bottom for more info!"
        if not text.strip().endswith(required_suffix):
            text = text.strip() + "\n\n" + required_suffix
        return text

    def fetch_image(self, additional_details: str, neighborhood: str) -> Image.Image:
        """Fetches an image for a given neighborhood using a text prompt."""
        prompt_detail = f"Beautiful artistic view of {neighborhood} in {self.city}. {additional_details}"
        if IMAGE_MODEL == "imagen":
            return self.fetch_imagen_image(prompt_detail)
        elif IMAGE_MODEL == "gemini":
            return self.fetch_gemini_image(prompt_detail)
        else:
            st.error("Invalid IMAGE_MODEL setting. Please use 'gemini' or 'imagen'.")
            return None

    def fetch_gemini_image(self, prompt: str) -> Image.Image:
        """Fetches an image using the Gemini API."""
        try:
            api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"
            if GEMINI_API_KEY:
                headers = {"Content-Type": "application/json"}
                api_url += f"?key={GEMINI_API_KEY}"
            else:
                st.error("Missing GEMINI_API_KEY.")
                return None

            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"responseModalities": ["Text", "Image"]}
            }
            response = requests.post(api_url, headers=headers, json=payload)
            response_json = response.json()
            if "error" in response_json:
                st.error(f"Gemini API Error: {response_json['error']['message']}")
                return None

            prediction = response_json.get("predictions", [{}])[0]
            image_data = None
            if "candidates" in prediction and prediction["candidates"]:
                candidate = prediction["candidates"][0]
                image_data = candidate.get("bytesBase64Encoded") or candidate.get("data")
            else:
                image_data = prediction.get("bytesBase64Encoded") or prediction.get("data")
            if image_data:
                return Image.open(BytesIO(base64.b64decode(image_data))).convert("RGB")
            else:
                st.info("Image not available from Gemini API.")
                return None
        except Exception as e:
            st.error(f"Gemini image generation failed: {str(e)}")
            return None

    def fetch_imagen_image(self, prompt: str) -> Image.Image:
        """Fetches an image using the Vertex AI Imagen model."""
        try:
            if not PROJECT_ID:
                st.error("Missing PROJECT_ID for Vertex AI.")
                return None
            vertexai.init(project=PROJECT_ID, location=LOCATION)
            generation_model = ImageGenerationModel.from_pretrained("imagen-3.0-generate-002")
            images = generation_model.generate_images(
                prompt=prompt,
                number_of_images=1,
                aspect_ratio="1:1",
                negative_prompt="",
                person_generation="",
                safety_filter_level="",
                add_watermark=True
            )
            pil_image = images[0]._pil_image
            if pil_image.mode != "RGB":
                pil_image = pil_image.convert("RGB")
            return pil_image
        except Exception as e:
            st.error(f"Imagen image generation failed: {str(e)}")
            return None

    def parse_recommendations(self, text: str):
        """Parses the JSON block from the LLM output to extract recommendations."""
        json_str_match = re.search(r'\{.*\}', text, re.DOTALL)
        recommendations = []
        if json_str_match:
            try:
                data = json.loads(json_str_match.group(0))
                recommendations = data.get("recommendations", [])
            except json.JSONDecodeError as e:
                st.error(f"JSON Decode Error: {str(e)}")
        return recommendations

# -------------------------------
# Step 4: Streamlit UI
# -------------------------------
st.set_page_config(layout="wide")
st.title("🏡 Neighborhood Matchmaker")

# User enters the city.
city_input = st.text_input("Enter your city", value="St. Louis")
# CSV file location.
rag_csv = "data/housing-data-slim-2024.csv" if os.path.exists("data/housing-data-slim-2024.csv") else None
# Initialize the matchmaker using the city input and CSV path.
matchmaker = NeighborhoodMatchmaker(city=city_input, rag_data_path=rag_csv)

st.subheader("Your Preferences")
amenities_list = [
    "Good Schools", "Parks", "Shopping Centers", "Public Transport", 
    "Restaurants", "Gyms", "Cafes", "Nightlife", "Hospitals", "Libraries", "Grocery Stores"
]
selected_amenities = st.multiselect("Select amenities", amenities_list)
amenity_proximity = st.selectbox("Proximity to Amenities", [
    "Walking distance (.5-1 miles)",
    "Short drive (2-5 miles)",
    "Far is okay (5+ miles)"
])
user_details = st.text_area("Additional Details", placeholder="e.g., I love tree-lined streets, local parks, and vibrant neighborhoods.")

if st.button('Find Neighborhood'):
    try:
        # Get recommendations from the LLM.
        recommendation_output = matchmaker.get_recommendation(user_details, selected_amenities, amenity_proximity)
        recommendations = matchmaker.parse_recommendations(recommendation_output)
        if recommendations:
            # Build a list of valid neighborhood names from the filtered CSV.
            valid_names = matchmaker.rag_data["RegionName"].dropna().unique().tolist() if matchmaker.rag_data is not None else []
            st.subheader("🏘️ Recommended Neighborhoods")
            for rec in recommendations:
                raw_neighborhood = rec.get("neighborhood") or "Unknown"
                # Use fuzzy matching to map the LLM output to a valid neighborhood.
                neighborhood = get_valid_neighborhood(raw_neighborhood, valid_names)
                explanation = rec.get("explanation", "No details provided.")
                st.markdown(f"### {neighborhood}")
                st.write(explanation)
                
                # Retrieve and display pricing details.
                pricing_details = get_pricing_details(neighborhood, matchmaker.rag_data)
                if "error" not in pricing_details:
                    formatted_price = (
                        "${:,.2f}".format(pricing_details["average_price"])
                        if pricing_details["average_price"] is not None
                        else "N/A"
                    )
                    st.write(f"**Data Transparency:** {pricing_details['explanation']}")
                    st.write(f"**Average Price:** {formatted_price}")
                else:
                    st.info(pricing_details["error"])
                
                # Retrieve and display an image.
                img = matchmaker.fetch_image(user_details, neighborhood)
                if img:
                    st.image(img, caption=f"View of {neighborhood} in {city_input}")
                else:
                    st.info("Image not available for this neighborhood.")
        else:
            st.error("No recommendations found.")
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")