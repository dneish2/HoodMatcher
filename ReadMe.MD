# Neighborhood Matchmaker

This is a Streamlit application that helps users find suitable neighborhoods based on their preferences. The app integrates OpenAI's GPT-4 for text-based recommendations and an image generation API for neighborhood visualizations.

## Features
- AI-powered neighborhood recommendations
- Customizable search with amenities and proximity preferences
- Dynamic image generation for neighborhoods
- Optional RAG-based data filtering for localized insights

## Installation and Deployment

### Prerequisites
- Docker installed on your system
- OpenAI API Key and optional image generation API key

### Running with Docker
1. Clone the repository:
   ```sh
   git clone https://github.com/your-repo/neighborhood-matchmaker.git
   cd neighborhood-matchmaker
   ```
2. Build the Docker image:
   ```sh
   docker build -t neighborhood-matchmaker .
   ```
3. Run the container:
   ```sh
   docker run -p 8501:8501 -e YOUR_OPENAI_API_KEY="your-api-key" -e YOUR_IMAGE_API_KEY="your-image-api-key" neighborhood-matchmaker
   ```
4. Access the app in your browser at `http://localhost:8501`

### Running Locally (Without Docker)
1. Clone the repository and navigate to the project folder.
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Set environment variables for API keys:
   ```sh
   export YOUR_OPENAI_API_KEY="your-api-key"
   export YOUR_IMAGE_API_KEY="your-image-api-key"
   ```
4. Run the Streamlit app:
   ```sh
   streamlit run app.py
   ```

## API Configuration
Ensure that your API keys are set correctly. You can export them in your terminal or set them in a `.env` file.

## Contribution
If you wish to contribute, feel free to fork the repository and submit a pull request.

## License
This project is licensed under the MIT License.

