FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy application files
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir \
    openai \
    streamlit \
    requests \
    pandas \
    pillow \
    tenacity

# Expose the necessary port for Streamlit
EXPOSE 8501

# Set environment variables
ENV STREAMLIT_SERVER_PORT=8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py"]
