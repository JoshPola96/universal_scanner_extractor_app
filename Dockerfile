# --- Stage 1: Builder ---
# Use a standard Python image to build dependencies.
FROM python:3.10 AS builder

# Set the working directory
WORKDIR /install

# Copy requirements and install Python dependencies.
# We don't use --target, so pip installs to its default locations,
# including the bin directory for executables.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- Stage 2: Final Image ---
# Use the correct slim Python image for the final runtime.
FROM python:3.10-slim

# Install system dependencies needed at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libfontconfig1 \
    libxrender1 \
    libgomp1 \
    wget \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory for the application
WORKDIR /app

# Copy the installed Python dependencies from the 'builder' stage.
# This includes the libraries and the 'streamlit' executable.
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy the application code
COPY . .

# --- NEW STEP: Pre-download PaddleOCR models for caching ---
# We use a simple python command to initialize the models.
# This downloads them once during the build, so they are available at runtime.
RUN python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='en')"

# Create the .streamlit directory and config, with the address value correctly quoted
RUN mkdir -p .streamlit && \
    echo "[server]" > .streamlit/config.toml && \
    echo "headless = true" >> .streamlit/config.toml && \
    echo "port = 8501" >> .streamlit/config.toml && \
    echo "enableCORS = false" >> .streamlit/config.toml && \
    echo "enableXsrfProtection = false" >> .streamlit/config.toml

# Expose the port Streamlit runs on
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py"]


