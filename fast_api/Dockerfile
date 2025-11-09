# Start from a slim Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies for MoviePy and image processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy *only* the requirements file first for Docker caching
COPY requirements.txt .

# Install all the "Kitchen" dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now, copy the rest of your application code
COPY . .

# Create output directory for generated content
RUN mkdir -p /app/generated_content/ai_images

# This is the command that runs when the container starts
# It tells uvicorn to run the 'app' object from the 'app.main' file
# --host 0.0.0.0 is MANDATORY. It tells it to listen to requests
# from *outside* the container (i.e., from the Streamlit app).
# --reload enables hot-reloading when code changes (development mode)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001", "--reload"]
