FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies for MoviePy and image processing
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create output directory for generated content
RUN mkdir -p generated_content/ai_images

# Expose port if running as a service (optional)
# EXPOSE 8002

# Run the content generation agent
CMD ["python", "content_generation_agent.py"]
