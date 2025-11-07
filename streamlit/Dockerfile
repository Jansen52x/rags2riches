# Start from a slim Python image
FROM python:3.11-slim

# Set the working directory
WORKDIR /app

# Copy *only* the requirements file first
COPY requirements.txt .

# Install all the "UI" dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now, copy the rest of your UI code
# This copies 1_RAG_Agent.py, pages/ folder, etc.
COPY . .

# This is the command that runs when the container starts
# --server.port 8501 matches our docker-compose.yml
# --server.address 0.0.0.0 is MANDATORY.
EXPOSE 8501
CMD ["streamlit", "run", "1_RAG_Agent.py", "--server.port=8501", "--server.address=0.0.0.0"]