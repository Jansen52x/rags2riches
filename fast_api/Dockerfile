# Start from a slim Python image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Copy *only* the requirements file first for Docker caching
COPY requirements.txt .

# Install all the "Kitchen" dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Now, copy the rest of your application code
COPY . .

# This is the command that runs when the container starts
# It tells uvicorn to run the 'app' object from the 'app.main' file
# --host 0.0.0.0 is MANDATORY. It tells it to listen to requests
# from *outside* the container (i.e., from the Streamlit app).
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]