# Force Python 3.10.9 with OpenCV dependencies
FROM python:3.10.9-slim

# Install system dependencies for OpenCV
RUN apt-get update && \
    apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Run the app
CMD ["gunicorn", "app:app"]