# Python 3.10.9 (TensorFlow 2.10 compatible)
FROM python:3.10.9-slim

# System deps for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose (Render sets PORT)
ENV PORT=10000
CMD exec gunicorn --bind 0.0.0.0:$PORT app:app
