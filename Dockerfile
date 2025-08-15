# Python 3.10 to match TF 2.10 wheels nicely
FROM python:3.10.9-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# System packages (Pillow wheels include libjpeg; slim still needs some basics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY . .

# Expose the default Flask port
EXPOSE 5000

# Gunicorn (CPU only). 2 workers is enough for free tiers.
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]
