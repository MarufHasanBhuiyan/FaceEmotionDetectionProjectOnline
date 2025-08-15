# Force Python 3.10.9
FROM python:3.10.9-slim

# Set working directory
WORKDIR /app

# Install dependencies first (for caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy all files
COPY . .

# Run the app
CMD ["gunicorn", "app:app"]