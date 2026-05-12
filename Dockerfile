# Use the official Python image
FROM python:3.14.5-slim-bookworm
# Set the database path environment variable
ENV APP_DIR=/app
ENV DATA_DIR=/app/assets/data
ENV DATABASE_PATH=/assets/data/BSLVC_sqlite.db
ENV ADVANCED_MAPPING_PATH=/assets/data/advanced_regional_mapping.csv
# Optional: Set custom cache directories
ENV CACHE_DIR=/app/cache
ENV MPLCONFIGDIR=/tmp/matplotlib

# Install dependencies
COPY requirements_3.14.txt ./requirements.txt
RUN pip install --no-cache-dir -r ./requirements.txt
COPY . ./

# Create cache directories with proper permissions
RUN mkdir -p /app/cache /tmp/matplotlib && \
    chmod 777 /app/cache /tmp/matplotlib
RUN mkdir -p /cache && \
    chmod 777 /cache
    
CMD gunicorn -b 0.0.0.0:27589 app:server

