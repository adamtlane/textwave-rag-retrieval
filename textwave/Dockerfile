# Use a slim Python base image.
FROM python:3.10-slim

# Set environment variables.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Set the working directory inside the container.
WORKDIR /app

# Install system dependencies.
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file and install dependencies.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project into the container.
COPY . .

# Expose the port Flask runs on.
EXPOSE 5000

# Set the entrypoint to run the Flask app.
CMD ["python", "textwave/app.py"]