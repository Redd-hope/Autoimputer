# Use an official Python image as a base
FROM python:3.12-slim

# Set working directory to /app
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port 8020
EXPOSE 8021

# Run command to start the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8021"]