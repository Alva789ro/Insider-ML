# Use an official Python base image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY app/requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire app folder
COPY app /app

# Expose the port (FastAPI default: 8000)
EXPOSE 8000

# Command to run your app using Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]