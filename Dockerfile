# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Prevent Python from writing pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy only the requirements file first to leverage caching
COPY requirements.txt .

# Install dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port (Azure will pass the PORT env variable, defaulting to 8000)
EXPOSE 8000

# Start the app using Uvicorn, binding to 0.0.0.0 to allow external access
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
