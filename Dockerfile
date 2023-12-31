# Use the official Python image as the base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code to the container
COPY . .

# Expose port 5000 for the Flask application
EXPOSE 5000

# Set the environment variable
ENV FLASK_APP=index.py

# Run the Flask application
CMD ["flask", "run", "--host=0.0.0.0"]
