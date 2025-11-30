# frontend/Dockerfile
FROM python:3.12-slim

# Set working directory inside container
WORKDIR /app

# Copy your static site files
COPY . /app

# Expose the port the HTTP server will listen on
EXPOSE 5500

# Run Python’s built-in web server
CMD ["python3", "-m", "http.server", "5500", "--bind", "0.0.0.0"]
