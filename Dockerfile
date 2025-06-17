# Use the official Python 3.12 image
FROM python:3.12

# Set environment variables to force CPU-only PyTorch
ENV CUDA_VISIBLE_DEVICES=""
ENV TORCH_USE_CUDA_DSA=0

# Set the working directory
WORKDIR /app

# Install pipenv and system dependencies
RUN pip install --no-cache-dir pipenv

# Copy Pipfile and Pipfile.lock
COPY Pipfile Pipfile.lock ./

# Copy .env file
COPY .env ./

# Install dependencies via pipenv
RUN pipenv install --deploy --ignore-pipfile

# Copy the application code
COPY src/ ./

# Expose the port FastAPI will run on
EXPOSE 8000

# Run the application
CMD ["pipenv", "run", "fastapi", "run", "main.py", "--host", "0.0.0.0", "--port", "8000"]