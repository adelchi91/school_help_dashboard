# Use the official Python image as base
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file from the root directory into the container at /app
COPY requirements.txt .

# Install any dependencies specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the contents of the app directory into the container at /app
COPY ./app/ .

# Copy the data directory from the root directory into the container at /app/data
COPY ./data/ ./data/

# Expose the port that Streamlit listens on
EXPOSE 8501

# Command to run the Streamlit app when the container starts
CMD ["streamlit", "run", "dashboard.py"]

