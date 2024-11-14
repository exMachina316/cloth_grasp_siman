# Use an official Python image as a base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    cmake \
    build-essential \
    libgl1-mesa-glx \
    libopencv-dev \
    libeigen3-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

# Set the environment variable to avoid Open3D GLX errors
ENV LIBGL_ALWAYS_INDIRECT=1
