# Use an official Ubuntu as a parent image
FROM ubuntu:18.04

# Set the maintainer label
LABEL maintainer="youremail@example.com"

# Set environment variables to non-interactive (this prevents some prompts)
ENV DEBIAN_FRONTEND=non-interactive

# Install necessary packages
RUN apt-get update && apt-get install -y \
    cmake \
    g++ \
    gcc \
    make \
    wget \
    unzip \
    libgtk2.0-dev \
    pkg-config \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    python-dev \
    python-numpy \
    libtbb2 \
    libtbb-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libdc1394-22-dev

# Download OpenCV and OpenCV Contrib modules
WORKDIR /opencv
RUN wget -O opencv.zip https://github.com/opencv/opencv/archive/4.1.0.zip && \
    wget -O opencv_contrib.zip https://github.com/opencv/opencv_contrib/archive/4.1.0.zip && \
    unzip opencv.zip && \
    unzip opencv_contrib.zip && \
    mkdir opencv-4.1.0/build && \
    cd opencv-4.1.0/build && \
    cmake -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_INSTALL_PREFIX=/usr/local \
          -D OPENCV_EXTRA_MODULES_PATH=../../opencv_contrib-4.1.0/modules \
          -D OPENCV_ENABLE_NONFREE=ON .. && \
    make -j$(nproc) && \
    make install

# Clean up the OpenCV directory
WORKDIR /
RUN rm -rf /opencv

# Set the working directory in the container to /app
WORKDIR /app

# Specify an absolute path - use `docker run` command to mount a host directory to the container
CMD ["bash"]
