FROM ubuntu:22.04

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    unzip \
    libprotobuf-dev \
    protobuf-compiler

# Download ONNX Runtime
WORKDIR /opt
RUN wget https://github.com/microsoft/onnxruntime/releases/download/v1.22.0/onnxruntime-linux-x64-1.22.0.tgz && \
    tar -xzf onnxruntime-linux-x64-1.22.0.tgz && \
    rm onnxruntime-linux-x64-1.22.0.tgz

ENV ONNXRUNTIME_DIR=/opt/onnxruntime-linux-x64-1.22.0

# Copy source code
WORKDIR /app
COPY . .

# Build using ONNX Runtime
RUN cmake -B build -S . -DONNXRUNTIME_DIR=$ONNXRUNTIME_DIR && \
    cmake --build build

CMD ["./build/my_executable"]
