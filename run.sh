#!/bin/bash

echo "📦 Rebuilding Docker image..."
docker build -t onnx_win_build -f docker/ubuntu.Dockerfile .

if [ $? -ne 0 ]; then
  echo "❌ Build failed"
  exit 1
fi

echo "🚀 Running container with volume..."
docker run --rm -v $(pwd):/app onnx_win_build