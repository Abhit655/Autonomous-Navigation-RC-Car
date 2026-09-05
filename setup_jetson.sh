#!/usr/bin/env bash
# setup_jetson.sh
# ---------------
# Environment setup for Jetson Nano (JetPack 4.6 / 5.x)
# Run once after flashing JetPack.
#
# Usage:
#   bash deployment/jetson/setup_jetson.sh

set -e
echo "=============================================="
echo "  Jetson Nano — Segmentation Inference Setup"
echo "=============================================="

# ── System packages ────────────────────────────────────────────────────────
sudo apt-get update -qq
sudo apt-get install -y \
    python3-pip \
    python3-dev \
    libopencv-dev \
    python3-opencv \
    liblapack-dev \
    gstreamer1.0-tools \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav

# ── Python packages ────────────────────────────────────────────────────────
pip3 install --upgrade pip

# NumPy — pin version compatible with JetPack
pip3 install "numpy<1.24"

# PyCUDA
pip3 install pycuda

# ONNX (CPU only — for validation before TRT build)
pip3 install onnx onnxruntime

# Utilities
pip3 install pyyaml tqdm Pillow

echo ""
echo "✓ System packages installed"

# ── Verify TensorRT ────────────────────────────────────────────────────────
echo ""
echo "Checking TensorRT..."
python3 -c "import tensorrt as trt; print(f'  TensorRT version: {trt.__version__}')" \
    || echo "  ⚠ TensorRT not found. Ensure JetPack is correctly installed."

# ── Verify CUDA ────────────────────────────────────────────────────────────
echo ""
echo "Checking CUDA..."
python3 -c "import pycuda.driver as cuda; cuda.init(); \
    dev=cuda.Device(0); \
    print(f'  CUDA device: {dev.name()}, Memory: {dev.total_memory()//1024//1024} MB')"

# ── Copy model files (adjust path as needed) ──────────────────────────────
echo ""
echo "=============================================="
echo "  Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Copy your ONNX model to the Jetson:"
echo "     scp models/exported/segformer_b0.onnx nano@<IP>:~/project/models/exported/"
echo ""
echo "  2. Build TensorRT engine ON the Jetson:"
echo "     python3 deployment/tensorrt/build_engine.py \\"
echo "             --onnx models/exported/segformer_b0.onnx"
echo ""
echo "  3. Run real-time inference:"
echo "     python3 deployment/jetson/camera_inference.py \\"
echo "             --engine models/exported/segformer_b0.trt"
echo "=============================================="
