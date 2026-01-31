#!/bin/bash

# YOLO Object Detection Setup Script
# Automates the complete setup process

set -e  # Exit on any error

echo "🎯 YOLO Object Detection Setup"
echo "================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3 and try again."
    exit 1
fi

echo "✅ Python 3 found"

# Check if pip is installed
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 is required but not installed."
    echo "Please install pip and try again."
    exit 1
fi

echo "✅ pip3 found"

# Install Python dependencies
echo ""
echo "📦 Installing Python dependencies..."
pip3 install ultralytics onnx

if [ $? -eq 0 ]; then
    echo "✅ Dependencies installed successfully"
else
    echo "❌ Failed to install dependencies"
    exit 1
fi

# Export YOLO model
echo ""
echo "🤖 Exporting YOLOv8 model to ONNX format..."
python3 export_yolo.py n

if [ $? -eq 0 ]; then
    echo "✅ Model exported successfully"
else
    echo "❌ Failed to export model"
    exit 1
fi

# Check if model file exists
if [ -f "yolov8n.onnx" ]; then
    model_size=$(du -h yolov8n.onnx | cut -f1)
    echo "📁 Model file: yolov8n.onnx ($model_size)"
else
    echo "❌ Model file not found"
    exit 1
fi

# Start local server
echo ""
echo "🚀 Starting local web server..."
echo "📍 Server URL: http://localhost:8080"
echo "🔄 Starting server... (Press Ctrl+C to stop)"
echo ""

# Try different server options
if command -v python3 &> /dev/null; then
    echo "Using Python HTTP server..."
    python3 -m http.server 8080
elif command -v node &> /dev/null && command -v npx &> /dev/null; then
    echo "Using Node.js HTTP server..."
    npx http-server -p 8080
else
    echo "⚠️  No suitable HTTP server found."
    echo "Please install Node.js or use Python 3 to run:"
    echo "python3 -m http.server 8080"
    echo ""
    echo "Then open http://localhost:8080 in your browser."
fi