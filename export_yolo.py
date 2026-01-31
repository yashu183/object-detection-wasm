#!/usr/bin/env python3
"""
YOLO Model Export Script
Exports YOLOv8 model to ONNX format for web deployment
"""

import os
import sys
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import ultralytics
        import onnx
        print("✅ All dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("\n📦 Install required packages:")
        print("pip install ultralytics onnx")
        return False

def export_yolo_model(model_size='n', output_path=None):
    """
    Export YOLO model to ONNX format
    
    Args:
        model_size (str): Model size - 'n' (nano), 's' (small), 'm' (medium), 'l' (large), 'x' (extra large)
        output_path (str): Output path for the ONNX file
    """
    try:
        from ultralytics import YOLO
        
        # Model size configurations
        model_configs = {
            'n': {'name': 'yolov8n.pt', 'size': '~6MB', 'speed': 'fastest'},
            's': {'name': 'yolov8s.pt', 'size': '~22MB', 'speed': 'fast'},
            'm': {'name': 'yolov8m.pt', 'size': '~50MB', 'speed': 'medium'},
            'l': {'name': 'yolov8l.pt', 'size': '~87MB', 'speed': 'slow'},
            'x': {'name': 'yolov8x.pt', 'size': '~136MB', 'speed': 'slowest'},
        }
        
        if model_size not in model_configs:
            print(f"❌ Invalid model size. Choose from: {', '.join(model_configs.keys())}")
            return False
        
        config = model_configs[model_size]
        model_name = config['name']
        
        print(f"🚀 Exporting YOLOv8{model_size} model...")
        print(f"📊 Model info: {config['size']}, {config['speed']} inference")
        
        # Load YOLOv8 model
        print(f"📥 Loading {model_name}...")
        model = YOLO(model_name)
        
        # Set output path
        if output_path is None:
            output_path = f'yolov8{model_size}.onnx'
        
        # Export to ONNX format
        print("🔄 Exporting to ONNX format...")
        model.export(
            format='onnx',
            dynamic=False,  # Static input shape for better performance
            simplify=True,  # Simplify the model
            opset=12,      # ONNX opset version (compatible with ONNX Runtime Web)
            imgsz=640,     # Input image size
        )
        
        # Check if export was successful
        onnx_path = Path(model_name).stem + '.onnx'
        if os.path.exists(onnx_path):
            # Rename to desired output path if different
            if onnx_path != output_path:
                os.rename(onnx_path, output_path)
            
            file_size = os.path.getsize(output_path) / (1024 * 1024)  # Size in MB
            print(f"✅ Export successful!")
            print(f"📁 Output: {output_path}")
            print(f"📦 File size: {file_size:.1f}MB")
            print(f"🔧 Input shape: [1, 3, 640, 640]")
            print(f"⚡ Output shape: [1, 84, 8400]")
            
            return True
        else:
            print("❌ Export failed - output file not found")
            return False
            
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False

def main():
    """Main function"""
    print("🎯 YOLO to ONNX Export Tool")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Get model size from command line or use default
    model_size = 'n'  # Default to nano (smallest/fastest)
    if len(sys.argv) > 1:
        model_size = sys.argv[1].lower()
    
    # Get output path from command line if provided
    output_path = None
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    print(f"\n🎯 Configuration:")
    print(f"   Model size: YOLOv8{model_size}")
    print(f"   Output: {output_path or f'yolov8{model_size}.onnx'}")
    
    # Export model
    success = export_yolo_model(model_size, output_path)
    
    if success:
        print("\n🎉 Next steps:")
        print("1. Place the .onnx file in your web project directory")
        print("2. Start a local web server (python -m http.server 8080)")
        print("3. Open http://localhost:8080 in your browser")
        print("4. Upload images and enjoy real AI-powered object detection!")
        
        print("\n📚 Model performance guide:")
        print("   • YOLOv8n: ~6MB,  fastest inference (~20-30ms)")
        print("   • YOLOv8s: ~22MB, fast inference (~30-40ms)")
        print("   • YOLOv8m: ~50MB, balanced (~40-60ms)")
        print("   • YOLOv8l: ~87MB, accurate (~60-80ms)")
        print("   • YOLOv8x: ~136MB, most accurate (~80-120ms)")
    else:
        print("\n❌ Export failed. Please check the error messages above.")
        sys.exit(1)

if __name__ == "__main__":
    main()