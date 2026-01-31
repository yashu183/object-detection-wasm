# YOLO Object Detection with WebAssembly

Real-time AI-powered object detection running entirely in your browser using YOLOv8 and ONNX Runtime Web.

## 🎯 Features

- ✅ **Real AI Detection**: YOLOv8 ONNX model with 80 object classes
- ✅ **WebAssembly Powered**: ONNX Runtime Web for fast inference
- ✅ **Client-Side Processing**: No server required, privacy-first
- ✅ **Real-Time Performance**: Sub-50ms inference time
- ✅ **Professional UI**: Modern interface with metrics
- ✅ **Cross-Platform**: Works on desktop and mobile browsers

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install Python dependencies for model export
pip install ultralytics onnx
```

### 2. Export YOLO Model

```bash
# Export YOLOv8 nano model (recommended for web)
python export_yolo.py n

# Or export other sizes:
# python export_yolo.py s  # Small (~22MB)
# python export_yolo.py m  # Medium (~50MB)
# python export_yolo.py l  # Large (~87MB)
```

This creates `yolov8n.onnx` (~6MB) in your project directory.

### 3. Start Local Server

```bash
# Using Python
python -m http.server 8080

# Or using Node.js
npx http-server -p 8080

# Or using any other static file server
```

### 4. Open in Browser

Navigate to: http://localhost:8080

🎉 **That's it!** Upload images and see real AI-powered object detection in action.

## 📊 Performance Guide

| Model | Size | Inference Time | Accuracy | Best For |
|-------|------|----------------|----------|----------|
| YOLOv8n | ~6MB | 20-30ms | Good | **Web deployment** |
| YOLOv8s | ~22MB | 30-40ms | Better | Balance |
| YOLOv8m | ~50MB | 40-60ms | Very Good | Accuracy focus |
| YOLOv8l | ~87MB | 60-80ms | Excellent | High accuracy |
| YOLOv8x | ~136MB | 80-120ms | Best | Research/offline |

> **Recommendation**: Use YOLOv8n for web demos - it provides excellent performance with minimal size.

## 🎬 Demo Features

### Object Detection
- **80 COCO Classes**: person, car, dog, cat, bicycle, etc.
- **Confidence Threshold**: 25% (adjustable in code)
- **Non-Maximum Suppression**: Removes duplicate detections
- **Color-Coded Boxes**: Each class gets a unique color

### Performance Metrics
- **Inference Time**: Pure model execution time
- **Total Time**: Including preprocessing and postprocessing
- **Object Count**: Number of detected objects
- **Model Size**: Display of loaded model size

### User Interface
- **Drag & Drop**: Upload images easily
- **Sample Images**: One-click demo with sample image
- **Real-time Results**: Instant visual feedback
- **Responsive Design**: Works on mobile and desktop

## 🛠️ Technical Architecture

```
User Image → Canvas Preprocessing → ONNX Runtime (WASM) → Postprocessing → UI Display
     ↓              ↓                      ↓                   ↓             ↓
  Browser File   640x640 Float32      YOLOv8 Inference   NMS + Filtering   Canvas Overlay
```

### Key Components

1. **Image Preprocessing**
   - Resize to 640x640 with aspect ratio preservation
   - Normalize to [0,1] float32 values
   - Convert RGB to CHW format (channels-first)

2. **ONNX Runtime Web**
   - WebAssembly backend for CPU inference
   - Optimized for browser performance
   - Same engine used by Microsoft Office, Adobe, Google

3. **Postprocessing**
   - Parse 8400 detections from model output
   - Apply confidence filtering (25% threshold)
   - Non-Maximum Suppression (45% IoU threshold)
   - Convert coordinates back to original image scale

## 🔧 Customization

### Change Model
Replace `yolov8n.onnx` with any other YOLOv8 ONNX model:
```javascript
session = await ort.InferenceSession.create('yolov8s.onnx', {
    executionProviders: ['wasm'],
    graphOptimizationLevel: 'all'
});
```

### Adjust Thresholds
In `yolo-detection.js`:
```javascript
const confidenceThreshold = 0.25; // Confidence threshold (0-1)
const iouThreshold = 0.45;         // NMS IoU threshold (0-1)
```

### Add Custom Classes
Extend the `classNames` array for custom trained models:
```javascript
const classNames = [
    'your_class_1', 'your_class_2', // ... custom classes
];
```

## 📱 Browser Compatibility

- ✅ **Chrome/Edge**: Full WebAssembly support
- ✅ **Firefox**: Full WebAssembly support  
- ✅ **Safari**: Full WebAssembly support
- ✅ **Mobile**: iOS Safari, Chrome Mobile

## 🎓 Educational Value

Perfect for demonstrating:
- **WebAssembly in Action**: Real ML inference in browsers
- **Client-Side AI**: Privacy-preserving machine learning
- **Modern Web APIs**: File handling, Canvas, Web Workers
- **Performance Optimization**: WASM vs JavaScript comparison
- **Real-World Architecture**: Same stack as Google Meet, Adobe Photoshop Web

## 🤝 Common Issues

### Model Loading Fails
- Ensure `yolov8n.onnx` is in the same directory as `index.html`
- Check browser console for specific error messages
- Verify you're using HTTP server (not file:// protocol)

### Slow Performance
- Try YOLOv8n instead of larger models
- Check if WebAssembly is enabled in browser
- Close other browser tabs to free up memory

### CORS Errors
- Use a proper HTTP server instead of opening HTML directly
- For production, serve files from same domain

## 📚 Learn More

- [ONNX Runtime Web Documentation](https://onnxruntime.ai/docs/tutorials/web/)
- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [WebAssembly Documentation](https://webassembly.org/)
- [Canvas API Documentation](https://developer.mozilla.org/en-US/docs/Web/API/Canvas_API)

## 🌟 Next Steps

1. **Deploy to Production**: Host on GitHub Pages, Netlify, or Vercel
2. **Add Video Support**: Real-time webcam object detection
3. **Custom Training**: Train YOLOv8 on your own datasets
4. **Performance Optimization**: WebGPU backend for GPU acceleration
5. **Mobile App**: Wrap in Capacitor/Cordova for mobile deployment

---

**Built with ❤️ using YOLOv8, ONNX Runtime Web, and WebAssembly**