// YOLO Object Detection with ONNX Runtime Web
// Real AI-powered object detection in the browser

let session = null;
let isModelLoaded = false;
let currentImage = null;

// COCO dataset class names (80 classes)
const classNames = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'monkey', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
];

// Color palette for bounding boxes
const colors = [
    '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57', '#FF9FF3', '#54A0FF',
    '#5F27CD', '#00D2D3', '#FF9F43', '#10AC84', '#EE5A24', '#0984E3', '#B55400',
    '#00B894', '#FDCB6E', '#6C5CE7', '#A29BFE', '#FD79A8', '#E17055'
];

// DOM elements
const imageInput = document.getElementById('imageInput');
const uploadArea = document.getElementById('uploadArea');
const detectBtn = document.getElementById('detectBtn');
const clearBtn = document.getElementById('clearBtn');
const buttonContainer = document.getElementById('buttonContainer');

const statusEl = document.getElementById('status');
const imageDisplay = document.getElementById('imageDisplay');
const detectionsList = document.getElementById('detectionsList');
const inferenceTimeEl = document.getElementById('inferenceTime');
const objectCountEl = document.getElementById('objectCount');
const processingTimeEl = document.getElementById('processingTime');

// Initialize the application
async function initApp() {
    console.log('🚀 Initializing YOLO Object Detection App...');
    
    // Check if elements exist
    console.log('detectBtn exists:', !!detectBtn);
    console.log('clearBtn exists:', !!clearBtn);
    
    // Setup event listeners
    imageInput.addEventListener('change', handleImageUpload);
    uploadArea.addEventListener('click', () => imageInput.click());
    detectBtn.addEventListener('click', detectObjects);
    if (clearBtn) {
        clearBtn.addEventListener('click', clearDetectionResults);
    } else {
        console.error('clearBtn element not found!');
    }

    
    // Configure ONNX Runtime
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.17.0/dist/';
    
    // Load the YOLO model
    await loadModel();
}

// Load YOLO model
async function loadModel() {
    try {
        statusEl.className = 'status loading';
        statusEl.textContent = 'Loading YOLOv8 model... (~6MB download)';
        
        console.log('Loading ONNX model...');
        
        // Load the YOLO model
        session = await ort.InferenceSession.create('yolov8n.onnx', {
            executionProviders: ['wasm'],
            graphOptimizationLevel: 'all'
        });
        
        isModelLoaded = true;
        
        statusEl.className = 'status success';
        statusEl.textContent = 'YOLOv8 model loaded! Ready for object detection.';
        
        // Enable UI elements
        detectBtn.disabled = false;

        
        console.log('Model loaded successfully');
        console.log('Input shape:', session.inputNames);
        console.log('Output shape:', session.outputNames);
        
    } catch (error) {
        console.error('Model loading error:', error);
        statusEl.className = 'status error';
        statusEl.textContent = 'Error loading model. Please ensure yolov8n.onnx is available.';
    }
}

// Handle image upload
function handleImageUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    const reader = new FileReader();
    reader.onload = function(e) {
        loadImageFromDataUrl(e.target.result);
    };
    reader.readAsDataURL(file);
}



// Load image from data URL
function loadImageFromDataUrl(dataUrl) {
    const img = new Image();
    img.onload = () => loadImageFromElement(img);
    img.src = dataUrl;
}

// Load image from element
function loadImageFromElement(img) {
    currentImage = img;
    
    // Clear previous results
    clearResults();
    
    // Hide upload area and show image display
    uploadArea.classList.add('hidden');
    imageDisplay.classList.remove('hidden');
    buttonContainer.classList.remove('hidden');
    
    // Show detect button and hide clear button (reset to initial state)
    detectBtn.classList.remove('hidden');
    clearBtn.classList.add('hidden');
    
    // Display the image
    imageDisplay.innerHTML = '';
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // Calculate display size while maintaining aspect ratio
    const maxSize = 400;
    let { width, height } = calculateDisplaySize(img.width, img.height, maxSize);
    
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(img, 0, 0, width, height);
    
    imageDisplay.appendChild(canvas);
    
    console.log(`📷 Image loaded: ${img.width}x${img.height} -> ${width}x${height}`);
}

// Calculate display size maintaining aspect ratio
function calculateDisplaySize(originalWidth, originalHeight, maxSize) {
    if (originalWidth <= maxSize && originalHeight <= maxSize) {
        return { width: originalWidth, height: originalHeight };
    }
    
    const ratio = Math.min(maxSize / originalWidth, maxSize / originalHeight);
    return {
        width: Math.round(originalWidth * ratio),
        height: Math.round(originalHeight * ratio)
    };
}

// Main detection function
async function detectObjects() {
    if (!isModelLoaded || !currentImage) {
        alert('Please wait for model to load and upload an image first!');
        return;
    }
    
    const startTime = performance.now();
    
    try {
        detectBtn.disabled = true;
        detectBtn.textContent = '🔍 Detecting...';
        
        // Preprocess image
        console.log('Preprocessing image...');
        const preprocessed = preprocessImage(currentImage);
        
        // Run inference
        console.log('Running YOLO inference...');
        const inferenceStart = performance.now();
        const detections = await runInference(preprocessed);
        const inferenceTime = performance.now() - inferenceStart;
        
        // Draw results
        drawDetections(detections, preprocessed);
        
        // Update metrics
        const totalTime = performance.now() - startTime;
        updateMetrics(inferenceTime, totalTime, detections.length);
        
        // Update detections list
        updateDetectionsList(detections);
        
        // Show clear button and hide detect button
        console.log('Showing clear button, hiding detect button');
        detectBtn.classList.add('hidden');
        clearBtn.classList.remove('hidden');
        
        console.log(`Detection complete: ${detections.length} objects in ${totalTime.toFixed(1)}ms`);
        
    } catch (error) {
        console.error('Detection error:', error);
        statusEl.className = 'status error';
        statusEl.textContent = 'Detection failed: ' + error.message;
    } finally {
        detectBtn.disabled = false;
        detectBtn.textContent = 'Detect Objects';
    }
}

// Preprocess image for YOLO
function preprocessImage(img) {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    // YOLO input size
    const inputSize = 640;
    canvas.width = inputSize;
    canvas.height = inputSize;
    
    // Calculate scaling and padding
    const scale = Math.min(inputSize / img.width, inputSize / img.height);
    const scaledWidth = img.width * scale;
    const scaledHeight = img.height * scale;
    const offsetX = (inputSize - scaledWidth) / 2;
    const offsetY = (inputSize - scaledHeight) / 2;
    
    // Fill with gray and draw scaled image
    ctx.fillStyle = '#808080';
    ctx.fillRect(0, 0, inputSize, inputSize);
    ctx.drawImage(img, offsetX, offsetY, scaledWidth, scaledHeight);
    
    // Get image data and convert to float32 tensor
    const imageData = ctx.getImageData(0, 0, inputSize, inputSize);
    const { data } = imageData;
    
    // Convert to RGB and normalize [0, 255] -> [0, 1]
    const float32Data = new Float32Array(3 * inputSize * inputSize);
    for (let i = 0; i < inputSize * inputSize; i++) {
        float32Data[i] = data[i * 4] / 255.0; // Red
        float32Data[inputSize * inputSize + i] = data[i * 4 + 1] / 255.0; // Green
        float32Data[2 * inputSize * inputSize + i] = data[i * 4 + 2] / 255.0; // Blue
    }
    
    return {
        data: float32Data,
        scale: scale,
        offsetX: offsetX,
        offsetY: offsetY
    };
}

// Run YOLO inference
async function runInference(preprocessed) {
    // Create input tensor
    const inputTensor = new ort.Tensor('float32', preprocessed.data, [1, 3, 640, 640]);
    
    // Prepare feeds
    const feeds = {};
    feeds[session.inputNames[0]] = inputTensor;
    
    // Run the model
    const outputData = await session.run(feeds);
    const output = outputData[session.outputNames[0]];
    
    console.log('Model output shape:', output.dims);
    console.log('Model output data length:', output.data.length);
    console.log('First 10 values:', Array.from(output.data.slice(0, 10)));
    
    // Process output
    return processYOLOOutput(output.data, preprocessed);
}

// Process YOLO output
function processYOLOOutput(output, preprocessed) {
    const detections = [];
    const confidenceThreshold = 0.4; // Reasonable threshold
    const numDetections = 8400; // YOLOv8 outputs 8400 detections
    
    // YOLOv8 output format: [1, 84, 8400]
    // 84 = 4 bbox coords (x_center, y_center, width, height) + 80 class scores
    
    for (let i = 0; i < numDetections; i++) {
        // Get bounding box coordinates (center format) - the output is [84, 8400] after removing batch dimension
        const x_center = output[0 * numDetections + i];  // x center
        const y_center = output[1 * numDetections + i];  // y center
        const width = output[2 * numDetections + i];     // width
        const height = output[3 * numDetections + i];    // height
        
        // Find best class and confidence
        let maxScore = 0;
        let maxClass = 0;
        
        for (let c = 0; c < 80; c++) {
            const score = output[(4 + c) * numDetections + i];
            if (score > maxScore) {
                maxScore = score;
                maxClass = c;
            }
        }
        
        // Only keep detections with high confidence
        if (maxScore > confidenceThreshold) {
            // Convert from normalized coordinates to pixel coordinates
            const x_pixel = x_center * (640 / 640);  // Model uses 640x640
            const y_pixel = y_center * (640 / 640);
            const w_pixel = width * (640 / 640);
            const h_pixel = height * (640 / 640);
            
            // Convert from model space to original image space
            const originalX = (x_pixel - w_pixel/2 - preprocessed.offsetX) / preprocessed.scale;
            const originalY = (y_pixel - h_pixel/2 - preprocessed.offsetY) / preprocessed.scale;
            const originalW = w_pixel / preprocessed.scale;
            const originalH = h_pixel / preprocessed.scale;
            
            // Ensure coordinates are within image bounds
            const x1 = Math.max(0, originalX);
            const y1 = Math.max(0, originalY);
            const x2 = Math.min(currentImage.width, originalX + originalW);
            const y2 = Math.min(currentImage.height, originalY + originalH);
            const boxW = x2 - x1;
            const boxH = y2 - y1;
            
            // Only keep reasonable sized boxes
            if (boxW > 10 && boxH > 10) {
                detections.push({
                    class: classNames[maxClass],
                    confidence: maxScore,
                    bbox: [x1, y1, boxW, boxH],
                    classId: maxClass
                });
            }
        }
    }
    
    console.log(`Raw detections: ${detections.length}`);
    
    // Apply Non-Maximum Suppression
    const finalDetections = applyNMS(detections, 0.4);
    console.log(`After NMS: ${finalDetections.length}`);
    
    return finalDetections;
}

// Non-Maximum Suppression
function applyNMS(detections, iouThreshold) {
    // Sort by confidence (descending)
    detections.sort((a, b) => b.confidence - a.confidence);
    
    const keep = [];
    const suppressed = new Set();
    
    for (let i = 0; i < detections.length; i++) {
        if (suppressed.has(i)) continue;
        
        keep.push(detections[i]);
        
        // Suppress overlapping detections
        for (let j = i + 1; j < detections.length; j++) {
            if (suppressed.has(j)) continue;
            
            const iou = calculateIoU(detections[i].bbox, detections[j].bbox);
            if (iou > iouThreshold) {
                suppressed.add(j);
            }
        }
    }
    
    return keep;
}

// Calculate Intersection over Union
function calculateIoU(box1, box2) {
    const [x1, y1, w1, h1] = box1;
    const [x2, y2, w2, h2] = box2;
    
    // Calculate intersection
    const intersectX = Math.max(0, Math.min(x1 + w1, x2 + w2) - Math.max(x1, x2));
    const intersectY = Math.max(0, Math.min(y1 + h1, y2 + h2) - Math.max(y1, y2));
    const intersectArea = intersectX * intersectY;
    
    // Calculate union
    const box1Area = w1 * h1;
    const box2Area = w2 * h2;
    const unionArea = box1Area + box2Area - intersectArea;
    
    return intersectArea / unionArea;
}

// Draw detection results
function drawDetections(detections, preprocessed) {
    const canvas = imageDisplay.querySelector('canvas');
    const ctx = canvas.getContext('2d');
    
    // Clear previous drawings
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    // Redraw the image first
    ctx.drawImage(currentImage, 0, 0, canvas.width, canvas.height);
    
    // Calculate scaling factor from original image to display canvas
    const scaleX = canvas.width / currentImage.width;
    const scaleY = canvas.height / currentImage.height;
    
    // Draw bounding boxes
    detections.forEach((detection, index) => {
        const [x, y, w, h] = detection.bbox;
        const color = colors[detection.classId % colors.length];
        
        // Scale coordinates to canvas size
        const canvasX = x * scaleX;
        const canvasY = y * scaleY;
        const canvasW = w * scaleX;
        const canvasH = h * scaleY;
        
        // Draw bounding box
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(canvasX, canvasY, canvasW, canvasH);
        
        // Draw label background
        const label = `${detection.class} ${(detection.confidence * 100).toFixed(1)}%`;
        ctx.font = 'bold 16px Arial';
        const textMetrics = ctx.measureText(label);
        const textWidth = textMetrics.width;
        const textHeight = 20;
        
        // Ensure label stays within canvas bounds
        const labelX = Math.min(canvasX, canvas.width - textWidth - 10);
        const labelY = Math.max(textHeight, canvasY);
        
        ctx.fillStyle = color;
        ctx.fillRect(labelX, labelY - textHeight, textWidth + 10, textHeight + 5);
        
        // Draw label text
        ctx.fillStyle = 'white';
        ctx.textBaseline = 'top';
        ctx.fillText(label, labelX + 5, labelY - textHeight + 2);
    });
}

// Update performance metrics
function updateMetrics(inferenceTime, totalTime, objectCount) {
    inferenceTimeEl.textContent = Math.round(inferenceTime);
    processingTimeEl.textContent = Math.round(totalTime);
    objectCountEl.textContent = objectCount;
}

// Update detections list
function updateDetectionsList(detections) {
    if (detections.length === 0) {
        detectionsList.innerHTML = '<h3>Results</h3> <p style="color: #666; margin-bottom: 20px; text-align: center;">No objects detected</p>';
        return;
    }
    
    const html = detections.map(detection => `
        <div class="detection-item">
            <span class="detection-class">${detection.class}</span>
            <span class="detection-confidence">${(detection.confidence * 100).toFixed(1)}%</span>
        </div>
    `).join('');
    
    detectionsList.innerHTML = `<h3>Results</h3> ${html}`;
}

// Clear detection results and show detect button again
function clearDetectionResults() {
    console.log('Clearing detection results');
    // Clear results
    clearResults();
    
    // Reset current image
    currentImage = null;
    
    // Show upload area and hide image display
    uploadArea.classList.remove('hidden');
    imageDisplay.classList.add('hidden');
    buttonContainer.classList.add('hidden');
    
    // Clear image display content
    imageDisplay.innerHTML = '';
    
    console.log('Detection results cleared - back to upload state');
}

// Clear results
function clearResults() {
    inferenceTimeEl.textContent = '-';
    processingTimeEl.textContent = '-';
    objectCountEl.textContent = '-';
    detectionsList.innerHTML = '<h3>Results</h3> <p style="color: #666; margin-bottom: 20px; text-align: center;">Run detection to see results</p>';
}

// Initialize app when page loads
document.addEventListener('DOMContentLoaded', initApp);
