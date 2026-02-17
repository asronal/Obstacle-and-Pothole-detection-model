# 🚧 Real-Time Road Obstacle & Pothole Detection (YOLOv8 ONNX)

## 📌 Overview
This project is a real-time computer vision system designed to detect road obstacles and potholes using a lightweight YOLOv8 ONNX model. It is optimized for edge devices like Raspberry Pi, enabling efficient on-device inference for smart transportation, autonomous navigation, and road safety applications.

The system supports:
- Raspberry Pi Camera  
- USB Webcam  
- Video files  
- Single images  

Detected objects are tracked and saved with spatial filtering to avoid duplicate detections.

---

## ⚡ Features
- Real-time pothole and obstacle detection  
- Lightweight ONNX model for edge deployment  
- Multi-source input (Pi Camera, webcam, video, image)  
- Automatic white balance and color stabilization  
- Threaded frame capture  
- Smart spatial duplicate filtering  
- Cooldown mechanism for saving detections  
- Bounding box visualization  
- Timestamped detection saving  
- FPS and inference monitoring  

---

## 🧠 Use Cases
- Autonomous vehicle perception  
- Smart road monitoring  
- Robotics and navigation  
- Driver assistance systems  
- Road safety analytics  

---

## 🛠 Tech Stack
- Python  
- OpenCV  
- YOLOv8  
- ONNX  
- NumPy  
- Raspberry Pi / Edge AI  

---

## 📂 Project Structure
.
├── beta.py  
├── best.onnx  
├── detections/  
│   ├── obstacle/  
│   └── pothole/  
├── test/  
├── requirements.txt  
└── README.md  

---

## 📦 Requirements
This project uses minimal runtime dependencies to ensure smooth deployment on edge devices.

Tested on:
- Raspberry Pi OS  
- Python 3.11  

Install:

pip install -r requirements.txt  

requirements.txt:

numpy==2.4.1  
opencv-python-headless==4.13.0.90  
picamera2==0.3.34  

These are the only libraries required for real-time inference.

---

## ⚠️ Raspberry Pi System Dependencies
Some OpenCV and camera features require system libraries:

sudo apt update  
sudo apt install -y libatlas-base-dev libjpeg-dev libtiff-dev libopenjp2-7-dev  

---

## ▶ Installation

Clone the repository:

git clone https://github.com/asronal/Obstacle-and-Pothole-detection-model.git  
cd road-detection-yolo  

Create virtual environment:

python3 -m venv venv  
source venv/bin/activate  

Install dependencies:

pip install -r requirements.txt  

---

## ▶ Usage

### Raspberry Pi Camera
python beta.py  

### USB Webcam
python beta.py --webcam  

Specify index:
python beta.py --webcam 2  

### Video file
python beta.py --video video.mp4  

### Image
python beta.py --image image.jpg  

---

## 📊 Output
Console shows:
- FPS  
- Inference time  
- Detection count  
- Saved detections  

Images are saved in:

detections/obstacle/  
detections/pothole/  

Each saved image contains bounding boxes, labels, confidence, and timestamps.

---

## ⚡ Performance
Optimized for Raspberry Pi 4 with CPU inference and multithreading.

Typical performance:
5–7 FPS depending on model and resolution.

---

## 🔧 Configuration
Modify these in the script:
- CONF_THRESHOLD  
- NMS_THRESHOLD  
- SPATIAL_THRESHOLD  
- COOLDOWN_SECONDS  
- INPUT_SIZE  

---

## 🚀 Future Improvements 
- Model quantization  
- Object tracking    
- GPS pothole mapping  
- Cloud dashboard  

---
