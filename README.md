# AI Camera System - Phase 1 🎥

A comprehensive **real-time AI-powered surveillance and analytics system** using YOLOv8 and OpenCV. This project includes multiple intelligent detection modules for crowd monitoring, face recognition, security threat detection, and behavioral analysis.

---

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Modules](#modules)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

### Core Capabilities
- **Real-time Person Detection** - Detect and track individuals using YOLOv8
- **Unique ID Tracking** - Assign and maintain unique identifiers for each person
- **Crowd Density Analysis** - Calculate and monitor crowd density patterns
- **FPS Monitoring** - Real-time performance metrics
- **Grid-based Heatmap** - Visualize crowd distribution across zones

### Advanced Detection Modules
- **PPE Detection** - Detect Personal Protective Equipment (helmets, masks, vests)
- **Face Recognition** - Identify and register individuals using facial recognition
- **Loitering Detection** - Detect people remaining in a region for extended periods
- **Intrusion Detection** - Identify unauthorized zone entries
- **Line Crossing** - Track objects crossing defined virtual lines
- **Region Entrance/Exit** - Monitor people entering and leaving specific regions
- **Heatmap Visualization** - Generate spatial density maps

---

## 🗂️ Project Structure

```
Ai_system_phase_1_repo/
├── Core_model_1/                      # Main AI detection engine
│   ├── AI_CAMERA_PHASE1/              # Core model package
│   ├── ai_camera_system.py            # Primary detection script
│   ├── streamlit_rtsp_app.py          # Web dashboard interface
│   ├── Core_Model_1.pt                # Pre-trained YOLOv8 model
│   └── requirement.txt                # Dependencies
│
├── AI_FACE_DASHBOARD_FINAL/           # Face recognition system
│   ├── register_dashboard/            # Face registration interface
│   └── face_database/                 # Database of registered faces
│
├── ppe_detection/                     # Personal Protective Equipment detection
│   ├── app.py                         # PPE detection application
│   ├── core/                          # Core PPE detection modules
│   └── face_database/                 # Associated face data
│
├── ppe_detection2/                    # Enhanced PPE detection variant
│
├── HeatMap/                           # Crowd heatmap generation
│
├── Line_crossing/                     # Virtual line crossing detection
│
├── Region_Entrance_Exit/              # Region boundary monitoring
│
├── intrusion detection/               # Unauthorized access detection
│
├── loitering detection/               # Prolonged presence detection
│
├── recognition_dashboard/             # Recognition system interface
│
└── README.md                          # This file
```

---

## 🛠️ Tech Stack

### Core Technologies
- **Python 3.8+** - Main programming language
- **OpenCV** - Computer vision and image processing
- **Ultralytics YOLOv8** - Object detection framework
- **NumPy** - Numerical computations
- **Streamlit** - Web application framework for dashboards
- **PyTorch** - Deep learning framework

### Optional Components
- **RTSP Support** - Stream from IP cameras
- **Face Recognition Libraries** - Facial analysis modules
- **Database System** - For face registration and tracking

---

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- Webcam or IP camera (for video input)
- 4GB+ RAM (recommended)
- CUDA-capable GPU (optional, for faster inference)

### Setup Instructions

#### 1. Clone the Repository
```bash
git clone https://github.com/ShivSingh-17/Ai_system_phase_1_repo.git
cd Ai_system_phase_1_repo
```

#### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### 3. Install Dependencies

**For Core Model:**
```bash
cd Core_model_1
pip install -r requirement.txt
```

**For PPE Detection:**
```bash
cd ppe_detection
pip install opencv-python ultralytics torch numpy streamlit
```

**General Dependencies:**
```bash
pip install opencv-python ultralytics torch torchvision numpy streamlit pillow
```

---

## 🚀 Usage

### 1. Core AI Camera System

#### Run the detection script:
```bash
cd Core_model_1
python ai_camera_system.py
```

#### Or use the Streamlit web interface:
```bash
cd Core_model_1
streamlit run streamlit_rtsp_app.py
```

This launches an interactive web dashboard at `http://localhost:8501`

### 2. Face Recognition Dashboard

```bash
cd AI_FACE_DASHBOARD_FINAL/register_dashboard
python app.py
```

Access the registration interface to add new faces to the database.

### 3. PPE Detection

```bash
cd ppe_detection
python app.py
```

Monitors video feed for PPE compliance (helmet, mask, vest detection).

### 4. Individual Detection Modules

Each module can be run independently:

```bash
# Loitering Detection
cd loitering\ detection
python app.py

# Intrusion Detection
cd intrusion\ detection
python app.py

# Line Crossing Detection
cd Line_crossing
python app.py

# Region Entrance/Exit Monitoring
cd Region_Entrance_Exit
python app.py
```

---

## 🔧 Modules Overview

### Core_model_1
The foundation of the system providing real-time person detection and tracking capabilities using YOLOv8.

**Key Files:**
- `ai_camera_system.py` - Main detection engine
- `streamlit_rtsp_app.py` - Web-based monitoring dashboard
- `Core_Model_1.pt` - Pre-trained model weights

### AI_FACE_DASHBOARD_FINAL
Comprehensive face recognition system for identifying and registering individuals.

**Features:**
- Face detection and extraction
- Face encoding and comparison
- Registration dashboard for new individuals
- Real-time recognition in video streams

### PPE Detection
Specialized module for detecting personal protective equipment in video feeds.

**Detects:**
- Safety helmets
- Face masks
- Protective vests
- Gloves (optional)

### Loitering Detection
Identifies individuals who remain in a specific region for abnormally long periods.

### Intrusion Detection
Alerts when unauthorized personnel enter restricted zones.

### Line Crossing Detection
Tracks when objects or people cross defined virtual lines in the video feed.

### Region Entrance/Exit
Monitors entries and exits in specific geographical areas of interest.

### HeatMap
Generates visual heatmaps showing crowd density and concentration areas.

---

## ⚙️ Configuration

### Video Input Sources
The system supports multiple input types:

```python
# Webcam
cap = cv2.VideoCapture(0)

# IP Camera (RTSP)
cap = cv2.VideoCapture("rtsp://username:password@camera-ip:port/path")

# Video File
cap = cv2.VideoCapture("path/to/video.mp4")
```

### Model Parameters
Adjust detection sensitivity and performance in your scripts:

```python
from ultralytics import YOLO

model = YOLO('Core_model_1.pt')

# Confidence threshold (0.0-1.0)
results = model.predict(frame, conf=0.5)

# IoU threshold
results = model.predict(frame, iou=0.45)
```

### Video Resolution & FPS
Modify video capture properties:

```python
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)
```

---

## 📊 Example Output

The system provides:
- **Bounding boxes** around detected persons
- **Unique ID labels** for tracking
- **Confidence scores** for detections
- **Real-time FPS** metrics
- **Crowd density heatmaps**
- **Alert notifications** for security events
- **Dashboard visualizations** via Streamlit

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is open source and available under the MIT License.

---

## 📞 Support & Contact

For questions, issues, or suggestions:
- Open an issue on GitHub
- Contact: [Your contact information if desired]

---

## 🔗 Resources

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)

---

## ⚡ Performance Tips

1. **GPU Acceleration** - Install CUDA and PyTorch GPU version for faster inference
2. **Resolution Adjustment** - Lower resolution = faster processing but less detail
3. **Batch Processing** - Process multiple frames in batches for better throughput
4. **Model Optimization** - Consider using quantized or pruned models for edge devices
5. **Multi-threading** - Separate detection from display for smoother performance

---

**Last Updated:** 2026  
**Version:** Phase 1.0  
**Status:** Active Development
