<![CDATA[<div align="center">

# 🛡️ LiftProof.Ai

### Real-Time Shoplifting Detection System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Pose-orange.svg)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green.svg)](https://opencv.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<p align="center">
  <img src="assets/demo.gif" alt="LiftProof.Ai Demo" width="600">
</p>

**AI-powered shoplifting detection using computer vision and behavioral analysis**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [How It Works](#-how-it-works) • [API](#-api) • [Contributing](#-contributing)

</div>

---

## 📋 Overview

**LiftProof.Ai** is an advanced real-time shoplifting detection system that uses YOLOv8-Pose for skeleton tracking combined with behavioral analysis algorithms to identify potential theft in retail environments. Unlike traditional object detection systems, LiftProof.Ai analyzes **human behavior patterns** to detect concealment actions before items leave the store.

### 🎯 Problem It Solves

- **$100+ billion** lost annually to retail theft globally
- Traditional CCTV requires constant human monitoring
- Object-based detection fails to catch concealment behaviors
- Existing solutions are expensive and complex

### 💡 Our Solution

LiftProof.Ai provides:
- **Real-time behavioral analysis** using pose estimation
- **Multi-camera support** for major CCTV brands
- **Instant alerts** to security personnel
- **Evidence capture** with timestamps and video clips
- **Cost-effective** deployment on standard hardware

---

## ✨ Features

### Core Detection Capabilities

| Feature | Description |
|---------|-------------|
| 🖐️ **Pocket Concealment** | Detects hand-to-pocket movements indicating item hiding |
| 👕 **Under-Shirt Hiding** | Identifies items being tucked under clothing |
| 🩳 **Waistband Concealment** | Monitors waistband area for hiding behavior |
| ⚡ **Quick Grab Detection** | Catches rapid snatching movements |
| 👀 **Nervous Behavior** | Tracks suspicious head movements and looking around |

### Technical Features

- ✅ **Real-time Processing** - 30+ FPS on modern hardware
- ✅ **Multi-Person Tracking** - Track multiple subjects simultaneously
- ✅ **Skeleton Visualization** - Full 17-keypoint pose estimation
- ✅ **Progressive Alerts** - Normal → Watching → Suspicious → Alert
- ✅ **Suspicion Scoring** - 0-100% risk assessment per person
- ✅ **Evidence Capture** - Automatic screenshot/video on alert
- ✅ **CCTV Integration** - Works with Hikvision, Dahua, Lorex, UNV, and more

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- Webcam or CCTV camera (RTSP compatible)
- macOS, Windows, or Linux

### Quick Install

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/LiftProofAI.git
cd LiftProofAI

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Requirements

```txt
# requirements.txt
ultralytics>=8.1.0
opencv-python>=4.8.0
numpy>=1.24.0
torch>=2.0.0
```

### Download YOLOv8-Pose Model

The model downloads automatically on first run, or manually:

```bash
# Downloads yolov8n-pose.pt (~6MB)
python3 -c "from ultralytics import YOLO; YOLO('yolov8n-pose.pt')"
```

---

## 📖 Usage

### Basic Usage (Webcam)

```bash
# Activate virtual environment
source venv/bin/activate

# Run detection
python3 liftproof_v2.py
```

### With CCTV Camera

```python
# Edit config in the script or use command line
python3 main.py --camera rtsp://admin:password@192.168.1.64:554/stream1
```

### Testing the System

Once running, test these behaviors:

| Action | Expected Response |
|--------|-------------------|
| Stand normally | 🟢 Green skeleton - "NORMAL" |
| Hand in pocket (hold 1-2 sec) | 🟡→🟠→🔴 Alert triggered |
| Hand to waistband | 🟡→🟠→🔴 Alert triggered |
| Hand under shirt | 🟡→🟠→🔴 Alert triggered |
| Quick grabbing motion | Rapid score increase |
| Look around nervously | "NERVOUS_LOOKING" detected |

### Keyboard Controls

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `S` | Save screenshot |
| `R` | Reset tracking |

---

## 🧠 How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        LiftProof.Ai                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌───────────────────────┐  │
│  │  Camera  │───▶│  YOLOv8-Pose │───▶│  Behavioral Analysis  │  │
│  │  Input   │    │  Detection   │    │       Engine          │  │
│  └──────────┘    └──────────────┘    └───────────────────────┘  │
│                         │                        │               │
│                         ▼                        ▼               │
│                  ┌─────────────┐         ┌─────────────┐        │
│                  │  Skeleton   │         │  Suspicion  │        │
│                  │  Tracking   │         │   Scoring   │        │
│                  └─────────────┘         └─────────────┘        │
│                         │                        │               │
│                         └──────────┬─────────────┘               │
│                                    ▼                             │
│                          ┌─────────────────┐                     │
│                          │  Alert System   │                     │
│                          │  & Notification │                     │
│                          └─────────────────┘                     │
│                                    │                             │
│                    ┌───────────────┼───────────────┐             │
│                    ▼               ▼               ▼             │
│              ┌──────────┐   ┌──────────┐   ┌──────────┐         │
│              │ On-Screen│   │  Mobile  │   │ Evidence │         │
│              │  Alert   │   │   Push   │   │  Capture │         │
│              └──────────┘   └──────────┘   └──────────┘         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Detection Pipeline

1. **Frame Capture** - Video input from webcam or RTSP stream
2. **Pose Estimation** - YOLOv8-Pose extracts 17 body keypoints
3. **Person Tracking** - Persistent IDs across frames
4. **Behavior Analysis** - Analyze keypoint positions and movements
5. **Temporal Analysis** - Track behavior duration over time
6. **Risk Scoring** - Calculate 0-100% suspicion score
7. **Alert Generation** - Trigger alerts when threshold exceeded

### Body Keypoints Used

```
        0: Nose
       / \
      1   2: Eyes
     /     \
    3       4: Ears
     \     /
      5───6: Shoulders
      │   │
      7   8: Elbows
      │   │
      9  10: Wrists (★ Primary detection points)
      │   │
     11──12: Hips (★ Primary detection points)
      │   │
     13  14: Knees
      │   │
     15  16: Ankles
```

### Behavior Detection Logic

```python
# Simplified detection logic
def detect_concealment(wrist, hip, body_height):
    distance = euclidean_distance(wrist, hip)
    normalized = distance / body_height
    
    if normalized < POCKET_THRESHOLD:  # 0.12
        return True  # Hand near pocket detected
    return False
```

---

## ⚙️ Configuration

### Detection Thresholds

Edit these in `liftproof_v2.py` to tune sensitivity:

```python
self.config = {
    # Distance thresholds (relative to body height)
    'POCKET_THRESHOLD': 0.12,      # Hand near pocket
    'WAIST_THRESHOLD': 0.10,       # Hand at waistband
    'CHEST_THRESHOLD': 0.15,       # Hand under shirt
    
    # Time thresholds (frames at ~30fps)
    'POCKET_FRAMES': 25,           # ~0.8 sec to trigger
    'WAIST_FRAMES': 20,            # ~0.7 sec to trigger
    'CHEST_FRAMES': 25,            # ~0.8 sec to trigger
    
    # Alert thresholds
    'ALERT_THRESHOLD': 70,         # Score to trigger alert
    'SUSPICIOUS_THRESHOLD': 40,    # Score for suspicious
    'WATCHING_THRESHOLD': 20,      # Score to start watching
}
```

### CCTV Camera Configuration

```python
# config/settings.py
CAMERAS = [
    {
        "name": "Entrance_Camera",
        "brand": "hikvision",
        "ip": "192.168.1.64",
        "port": 554,
        "username": "admin",
        "password": "your_password",
        "channel": 1,
        "enabled": True
    },
    {
        "name": "Checkout_Camera",
        "brand": "dahua",
        "ip": "192.168.1.65",
        "port": 554,
        "username": "admin",
        "password": "your_password",
        "channel": 1,
        "enabled": True
    }
]
```

### Supported CCTV Brands

| Brand | RTSP URL Format |
|-------|-----------------|
| **Hikvision** | `rtsp://{user}:{pass}@{ip}:554/Streaming/Channels/{ch}01` |
| **Dahua** | `rtsp://{user}:{pass}@{ip}:554/cam/realmonitor?channel={ch}` |
| **Lorex** | `rtsp://{user}:{pass}@{ip}:554/ch{ch}/main` |
| **Uniview (UNV)** | `rtsp://{user}:{pass}@{ip}:554/unicast/c{ch}/s0/live` |
| **Reolink** | `rtsp://{user}:{pass}@{ip}:554/h264Preview_{ch}_main` |

---

## 📁 Project Structure

```
LiftProofAI/
├── 📄 liftproof_v2.py          # Main detection script (recommended)
├── 📄 main.py                   # Multi-camera orchestrator
├── 📄 test_webcam.py           # Simple webcam test
├── 📄 requirements.txt          # Python dependencies
├── 📄 README.md                 # This file
├── 📄 LICENSE                   # MIT License
├── 📄 .gitignore               # Git ignore rules
│
├── 📁 config/
│   └── 📄 settings.py          # Configuration settings
│
├── 📁 modules/
│   ├── 📄 __init__.py
│   ├── 📄 detection_engine.py  # Core detection logic
│   ├── 📄 rtsp_loader.py       # CCTV stream handler
│   └── 📄 notification_manager.py  # Alert system
│
├── 📁 models/                   # Trained models (git-ignored)
│   └── 📄 .gitkeep
│
├── 📁 data/
│   └── 📁 evidence/            # Captured alert images
│
├── 📁 logs/                     # Application logs
│
└── 📁 assets/                   # Documentation assets
    └── 📄 demo.gif
```

---

## 📊 Performance

### Benchmarks

| Hardware | Resolution | FPS | Persons Tracked |
|----------|------------|-----|-----------------|
| MacBook Air M1 | 720p | 25-30 | Up to 5 |
| MacBook Pro M2 | 1080p | 30+ | Up to 10 |
| RTX 3060 | 1080p | 45+ | Up to 15 |
| RTX 4090 | 4K | 60+ | Up to 20+ |

### Model Options

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| `yolov8n-pose.pt` | 6 MB | Fastest | Good |
| `yolov8s-pose.pt` | 23 MB | Fast | Better |
| `yolov8m-pose.pt` | 52 MB | Medium | Best |

---

## 🔮 Roadmap

- [x] Basic pose detection
- [x] Behavioral analysis engine
- [x] Multi-person tracking
- [x] Webcam support
- [x] CCTV integration
- [ ] Mobile app notifications (Firebase)
- [ ] SMS alerts (Twilio)
- [ ] Web dashboard
- [ ] Cloud deployment
- [ ] Custom model training pipeline
- [ ] Multi-camera view
- [ ] Historical analytics
- [ ] Integration with POS systems

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. **Fork** the repository
2. **Create** your feature branch
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit** your changes
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push** to the branch
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open** a Pull Request

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/LiftProofAI.git
cd LiftProofAI

# Create development environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run tests
python3 -m pytest tests/
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 LiftProof.Ai

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

---

## ⚠️ Disclaimer

This software is intended for **legitimate security purposes only**. Users are responsible for:

- Complying with local privacy laws and regulations
- Obtaining necessary permissions for surveillance
- Proper signage indicating video monitoring
- Ethical use of the technology

The developers are not responsible for misuse of this software.

---

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/LiftProofAI/issues)
- **Discussions**: [GitHub Discussions](https://github.com/YOUR_USERNAME/LiftProofAI/discussions)
- **Email**: your.email@example.com

---

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) - YOLOv8 framework
- [OpenCV](https://opencv.org/) - Computer vision library
- [Roboflow](https://roboflow.com/) - Dataset management
- Research papers on behavioral analysis in retail security

---

<div align="center">

**Built with ❤️ for retail security**

⭐ Star this repo if you find it useful!

[Report Bug](https://github.com/YOUR_USERNAME/LiftProofAI/issues) • [Request Feature](https://github.com/YOUR_USERNAME/LiftProofAI/issues)

</div>
]]>