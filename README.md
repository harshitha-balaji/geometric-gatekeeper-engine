# 🎯 Geometric Gatekeeper Engine (GGE)

> A configurable computer vision pipeline for detecting foreground objects, tracking their movement, and generating directional counting events.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
[![OpenCV](https://img.shields.io/badge/CV-OpenCV-5C3EE8?style=flat-square)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/Numerical-NumPy-013243?style=flat-square)](https://numpy.org/)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

---

## Overview

GGE is a lightweight computer vision project that processes video frames to detect foreground objects, track their movement, and count directional crossings. It uses foreground detection, centroid-based tracking, and a double-line hysteresis gate to determine when an object has crossed a counting region.

The implementation uses conventional image-processing techniques and geometric tracking rather than pretrained object-detection models. It is intended as an exploration of object tracking and directional counting, not as a general-purpose tracking system.

---

## Features

- Foreground object detection using OpenCV
- Centroid-based object tracking using Euclidean distance
- Double-line hysteresis gate for directional crossing detection
- Configurable environment profiles


---
## Pipeline

```text
Video Input
    │
    ▼
Foreground Detection
    │
    ▼
Centroid-Based Object Tracking
    │
    ▼
Hysteresis Gate Evaluation
    │
    ▼
Directional Count Events
```

---

## Tech Stack

- Python
- OpenCV
- NumPy
- JSON Configuration

---
## Project Structure

```text
geometric_gatekeeper_engine/
├── geometric_gatekeeper_engine.py
├── environment_profiles.json
├── engine_settings.json
└── requirements.txt
```

---

## Design

GGE separates foreground detection, object tracking, and crossing evaluation into distinct stages. Environment profiles provide configurable parameters for different scenes, while the tracking logic associates detected objects using centroid positions and distance-based matching.

The current approach relies on foreground extraction and geometric tracking. Its performance therefore depends on the video conditions, object movement, and selected configuration.

---
## Limitations

- Foreground extraction is sensitive to lighting changes and camera movement.
- Centroid tracking may struggle with heavy occlusion or overlapping objects.
- Objects entering or leaving the frame may affect track continuity.
- Tracking accuracy depends on suitable environment-specific configuration.
- Performance has not been established across all scene types and recording conditions.

---
## Future Improvements

- Kalman filter-based motion prediction
- Support for angled and multiple counting gates
- Adaptive gate sizing based on scene conditions
- Integration of object-detection models for crowded environments
- Live analytics dashboard

---
## Sample Video Attribution

Sample traffic footage used for testing was sourced from:

https://www.kaggle.com/code/mateuszk013/vehicle-speed-estimation-with-yolo11-opencv

Credit belongs to the original author and dataset providers.

---

*An exploration of foreground detection, geometric object tracking, and directional counting.*
