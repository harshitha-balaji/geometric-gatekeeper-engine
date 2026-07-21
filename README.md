# 🎯 Geometric Gatekeeper Engine (GGE)
![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
[![OpenCV](https://img.shields.io/badge/CV-OpenCV-5C3EE8?style=flat-square)](https://opencv.org/)
[![NumPy](https://img.shields.io/badge/Numerical-NumPy-013243?style=flat-square)](https://numpy.org/)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)

> A rule-based computer vision pipeline for object tracking and directional counting using centroid tracking and hysteresis-based gate crossing.

---

## Overview

Many object counting systems rely on a single counting line, making them prone to duplicate counts when objects pause or oscillate near the boundary. GGE addresses this by combining foreground detection, centroid-based object tracking, and a double-line hysteresis gate that records only complete crossings.

The project was built to explore lightweight, geometry-based object tracking without relying on deep learning models.

---

## Features

- Foreground object detection using OpenCV
- Euclidean centroid-based object tracking
- Double-line hysteresis gate for robust directional counting
- Configurable environment profiles
- Fully configuration-driven architecture

---

## Pipeline

```text
Video Input
     │
     ▼
Foreground Detection
     │
     ▼
Object Tracking
     │
     ▼
Hysteresis Gate Evaluation
     │
     ▼
Directional Count Report
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

GGE intentionally follows a deterministic computer vision pipeline built from geometric tracking rather than learned object detectors. Separating environment profiles from the core logic allows the same tracking pipeline to be adapted to different scenes without modifying the implementation.

---

## Limitations

- Foreground extraction is sensitive to lighting and camera movement.
- Centroid tracking can struggle during heavy occlusion or object overlap.
- Performance depends on appropriate environment-specific configuration.

---

## Future Improvements

- Kalman filter motion prediction
- Angled and multi-line counting gates
- Adaptive scene-aware gate sizing
- Detection model integration for crowded environments
- Live analytics dashboard

---

## Sample Video Attribution

Sample traffic footage used for testing was sourced from:

https://www.kaggle.com/code/mateuszk013/vehicle-speed-estimation-with-yolo11-opencv

Credit belongs to the original author and dataset providers.

---

*Built as an exploration of geometry-based object tracking and directional counting.*
