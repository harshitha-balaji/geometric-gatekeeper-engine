# 🎯 Geometric Gatekeeper Engine (GGE)

> **Config-driven object tracking and directional counting engine** — detects moving foreground objects, maintains identity across frames via Euclidean centroid mapping, and enforces a hysteresis gate to count only clean, complete crossings. No deep learning. Pure geometric computer vision.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
[![CV](https://img.shields.io/badge/CV-OpenCV-5C3EE8?style=flat-square)](https://opencv.org/)
[![Tracking](https://img.shields.io/badge/Tracking-NumPy-013243?style=flat-square)](https://numpy.org/)
![Config](https://img.shields.io/badge/architecture-config--driven-yellow?style=flat-square)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)

---

## What is this?

Single-line counting systems double-count objects that hover near a threshold. GGE solves this with a two-line hysteresis gate — an object is counted only when it fully traverses the buffer zone, never when it lingers at the boundary.

Give it a video source and an environment profile, and it tracks every moving object frame-by-frame: assigning identities, persisting them through occlusion, and updating directional counts only on clean, committed crossings.

```text
Available Environment Profiles:
> ['highway_traffic', 'warehouse_safety']

Enter profile name:
> highway_traffic

Enter video source selection:
> sample_video.mp4

═══════════════════════════════════════════════════════════════
         FINAL COUNTS SUMMARY REPORT
═══════════════════════════════════════════════════════════════
 Profile Domain : HIGHWAY_TRAFFIC
 Inbound        : 42
 Outbound       : 29
═══════════════════════════════════════════════════════════════
```

---

## Features

- **Foreground detection** — isolates moving objects using adaptive MOG2 Gaussian mixtures and morphological filtering
- **Euclidean identity assignment** — maps object centroids across sequential frames using vectorized distance matrices
- **Double-line hysteresis gating** — prevents bounce-induced double counting through a configurable deadband buffer zone
- **Hot-swappable environment profiles** — separates spatial constraints and operational parameters from engine logic
- **Temporal memory persistence** — maintains identities for temporarily occluded objects before clearing expired tracks
- **Multi-domain support** — switch instantly between highway traffic monitoring and warehouse safety analytics

---

## How It Works

Three sequential stages process each video frame:

### 1. Foreground Extraction
Six-stage pipeline that isolates moving objects from static background:
- **Frame normalization** — resizes incoming frames to a configured width for stable spatial calculations
- **Background subtraction** — adaptive MOG2 Gaussian mixture modeling isolates moving elements
- **Shadow suppression** — binary thresholding removes gray-level shadow artifacts
- **Morphological closing** — dilation and erosion passes clean foreground masks using an elliptical kernel
- **Contour extraction** — external object boundaries pulled from cleaned foreground masks
- **Area filtering** — retains only contours that satisfy the profile's min/max area constraints

### 2. Identity Tracking
Continuous object association across frames:
- **Centroid computation** — geometric center coordinate calculated for every detected object
- **Distance matrix assembly** — Euclidean distances computed between historical and incoming centroids
- **Proximity association** — identities assigned via sorted minimum-distance matching
- **Memory decay** — disappeared objects held temporarily before automatic deregistration

### 3. Hysteresis Gating
Directional count logic that prevents double-counting:
- **Buffer construction** — upper and lower safety thresholds built around a configurable baseline coordinate
- **Approach direction recording** — initial trajectory of each tracked object logged on first detection
- **Breach verification** — count incremented only when an object fully crosses the complete buffer region
- **State locking** — completed traversal states consumed permanently; re-entry cannot trigger a duplicate count

---

## Project Structure

```text
geometric_gatekeeper_engine/
├── geometric_gatekeeper_engine.py   # Core vision pipeline + interactive runner
├── environment_profiles.json        # Environment-specific boundaries, labels, and thresholds
├── engine_settings.json             # Tracking memory, scaling, and geometric constraints
└── requirements.txt
```

`geometric_gatekeeper_engine.py` contains only logic — never hardcoded environment values. All spatial parameters live in config.

---

## Tech Stack

| Layer | Library |
|-------|---------|
| Computer vision operations | `OpenCV` |
| Numerical computation & distance matrices | `NumPy` |

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the engine

```bash
python geometric_gatekeeper_engine.py
```

### 3. Follow the prompts

```text
Available Environment Profiles:
> ['highway_traffic', 'warehouse_safety']

Enter profile name:
> highway_traffic

Video Source Options:
  - Type '0' to launch your primary live hardware webcam.
  - Type a video file name (e.g. sample_video.mp4).

Enter video source selection:
> sample_video.mp4
```

Press `q` while the display window is active to stop execution.

---

## Configuration

### `engine_settings.json`

| Parameter | What it controls |
|-----------|-----------------|
| `target_width` | Normalized frame width used during preprocessing |
| `bg_history` | Background model history length used by MOG2 |
| `bg_threshold` | Variance threshold controlling foreground extraction sensitivity |
| `memory_limit` | Maximum frame lifespan for disappeared tracked identities |
| `spatial_proximity_threshold` | Distance limit used when associating centroids between frames |
| `buffer_offset` | Half-width of the hysteresis buffer zone surrounding the gate |

### `environment_profiles.json`

Each environment profile contains:
- `min_area` — minimum contour area accepted by the detector
- `max_area` — maximum contour area accepted by the detector
- `threshold_y` — baseline coordinate used to construct the gate
- `labels` — directional labels rendered in the interface and reports

To add a new environment, create a new top-level profile and load it through the engine runner.

---

## Design Decisions

**Why no ML model?**
GGE was built deterministic deliberately — spatial relationships and temporal continuity instead of learned representations. The engine stays lightweight, explainable, and capable of real-time execution on modest hardware without GPUs or pretrained weights. The ceiling of this approach (crowded scenes, severe occlusion) directly motivates why detection models like YOLO exist.

**Why config-driven?**
Hardcoded spatial thresholds make adaptation impossible across environments. Externalising profiles means the same tracking engine handles highway traffic and warehouse safety without touching a line of engine logic.

---

## Roadmap

- [ ] Kalman filter integration for motion prediction under occlusion
- [ ] Angled and vertical boundary support
- [ ] Adaptive buffer sizing based on estimated scene depth
- [ ] Multi-line gate structures for lane-level counting
- [ ] Live metrics export to time-series databases

---

## Limitations

- Sudden lighting changes or camera exposure fluctuations can affect foreground extraction quality
- Closely grouped objects may merge into a single contour, causing identity ambiguity
- Fast-moving objects can exceed the proximity threshold and trigger tracking discontinuities
- Camera vibration or unstable mounting may introduce centroid drift and counting inaccuracies
- Performance depends on appropriate environment-specific contour area constraints

---

## Sample Video Attribution

Sample traffic footage used for testing was sourced from:
https://www.kaggle.com/code/mateuszk013/vehicle-speed-estimation-with-yolo11-opencv

Credit belongs to the original author and dataset providers. Refer to the Kaggle page for licensing and usage terms.

---

*Built deterministic-first. The ceiling of this approach is the reason YOLO exists.*
