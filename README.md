# 🎯 Geometric Gatekeeper Engine (GGE)

> **Spatial-temporal object tracking and directional boundary pipeline** — detects foreground anomalies across frames, maps continuous identities using Euclidean centroid matrices, and evaluates crossing trajectories through an adaptive hysteresis buffer zone. No deep learning dependencies, pure geometric computer vision architecture.

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
[![CV](https://img.shields.io/badge/CV-OpenCV-5C3EE8?style=flat-square)](https://opencv.org/)
[![Tracking](https://img.shields.io/badge/Tracking-NumPy-013243?style=flat-square)](https://numpy.org/)
[![Architecture](https://img.shields.io/badge/Architecture-Config_Driven-yellow?style=flat-square)](https://github.com/)
![License](https://img.shields.io/badge/license-MIT-green?style=flat-square)
![Status](https://img.shields.io/badge/status-complete-brightgreen?style=flat-square)

---

## What is this?

Most object counting systems rely on fragile single-line triggers that frequently double-count targets when they hover near a threshold. GGE tracks moving foreground structures through a spatial-temporal state machine — maintaining object identity across frames, resolving proximity relationships through Euclidean centroid mapping, and enforcing a two-line hysteresis buffer — before updating directional metrics only when an object completely traverses the gate.

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

- **Geometric anomaly detection** — isolates moving structures using adaptive MOG2 Gaussian mixtures and morphological filtering
- **Euclidean identity assignment** — maps object centroids across sequential frames using optimized distance matrices
- **Double-line hysteresis gating** — prevents bounce-induced double counting through a configurable deadband buffer zone
- **Config-driven profile swapping** — separates environment constraints and operational parameters from engine logic
- **Temporal memory persistence** — maintains identities for temporarily occluded objects before clearing expired tracks
- **Multi-domain support** — switch instantly between highway traffic monitoring and warehouse safety analytics

---

## How It Works

The engine runs three core geometric stages:

### 1. Spatial Anomaly Extraction (The "Detector" Layer)

Six-stage foreground extraction pipeline:

- **Geometric Regularization** — resizes incoming frames to a normalized width for stable spatial calculations
- **Background Subtraction** — applies adaptive MOG2 Gaussian mixture modeling to isolate moving elements
- **Shadow Suppression** — removes gray-level shadow artifacts through binary thresholding
- **Morphological Closing** — performs dilation and erosion passes using an elliptical structural kernel
- **Contour Ingestion** — extracts external object boundaries from cleaned foreground masks
- **Domain Constraint Filtering** — retains only structures that satisfy configured area constraints

### 2. Identity State Assignment (The "Tracker" Layer)

Continuous object association pipeline:

- **Centroid Computation** — calculates the geometric center coordinate of every detected object
- **Distance Matrix Assembly** — computes Euclidean distance relationships between historical and incoming targets
- **Proximity Association** — maps identities using sorted minimum-distance assignments
- **Memory Queue Decay** — maintains disappeared objects temporarily before automatic deregistration

### 3. Hysteresis Evaluation (The "Gatekeeper" Layer)

Directional boundary evaluation pipeline:

- **Buffer Ingestion** — constructs upper and lower safety thresholds around a configurable baseline
- **Orientation Assignment** — records the initial approach direction of tracked objects
- **Breach Verification** — increments metrics only when a target completely crosses the full buffer region
- **State Locking** — prevents duplicate counts by permanently consuming completed traversal states

---

## Project Structure

```text
geometric_gatekeeper_engine/
├── geometric_gatekeeper_engine.py   # Core vision pipeline + interactive runner
├── environment_profiles.json        # Environment-specific boundaries, labels, and thresholds
├── engine_settings.json             # Tracking memory, scaling, and geometric constraints
└── requirements.txt
```

The engine is deliberately config-driven — `geometric_gatekeeper_engine.py` contains only logic, never hardcoded environment values.

---

## Tech Stack

| Layer | Library |
|---------|---------|
| Computer vision operations | `OpenCV` |
| Numerical computation & distance matrices | `NumPy` |
| Configuration loading | `json` (stdlib) |
| System integration | `os` (stdlib) |

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

### Why no ML model?

This project was intentionally built using deterministic geometric computer vision. By relying on spatial relationships and temporal continuity instead of learned representations, the engine remains lightweight, explainable, and capable of real-time execution on modest hardware without requiring GPUs or pretrained models.

### Why config-driven?

Hardcoded spatial thresholds make adaptation difficult. Externalizing environment profiles allows the same tracking engine to operate across multiple deployment scenarios simply by modifying configuration files rather than rewriting code.

---

## Roadmap

- [ ] Multi-line directional gate structures
- [ ] Angled and vertical boundary support
- [ ] Adaptive buffer sizing based on scene depth
- [ ] Kalman filter integration for motion prediction
- [ ] Live metrics export to time-series databases
- [ ] Multi-camera tracking support

---

## Limitations

- Sudden lighting changes, weather shifts, or camera exposure fluctuations can affect foreground extraction quality
- Closely grouped objects may merge into a single contour, causing identity ambiguity
- Fast-moving objects can exceed the configured proximity threshold and trigger tracking discontinuities
- Camera vibration or unstable mounting may introduce centroid drift and counting inaccuracies
- Performance depends on appropriate environment-specific contour area constraints

---

*Built as a deterministic computer vision project to explore foreground extraction, object identity persistence, and geometric boundary analytics in real-time video streams.*
