# Detecting Moving Objects in DAS Recordings

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-5C3EE8?style=flat-square&logo=opencv&logoColor=white)](https://opencv.org/)
[![Scikit-Image](https://img.shields.io/badge/Scikit--Image-0.21+-3E8ACC?style=flat-square&logo=scikitimage&logoColor=white)](https://scikit-image.org/)

Computer vision pipeline for detecting and tracking moving vehicles in fiber-optic DAS recordings. Built with advanced image processing, Hough Transform line detection, and velocity field analysis.

**[📄 Full Project Report](project-report.pdf)**

---

## 🎯 About the problem and dataset

Distributed Acoustic Sensing (DAS) transforms standard fiber-optic cables into continuous acoustic sensors spanning kilometers. When vehicles pass over or near the cable, they generate distinctive diagonal patterns in space-time visualizations. The challenge is to extract these lines from extremely noisy data. In this project we analyzed three temporal segments from real DAS recordings (May 2024), each spanning ~2 minutes of continuous monitoring. 

---

## 🔧 Technical Approach

### Preprocessing Pipeline
We developed a 7-stage preprocessing system to isolate vehicle lines:
- **Adaptive thresholding** using Li's method for information-preserving binarization
- **Morphological closing** to connect fragmented vehicle tracks interrupted by signal dropouts
- **Percentile-based clipping** (3rd-99th) to handle extreme outliers while preserving signal
- **Image resizing via averaging**  averaging adjacent frames creates square images which are optimal for line detection using Hough lines

### Detection & Validation
- **Hough Transform** with velocity-based filtering to extract only physically realistic vehicle trajectories
- **Multi-level clustering** combining spatial proximity and line thickness measurements
- **Velocity field analysis** using spatiotemporal gradients to validate detected motion patterns

---

## 🚀 Quick Start
1. Clone the repository
   
    ```bash
    git clone https://github.com/jakubradziejewski/detecting-moving-objects-in-das-recordings.git
    ```

2. Download DAS recordings from Google Drive:

   ```bash
   python download_data.py
   ```
3. Run the detection pipeline:

   ```bash
   python main.py
   ```
