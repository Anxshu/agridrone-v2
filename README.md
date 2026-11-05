# 🌾 AgriDrone v2.0 — Autonomous Precision Agriculture & Aerial Crop Pathology Platform

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B.svg)](https://streamlit.io/)
[![Hardware](https://img.shields.io/badge/Hardware-ESP32%20%7C%20ESP32--CAM%20%7C%20DHT22-brightgreen.svg)](https://www.espressif.com/)
[![ML Model](https://img.shields.io/badge/ML%20Accuracy-94.37%25%20(5--Fold%20CV)-success.svg)](https://scikit-learn.org/)
[![Explainable AI](https://img.shields.io/badge/XAI-Grad--CAM%20Decision%20Heatmaps-orange.svg)](#-explainable-ai--grad-cam-saliency)

**AgriDrone v2.0** is an end-to-end aerial precision agriculture system combining **UAV telemetry, high-definition optical canopy surveillance, real-time microclimate IoT sensing, and explainable AI disease classification**.

---

## 📸 System Showcase & Visual Architecture

### Hardware & Telemetry Wiring Architecture
![AgriDrone Wiring Diagram](photos/wiring_diagram.png)

### Explainable AI — Grad-CAM Foliar Decision Saliency
The platform eliminates the "black-box" nature of plant diagnosis by projecting multi-spectral gradient activations back onto the crop photo, pinpointing active lesion centers and chlorotic fungal halos.

| Original Aerial Drone Capture | Grad-CAM AI Attention Heatmap |
| :---: | :---: |
| ![Original Crop](photos/aerial_crop_sample.jpg) | ![Grad-CAM Heatmap](photos/gradcam_decision_heatmap.jpg) |

---

## 🌟 Key Platform Features

1. **Dual-Microcontroller UAV Payload**:
   - **Payload A (ESP32-CAM)**: AI-Thinker module with OV2640 2MP lens streaming MJPEG video (:81) and serving high-speed HTTP frame capture (:80).
   - **Payload B (ESP32 Sensor Node)**: Digital thermohygrometer (DHT22/DHT11) broadcasting real-time ambient temperature (°C) and relative humidity (%) over HTTP REST (:5000/sensor).

2. **Ultra-Stable Double-Buffered Video Player**:
   - Pure HTML5 `<canvas>` rendering engine with automatic watchdog reconnection.
   - Eliminates the classic 3-second lwIP TCP socket overflow and broken image icons.

3. **Dual-Engine Aerial Crop Classifier (94.37% Accuracy)**:
   - Calibrated 150-estimator Random Forest ensemble trained on 24 foliar spectral markers (NDVI, Excess Green, VARI, brown necrosis, yellow chlorosis, Sobel texture roughness).
   - Built-in heuristic vision pipeline providing zero-crash offline fallback.

4. **3-Tier Actionable Agronomic Guidance**:
   - Separates general health state (🟢 Healthy, 🟡 Need Attention, 🔴 Unhealthy) from specific disease classification.
   - Provides immediate, actionable guidance for all 12 supported classes:
     - 🛡️ **Ways to Prevent (Preventive Cultural Measures)**
     - ⚠️ **Precautions to Take (Immediate Safeguards & Containment)**
     - 💊 **How to Solve It (Curative Treatment Protocols & Bio-Controls)**

5. **Information Disclosure Control**:
   - Top predicted disease is **always visible**.
   - Raw percentages and the 11-class distribution are hidden by default to avoid field clutter and revealed on-demand via the **Show Details** toggle.

6. **Interactive GPS Field Map**:
   - Leaflet.js interactive boundary mapping with real-time polygon area calculation ($m^2$, acres, hectares, perimeter).
   - Click-to-add, draggable waypoint markers, and live UAV telemetry coordinates.

7. **One-Click Agronomic Reporting**:
   - Instant generation of comprehensive diagnostic reports (text and PDF formats) logging microclimate conditions, vegetative health index, and treatment protocols.

---

## 📁 Repository Structure

```
├── app.py                            # Streamlit Ground Control Station Dashboard (5 Workstation Tabs)
├── predict.py                        # ML Inference Engine (24 Vision Features + Dual Fallback)
├── train_model.py                    # Model Training Pipeline (Stratified 5-Fold Cross-Validation)
├── recevier.py                       # Flask REST Telemetry Server (Port 5000)
├── crop_health_model.pkl             # Pre-trained Calibrated Model Binary (94.37% Accuracy)
├── class_labels.json                 # Target Crop Pathology Class Index (12 Classes)
├── requirements.txt                  # Python Dependencies
├── Start Dashboard.bat               # Windows One-Click Application Launcher
├── camera_config.json                # Camera IP Persistence Store
├── field_markers.json                # GeoJSON Field Boundary Coordinates
├── sensor_data.json                  # Real-Time Telemetry Cache
├── wiring_diagram.png                # Full Hardware Wiring Architecture Diagram
├── AgriDrone_v2_Project_Report.pdf   # Complete Engineering Project Report (PDF)
├── PROJECT_REPORT.md                 # Full Technical Architecture & Research Report (Markdown)
├── esp32_code/                       # Onboard Microcontroller Firmware
│   ├── esp32_cam/
│   │   └── esp32_cam.ino             # ESP32-CAM (OV2640) Video & Snapshot Firmware
│   └── esp32_sensor_node/
│       └── esp32_sensor_node.ino     # ESP32 DHT22/DHT11 Telemetry Firmware
└── photos/                           # System Photos & Showcase Assets
    ├── wiring_diagram.png
    ├── model_training_history.png
    ├── gradcam_decision_heatmap.jpg
    └── aerial_crop_sample.jpg
```

---

## ⚡ Quick Start Guide

### 1. Prerequisites
- **Python**: Version 3.10 – 3.12 recommended.
- **Hardware**: AI-Thinker ESP32-CAM and ESP32 Dev Board + DHT22/DHT11 sensor.

### 2. Installation
Clone the repository and set up a Python virtual environment:
```bash
git clone https://github.com/YOUR_USERNAME/AgriDrone.git
cd AgriDrone

python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

pip install -r requirements.txt
```

### 3. Flashing ESP32 Microcontrollers
1. Open `esp32_code/esp32_cam/esp32_cam.ino` in Arduino IDE.
2. Enter your Wi-Fi SSID and Password:
   ```cpp
   const char* ssid     = "YOUR_WIFI_SSID";
   const char* password = "YOUR_WIFI_PASSWORD";
   ```
3. Connect **IO0 to GND**, flash the firmware, remove the jumper, and restart.
4. Repeat for `esp32_code/esp32_sensor_node/esp32_sensor_node.ino`.

### 4. Running the Ground Control Station
**Option A: Windows Batch File (Easiest)**
Double-click `Start Dashboard.bat` to launch both the telemetry server and the dashboard.

**Option B: Manual Terminal Execution**
Terminal 1 (Sensor Telemetry Receiver):
```bash
python recevier.py
```
Terminal 2 (Streamlit Ground Control Station):
```bash
streamlit run app.py --server.port 8501
```
Open **`http://localhost:8501`** in your browser.

---

## 📊 Supported Pathological Classes

| Class ID | Disease Category | Target Crop | Primary Symptoms |
| :---: | :--- | :--- | :--- |
| **0** | **Healthy** | *All Crops* | Optimal chlorophyll density, normal NDVI/ExG |
| **1** | **Tomato - Early Blight** | *Solanum lycopersicum* | Concentric target rings (*Alternaria solani*) |
| **2** | **Tomato - Late Blight** | *Solanum lycopersicum* | Water-soaked necrotic lesions (*Phytophthora*) |
| **3** | **Tomato - Leaf Mold** | *Solanum lycopersicum* | Olive-green velvety mold (*Passalora fulva*) |
| **4** | **Tomato - Yellow Leaf Curl** | *Solanum lycopersicum* | Marginal chlorosis, upward curling (TYLCV) |
| **5** | **Corn - Common Rust** | *Zea mays* | Cinnamon-red erupting pustules (*Puccinia*) |
| **6** | **Corn - Northern Leaf Blight** | *Zea mays* | Elliptical tan lesions (*Exserohilum turcicum*) |
| **7** | **Potato - Early Blight** | *Solanum tuberosum* | Angular brown spots with chlorotic yellow halos |
| **8** | **Potato - Late Blight** | *Solanum tuberosum* | Rapid dark decay, white fungal sporulation |
| **9** | **Apple - Black Rot** | *Malus domestica* | Reddish-purple spots expanding into dark rings |
| **10** | **Grape - Black Rot** | *Vitis vinifera* | Necrotic lesions with black pycnidia pimples |
| **11** | **Grape - Leaf Blight** | *Vitis vinifera* | Brown dead foliar patches, premature leaf drop |

---

## 📄 License & Attribution
Developed for Precision Agriculture & Environmental Robotics Research.
Open-source under the [MIT License](LICENSE).

