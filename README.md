# agridrone-v2
ESP32-CAM crop monitoring drone with sensor node and Streamlit dashboard
# AgriDrone v2.0

ESP32-CAM based drone for crop monitoring, built as a first-year B.Tech project at Chandigarh University.

## Features
- ESP32-CAM captures crop images from the drone
- GPS tagging using the drone's built-in GPS
- ESP32 ground sensor node with DHT sensor (temperature and humidity)
- Streamlit dashboard to view images and sensor readings

## Tech Used
Python, Streamlit, ESP32, ESP32-CAM, DHT sensor, C/C++ (Arduino framework)

## Folder Structure
- `esp32_code/` : firmware for ESP32-CAM and sensor node
- `dashboard/` : Streamlit app
- `docs/` : project report and diagrams

## How to Run the Dashboard
pip install streamlit
streamlit run dashboard/app.py

## Team
- Shivanshu Battu
- Harshaan virk
