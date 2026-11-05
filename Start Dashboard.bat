@echo off
cd /d "%~dp0"
start "Crop Sensor Receiver" cmd /k ""venv\Scripts\python.exe" "recevier.py""
timeout /t 2 /nobreak >nul
start "Crop Dashboard" cmd /k ""venv\Scripts\python.exe" -m streamlit run "app.py" --server.port 8501"
timeout /t 5 /nobreak >nul
start "" "http://localhost:8501"
