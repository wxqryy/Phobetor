@echo off
cd /d "%~dp0"
"D:\PhobetorBench\.venv\Scripts\python.exe" -u prepare_data.py
if errorlevel 1 exit /b %errorlevel%
"D:\PhobetorBench\.venv\Scripts\python.exe" -u train.py --micro-batch 16
