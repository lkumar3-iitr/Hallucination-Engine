@echo off
setlocal
cd /d "%~dp0.."
python -m unittest discover -s astra -p test_selector.py
if errorlevel 1 exit /b 1
python astra/evaluate.py --output astra/artifacts/reproduced_existing
if errorlevel 1 exit /b 1
