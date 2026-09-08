@echo off
setlocal
cd /d "%~dp0.."
python -m unittest discover -s astra -p test_*.py
if errorlevel 1 exit /b 1
python astra/evaluate_passby.py --reference web/carla_reference/static_left_tesla_ego_overtakes_200 --background web/background_reference/static_left_tesla_ego_overtakes_200 --output astra/artifacts/reproduced_passby --end 100
if errorlevel 1 exit /b 1
