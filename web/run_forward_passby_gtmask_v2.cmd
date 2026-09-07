@echo off
setlocal

cd /d D:\HallucinationEngine\web

echo ============================================================
echo STEP 1 - Copy current production renderer
echo ============================================================

python make_gtmask_renderer_copy.py
if errorlevel 1 goto :error

echo.
echo ============================================================
echo STEP 2 - Run stable forward pass-by
echo ============================================================

python render_forward_passby_gtmask_v2.py ^
  --carla-dir D:\HallucinationEngine\web\carla_reference\static_left_tesla_ego_overtakes_200 ^
  --bank D:\HallucinationEngine-asset\HE_v_0.1\assets\sprite_bank_native_production\tesla_model3_native_full_v3 ^
  --output-dir D:\HallucinationEngine\web\outputs\tesla_forward_passby_gtmask_v3 ^
  --frame-start 0 ^
  --frame-end 60 ^
  --frame-step 1 ^
  --distance-selection-mode linear ^
  --viewpoint-lateral-sign 1 ^
  --continuity-weight 0.015 ^
  --mask-dilate-px 7 ^
  --inpaint-radius 4 ^
  --save-frame-step 5

if errorlevel 1 goto :error

echo.
echo Finished successfully.
goto :eof

:error
echo.
echo ERROR: experiment failed.
exit /b 1
