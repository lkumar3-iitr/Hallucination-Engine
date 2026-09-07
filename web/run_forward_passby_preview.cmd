@echo off
setlocal

REM Run from D:\HallucinationEngine\web
set "CARLA_DIR=D:\HallucinationEngine\web\carla_reference\static_left_tesla_ego_overtakes_200"
set "BANK_DIR=D:\HallucinationEngine-asset\HE_v_0.1\assets\sprite_bank_native_production\tesla_model3_native_full_v3"
set "OUT_DIR=D:\HallucinationEngine\web\outputs\tesla_forward_passby_preview"

python render_forward_passby_oracle_v1.py ^
  --carla-dir "%CARLA_DIR%" ^
  --bank "%BANK_DIR%" ^
  --output-dir "%OUT_DIR%" ^
  --frame-start 0 ^
  --frame-end 80 ^
  --frame-step 1 ^
  --actor-length-m 4.792 ^
  --actor-width-m 2.163 ^
  --actor-height-m 1.488 ^
  --mask-dilate-px 9 ^
  --inpaint-radius 4 ^
  --save-frame-step 5 ^
  --top-k 5 ^
  --temporal-angle-window-deg 999

endlocal
