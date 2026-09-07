@echo off
setlocal

set "CARLA_DIR=D:\HallucinationEngine\web\carla_reference\static_left_tesla_close_yaw_minus60"
set "BANK_DIR=D:\HallucinationEngine-asset\HE_v_0.1\assets\sprite_bank_native_production\tesla_model3_native_full_v3"
set "OUT_DIR=D:\HallucinationEngine\web\outputs\tesla_oracle_close_yaw_minus60"

python oracle_4320_silhouette_v2.py ^
  --carla-dir "%CARLA_DIR%" ^
  --bank "%BANK_DIR%" ^
  --output-dir "%OUT_DIR%" ^
  --strict-4320 ^
  --canonical-size 160 ^
  --native-refine-k 48 ^
  --top-k 10 ^
  --frame-start 35 ^
  --frame-end 65 ^
  --frame-step 1 ^
  --skip-border-touching ^
  --min-box-width 8 ^
  --min-box-height 8 ^
  --diagnostic-step 1

endlocal