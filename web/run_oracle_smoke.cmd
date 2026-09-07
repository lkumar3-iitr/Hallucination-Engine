@echo off
setlocal

set "CARLA_DIR=D:\HallucinationEngine\web\carla_reference\static_left_tesla_ego_overtakes_200"

set "BANK_DIR=D:\HallucinationEngine-asset\HE_v_0.1\assets\sprite_bank_native_production\tesla_model3_native_full_v3"

set "OUT_DIR=D:\HallucinationEngine\web\outputs\tesla_oracle_smoke"

python oracle_4320_silhouette.py ^
  --carla-dir "%CARLA_DIR%" ^
  --bank "%BANK_DIR%" ^
  --output-dir "%OUT_DIR%" ^
  --strict-4320 ^
  --canonical-size 160 ^
  --native-refine-k 32 ^
  --top-k 10 ^
  --frames 0 20 40 50 55 60 62 64 66 68 70 72 74

endlocal