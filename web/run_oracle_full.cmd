@echo off
setlocal

REM ============================================================
REM HE 4320 full-bank silhouette oracle
REM
REM Run this from:
REM     D:\HallucinationEngine\web
REM
REM EDIT ONLY THE THREE PATHS BELOW FIRST.
REM ============================================================

set "CARLA_DIR=D:\HallucinationEngine\REPLACE_WITH_CARLA_REFERENCE_RUN"
set "BANK_DIR=D:\HallucinationEngine\REPLACE_WITH_ORIGINAL_TESLA_4320_BANK"
set "OUT_DIR=D:\HallucinationEngine\web\outputs\tesla_static_overtake_4320_oracle"

python oracle_4320_silhouette.py ^
  --carla-dir "%CARLA_DIR%" ^
  --bank "%BANK_DIR%" ^
  --output-dir "%OUT_DIR%" ^
  --strict-4320 ^
  --canonical-size 160 ^
  --native-refine-k 32 ^
  --top-k 10 ^
  --frame-start 0 ^
  --frame-end -1 ^
  --frame-step 1

endlocal
