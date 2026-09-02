python make_pair_comparison_video.py --run_dir paired_data/


generate pair run - generate_carla_pair_data

final refinerer --   python apply_local_refiner_box_predicted.py --pair_run_dir paired_data/pair_run_YYYY_MM_DD_HH_MM_SS_r00 --checkpoint refiner_checkpoints/local_refiner_YYYY_MM_DD_HH_MM_SS/checkpoints/local_refiner_epoch_050.pth



python build_box_dataset.py



python train_local_refiner.py --dataset_dir refiner_dataset/refiner_combined_2026_05_20_18_32_16 --epochs 50 --batch_size 4 --base_channels 32


python apply_box_predictor.py --pair_run_dir paired_data/pair_run_2026_05_20_20_41_02_r19



refiner_combined_2026_05_20_18_32_16


python apply_residual_refiner_box_predicted.py --pair_run_dir paired_data/pair_run_2026_05_20_20_41_02_r19 --checkpoint refiner_checkpoints/local_refiner_2026_05_20_22_23_45/checkpoints/local_refiner_epoch_025.pth


python apply_box_predictor.py --pair_run_dir paired_data/YOUR_TEST_PAIR_RUN







python run_learned_he_pipeline.py --pair_run_dir paired_data/pair_run_2026_05_23_17_38_47_r04 --box_checkpoint box_predictor_checkpoints/box_predictor_v2_2026_05_20_21_49_04/checkpoints/box_predictor_best.pth --refiner_checkpoint refiner_checkpoints/local_refiner_2026_05_20_22_23_45/checkpoints/local_refiner_epoch_050.pth



python run_pipeline_on_pairdata_folder.py --pairdata_dir paired_data --box_checkpoint box_predictor_checkpoints/box_predictor_v2_2026_05_20_21_49_04/checkpoints/box_predictor_best.pth --refiner_checkpoint refiner_checkpoints/local_refiner_2026_05_20_22_23_45/checkpoints/local_refiner_epoch_050.pth

python run_learned_he_on_video.py --video_path road_video2.mp4 --box_checkpoint box_predictor_checkpoints/box_predictor_v2_2026_05_20_21_49_04/checkpoints/box_predictor_best.pth --refiner_checkpoint refiner_checkpoints/local_refiner_2026_05_20_22_23_45/checkpoints/local_refiner_epoch_050.pth --initial_distance_m 8 --ego_speed_mps 2 --output_dir real_video_he_outputs/test01








sprite generation --- python generate_carla_360_rgba_fixed_camera.py --vehicle-blueprint vehicle.tesla.model3 --vehicle-name vehicle_blue_sedan --angle-step 1 --output-root assets\sprite_bank_rgba --window-mode black --vehicle-semantic-tag 14 --overwrite



python temporal_sprite_insert_fixed_angle.py --input-video "road_video2.mp4" --sprite-path assets\sprite_bank_rgba\vehicle_blue_sedan\rgba\angle_180_rgba.png --output-dir he_outputs\oncoming_fixed_angle_test --start-frame 0 --end-frame 150 --start-height 70 --end-height 260 --center-x-ratio 0.50 --bottom-y-ratio 0.92 --overwrite


Dense close-range sprite bank handoff plan:

- Keep every generated dense bank as its own folder under the cartesian-close asset root. Do not merge PNGs directly into a 4320 render output/scenario folder.
- After a server bank is complete and copied as a `.tar.gz`, extract it locally into `D:\HallucinationEngine-asset\HE_v_0.1\assets\sprite_bank_cartesian_close`.
- Validate each extracted dense bank before using it: expect `30240` RGBA sprites, `30241` `view_matrix.csv` lines, plus `asset_metadata.json` and `generation_config.json`.
- Point 4320 render configs at the extracted bank `view_matrix.csv` files. Treat the 4320 folder as an output/config consumer, not as the storage location for the bank assets.

Example Tesla right-bank extraction:

```powershell
cd D:\HallucinationEngine-asset\HE_v_0.1\assets\sprite_bank_cartesian_close
tar -xzf D:\HallucinationEngine-asset\tesla_model3_cartesian_close_right_dense360_v1.tar.gz

$ASSET_DIR = "D:\HallucinationEngine-asset\HE_v_0.1\assets\sprite_bank_cartesian_close\tesla_model3_cartesian_close_right_dense360_v1"
"RGBA: $((Get-ChildItem "$ASSET_DIR\rgba" -Filter "*_rgba.png" -File).Count) / 30240"
(Get-Content "$ASSET_DIR\view_matrix.csv").Count
Get-Item "$ASSET_DIR\asset_metadata.json", "$ASSET_DIR\generation_config.json"
```

Bus close-range preflight:

- The CARLA 0.9.15 Fuso Rosa blueprint is `vehicle.mitsubishi.fusorosa`, uses
  asset class `bus`, and has bbox-center height `2.1273534297943115 m`.
- To preserve the established close-bank camera height
  (`1.9515125452316284 m`), use `--target-up-m 0.175840884562683`.
- Do not reuse the Tesla/Nissan `0.25-7.0 m` rectangular dense grid for the
  bus. Smoke tests show the camera plane intersects the 10.27 m bus at near
  forward positions; some views are captured from inside the bus.
- Run representative yaws with full debug outputs and
  `--require-all-qa-pass` before starting either production side. File-count
  validation alone does not make a bank usable when rows have QA flags.

Geometry-aware full bus bank:

- Use one bank for both signed lateral sides. The generator first creates
  `candidate_pose_manifest.csv`, `accepted_view_manifest.csv`, and
  `rejected_pose_manifest.csv`.
- With forward positions `0.25-7.0 m`, absolute right offsets
  `3.0, 3.5, 4.0 m`, 360 relative yaws, and `0.10 m` camera clearance, the
  manifest contains `60,480` candidates: `48,956` accepted captures and
  `11,524` intentional camera-inside-bbox rejections.
- Geometry-approved views may be labelled `viewport_clipped`. Their original
  native-capture flags remain in `capture_qa_flags`; hard usable-bank failures
  remain in `qa_flags` and are enforced by `--require-all-qa-pass`.

UB production command after updating the branch and verifying CARLA 0.9.15:

```bash
cd "$HOME/Hallucination-Engine"

nohup conda run --no-capture-output -n he_assetgen \
python HE_v_0.1/generate_carla_cartesian_close_native_mask_v1.py \
--asset-id fuso_rosa_bus_cartesian_close_full_dense360_v1 \
--output-root "$HOME/HallucinationEngine-assets/HE_v_0.1/assets/sprite_bank_cartesian_close" \
--asset-class bus \
--actor-blueprint vehicle.mitsubishi.fusorosa \
--semantic-tag auto \
--color 0,0,255 \
--host 127.0.0.1 \
--port 2000 \
--timeout 60 \
--forward-start-m 0.25 \
--forward-stop-m 7.0 \
--forward-step-m 0.25 \
--right-offsets-m 3.0 3.5 4.0 \
--target-up-m 0.175840884562683 \
--all-relative-yaws \
--yaw-step-deg 1 \
--side-sign 1 \
--geometry-aware-bus \
--bus-camera-clearance-m 0.10 \
--compact-runtime-only \
--require-all-qa-pass \
--resume \
> "$HOME/fuso_rosa_bus_full_dense360_resume_1.log" 2>&1 &

echo "Generator PID: $!"
```

Monitor against the accepted manifest count:

```bash
ASSET_DIR="$HOME/HallucinationEngine-assets/HE_v_0.1/assets/sprite_bank_cartesian_close/fuso_rosa_bus_cartesian_close_full_dense360_v1"

echo "RGBA: $(find "$ASSET_DIR/rgba" -type f -name '*_rgba.png' | wc -l) / 48956"
wc -l "$ASSET_DIR/accepted_view_manifest.csv" "$ASSET_DIR/rejected_pose_manifest.csv"
tail -n 30 "$HOME/fuso_rosa_bus_full_dense360_resume_1.log"
```
