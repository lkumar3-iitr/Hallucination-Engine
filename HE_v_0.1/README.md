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