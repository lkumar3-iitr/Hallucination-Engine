from dataclasses import dataclass


@dataclass
class HEConfig:
    """
    Configuration for the Hallucination Engine.
    """

    enabled: bool = True

    # Scenario settings
    scenario_type: str = "wrong_way_vehicle"
    trigger_frame: int = 50
    duration_frames: int = 100

    # Hallucinated object settings
    object_id: str = "he_adv_001"
    object_class: str = "vehicle"

    # Placement settings
    initial_distance_m: float = 25.0
    min_distance_m: float = 5.0
    lateral_offset_m: float = 0.0

    # Motion settings
    relative_speed_mps: float = 5.0

    # Relative velocity settings
    use_ego_speed_for_relative_motion: bool = True
    adversary_speed_mps: float = 5.0
    max_closing_speed_mps: float = 25.0

    # Approximate vehicle dimensions
    vehicle_width_m: float = 1.9
    vehicle_height_m: float = 1.6
    vehicle_length_m: float = 4.5

    # Rendering settings
    render_mode: str = "box"
    box_color: tuple = (255, 0, 0)
    box_thickness: int = 3

    # Sprite rendering settings
    sprite_path: str = "assets/sprites/car_front.png"
    sprite_alpha: float = 1.0
    sprite_vertical_scale: float = 1.15
    sprite_horizontal_scale: float = 1.10
    sprite_y_offset_ratio: float = 0.0
    draw_sprite_debug_box: bool = False

    # Optional scenario-specific sprites
    wrong_way_sprite_path: str = "assets/sprites/car_front.png"
    stopped_vehicle_sprite_path: str = "assets/sprites/car_front.png"

    # Object insertion realism options
    match_brightness: bool = True
    add_shadow: bool = True
    soften_edges: bool = True
    motion_blur: bool = False

    # Fallback box if projection fails
    default_box_2d: tuple = (520, 260, 700, 500)

    # Image size
    image_width: int = 1280
    image_height: int = 720

    # Camera-only HE settings
    road_center_x_ratio: float = 0.5
    horizon_y_ratio: float = 0.45
    spawn_y_ratio: float = 0.48
    target_y_ratio: float = 0.82

    min_box_width: int = 70
    max_box_width: int = 520
    min_box_height: int = 55
    max_box_height: int = 430

    # Visual road estimator settings
    use_visual_road_estimator: bool = True
    road_estimator_roi_y_start_ratio: float = 0.55
    road_estimator_roi_y_end_ratio: float = 0.95
    road_center_smoothing_alpha: float = 0.85

    # Camera-only repeat settings
    repeat_after_finish: bool = True
    repeat_gap_frames: int = 30

    # Camera-only repeat settings
    repeat_after_finish: bool = True
    repeat_gap_frames: int = 30

    # Dynamic road following settings
    road_follow_base_gain: float = 0.12
    road_follow_conf_gain: float = 0.35