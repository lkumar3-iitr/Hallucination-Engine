import json

from he_runtime_adapter import HEPlacementRuntimeAdapter


def main():
    adapter = HEPlacementRuntimeAdapter(
        checkpoint_path="outputs/v1_mlp_combined/heplacement_v1_mlp_best.pt",
        image_width=1280,
        image_height=720,
        fov=90.0,
        device="cpu",
    )

    examples = [
        {"rel_x": 0.0, "rel_z": 60.0, "rel_yaw": 180.0},
        {"rel_x": 0.0, "rel_z": 35.0, "rel_yaw": 180.0},
        {"rel_x": 0.0, "rel_z": 10.0, "rel_yaw": 180.0},
        {"rel_x": -3.5, "rel_z": 20.0, "rel_yaw": 180.0},
        {"rel_x": 3.5, "rel_z": 20.0, "rel_yaw": 180.0},
    ]

    for state in examples:
        rect = adapter.predict_from_adversary_state(state)

        print()
        print("[STATE]")
        print(json.dumps(state, indent=2))

        print("[SPRITE RECT]")
        print(json.dumps(rect, indent=2))


if __name__ == "__main__":
    main()