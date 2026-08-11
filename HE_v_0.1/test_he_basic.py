import cv2
import numpy as np

from HallucinationEngine import HallucinationEngine, HEConfig


def main():
    config = HEConfig(
        enabled=True,
        scenario_type="wrong_way_vehicle",
        trigger_frame=10,
        duration_frames=80,
        render_mode="box",
        default_box_2d=(520, 260, 700, 500),
    )

    he = HallucinationEngine(config)

    width = 1280
    height = 720

    for frame_id in range(120):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # Make background gray so box is visible
        frame[:] = (40, 40, 40)

        metadata = {
            "frame": frame_id,
            "dt": 1.0 / 20.0,
        }

        modified_frame, updated_metadata = he.process(frame, metadata)

        cv2.imshow("HE Basic Test", modified_frame)

        key = cv2.waitKey(50)
        if key == ord("q"):
            break

        if updated_metadata["hallucination"]["active"]:
            print(updated_metadata["hallucination"])

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()