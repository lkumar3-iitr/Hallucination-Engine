import csv
import json


EVAL_CSV = (
    r"perception_evaluation\outputs"
    r"\v2_multi_actor_001_yolo11s_actor_eval_occlusion.csv"
)

HE_META = (
    r"he_outputs\v2_multi_actor_001_he\metadata.json"
)

HE_DET = (
    r"perception_evaluation\outputs"
    r"\v2_multi_actor_001_he_yolo11s.jsonl"
)

MIN_IOU = 0.10

VEHICLE_CLASSES = {
    "car",
    "truck",
    "bus",
}


def area(b):
    return max(0, b["x2"] - b["x1"]) * max(
        0, b["y2"] - b["y1"]
    )


def iou(a, b):
    x1 = max(a["x1"], b["x1"])
    y1 = max(a["y1"], b["y1"])
    x2 = min(a["x2"], b["x2"])
    y2 = min(a["y2"], b["y2"])

    inter = (
        max(0, x2 - x1)
        * max(0, y2 - y1)
    )

    union = area(a) + area(b) - inter

    if union <= 0:
        return 0.0

    return inter / union


# ------------------------------------------------------------
# Load disagreement rows
# ------------------------------------------------------------

with open(
    EVAL_CSV,
    encoding="utf-8"
) as f:
    rows = list(
        csv.DictReader(f)
    )


# ------------------------------------------------------------
# Load HE metadata
# ------------------------------------------------------------

with open(
    HE_META,
    encoding="utf-8"
) as f:
    metadata = json.load(f)


he_actors = {}

for frame in metadata["frames"]:

    frame_idx = int(
        frame["frame_idx"]
    )

    he_actors[frame_idx] = {}

    for actor in frame.get(
        "adversaries",
        []
    ):

        paste = actor.get(
            "paste"
        )

        if paste is None:
            continue

        he_actors[frame_idx][
            actor["id"]
        ] = {
            "x1": float(paste["x1"]),
            "y1": float(paste["y1"]),
            "x2": float(paste["x2"]),
            "y2": float(paste["y2"]),
        }


# ------------------------------------------------------------
# Load HE YOLO detections
# ------------------------------------------------------------

he_detections = {}

with open(
    HE_DET,
    encoding="utf-8"
) as f:

    for line in f:

        if not line.strip():
            continue

        record = json.loads(line)

        frame_idx = int(
            record["frame_idx"]
        )

        detections = []

        for det in record.get(
            "detections",
            []
        ):

            if (
                det["class_name"].lower()
                not in VEHICLE_CLASSES
            ):
                continue

            detections.append(det)

        he_detections[
            frame_idx
        ] = detections


# ------------------------------------------------------------
# Diagnose CARLA=yes / HE=no cases
# ------------------------------------------------------------

for actor_id in [
    "adv_cutin",
    "adv_following",
]:

    failures = [
        r
        for r in rows
        if r["actor_id"] == actor_id
        and r["carla_detected"] == "True"
        and r["he_detected"] == "False"
    ]

    candidate_exists = 0
    true_missing = 0

    print()
    print("=" * 72)
    print(actor_id)
    print("=" * 72)

    for row in failures:

        frame_idx = int(
            row["frame_idx"]
        )

        actor_box = (
            he_actors
            .get(frame_idx, {})
            .get(actor_id)
        )

        if actor_box is None:
            continue

        best_iou = 0.0
        best_det = None

        for det in he_detections.get(
            frame_idx,
            []
        ):

            det_box = {
                "x1": float(det["x1"]),
                "y1": float(det["y1"]),
                "x2": float(det["x2"]),
                "y2": float(det["y2"]),
            }

            value = iou(
                actor_box,
                det_box
            )

            if value > best_iou:

                best_iou = value
                best_det = det

        if best_iou >= MIN_IOU:

            candidate_exists += 1
            status = "MATCHING_AMBIGUITY"

        else:

            true_missing += 1
            status = "TRUE_HE_MISS"

        confidence = (
            best_det["confidence"]
            if best_det
            else None
        )

        print(
            f"frame={frame_idx:3d} "
            f"best_iou={best_iou:.3f} "
            f"conf={confidence} "
            f"{status}"
        )

    print()
    print(
        "disagreement frames     :",
        len(failures)
    )

    print(
        "candidate existed       :",
        candidate_exists
    )

    print(
        "true HE misses          :",
        true_missing
    )

    if failures:

        print(
            "ambiguity fraction     :",
            f"{candidate_exists / len(failures):.3f}"
        )