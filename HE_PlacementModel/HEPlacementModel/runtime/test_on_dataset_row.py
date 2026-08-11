import argparse
import json

from he_placement_model import HEPlacementModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, default="dataset/v1_straight_combined/labels.jsonl")
    parser.add_argument("--checkpoint", type=str, default="outputs/v1_mlp_combined/heplacement_v1_mlp_best.pt")
    parser.add_argument("--row-index", type=int, default=0)
    args = parser.parse_args()

    with open(args.labels, "r", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]

    row = rows[args.row_index]

    model = HEPlacementModel(args.checkpoint, device="cpu")

    pred = model.predict(
        relative_state=row["relative_state"],
        camera=row["camera"],
    )

    gt = row["target"]

    print("\n[INPUT relative_state]")
    print(json.dumps(row["relative_state"], indent=2))

    print("\n[INPUT camera]")
    print(json.dumps(row["camera"], indent=2))

    print("\n[GT target]")
    print(json.dumps({
        "center_x": gt["center_x"],
        "bottom_y": gt["bottom_y"],
        "box_width": gt["box_width"],
        "box_height": gt["box_height"],
        "visible": gt["visible"],
    }, indent=2))

    print("\n[PRED target]")
    print(json.dumps(pred, indent=2))

    print("\n[ABS ERROR px]")
    print(json.dumps({
        "center_x": abs(pred["center_x"] - gt["center_x"]),
        "bottom_y": abs(pred["bottom_y"] - gt["bottom_y"]),
        "box_width": abs(pred["box_width"] - gt["box_width"]),
        "box_height": abs(pred["box_height"] - gt["box_height"]),
    }, indent=2))


if __name__ == "__main__":
    main()