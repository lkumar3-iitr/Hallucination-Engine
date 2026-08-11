import os
import argparse

from apply_box_predictor import apply_box_predictor
from apply_residual_refiner_box_predicted import apply_residual_refiner_to_box_predicted
from make_residual_he_comparison_video import make_residual_comparison_video


def find_pair_runs(pairdata_dir):
    """
    Find all pair_run_* folders inside a paired_data directory.
    """
    if not os.path.isdir(pairdata_dir):
        raise RuntimeError(f"Pairdata directory not found: {pairdata_dir}")

    runs = []

    for name in os.listdir(pairdata_dir):
        path = os.path.join(pairdata_dir, name)

        if name.startswith("pair_run_") and os.path.isdir(path):
            metadata_path = os.path.join(path, "metadata", "sequence_metadata.json")

            if os.path.exists(metadata_path):
                runs.append(path)
            else:
                print(f"[PipelineAll] Skipping {path}, missing sequence_metadata.json")

    runs = sorted(runs, key=os.path.getmtime)

    return runs


def run_pipeline_for_one_pair_run(
    pair_run_dir,
    box_checkpoint,
    refiner_checkpoint,
    output_video_name,
    skip_box=False,
    skip_refiner=False,
    skip_video=False,
):
    print("=" * 80)
    print("[PipelineAll] Pair run:", pair_run_dir)
    print("=" * 80)

    if not skip_box:
        print("[PipelineAll] Step 1: Applying box predictor...")
        apply_box_predictor(
            pair_run_dir=pair_run_dir,
            checkpoint_path=box_checkpoint,
        )
    else:
        print("[PipelineAll] Step 1 skipped: box prediction")

    if not skip_refiner:
        print("[PipelineAll] Step 2: Applying residual refiner...")
        apply_residual_refiner_to_box_predicted(
            pair_run_dir=pair_run_dir,
            checkpoint_path=refiner_checkpoint,
        )
    else:
        print("[PipelineAll] Step 2 skipped: residual refiner")

    if not skip_video:
        print("[PipelineAll] Step 3: Making comparison video...")
        make_residual_comparison_video(
            run_dir=pair_run_dir,
            output_name=output_video_name,
        )
    else:
        print("[PipelineAll] Step 3 skipped: video")

    print("[PipelineAll] Finished:", pair_run_dir)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--pairdata_dir",
        type=str,
        default="paired_data",
        help="Directory containing pair_run_* folders.",
    )

    parser.add_argument(
        "--box_checkpoint",
        type=str,
        required=True,
        help="Path to v2 distance-aware box predictor checkpoint.",
    )

    parser.add_argument(
        "--refiner_checkpoint",
        type=str,
        required=True,
        help="Path to residual local refiner checkpoint.",
    )

    parser.add_argument(
        "--output_video_name",
        type=str,
        default="comparison_residual_learned_he.mp4",
        help="Output video name saved inside each pair_run folder.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Use only latest N pair runs.",
    )

    parser.add_argument(
        "--skip_box",
        action="store_true",
        help="Skip box predictor stage and reuse existing he_box_predicted.",
    )

    parser.add_argument(
        "--skip_refiner",
        action="store_true",
        help="Skip residual refiner stage and reuse existing he_box_predicted_residual_refined.",
    )

    parser.add_argument(
        "--skip_video",
        action="store_true",
        help="Skip video creation.",
    )

    args = parser.parse_args()

    pair_runs = find_pair_runs(args.pairdata_dir)

    if args.limit is not None:
        pair_runs = pair_runs[-int(args.limit):]

    if not pair_runs:
        raise RuntimeError(f"No valid pair_run_* folders found in {args.pairdata_dir}")

    print("[PipelineAll] Found pair runs:", len(pair_runs))

    for run in pair_runs:
        print("   ", run)

    for pair_run_dir in pair_runs:
        run_pipeline_for_one_pair_run(
            pair_run_dir=pair_run_dir,
            box_checkpoint=args.box_checkpoint,
            refiner_checkpoint=args.refiner_checkpoint,
            output_video_name=args.output_video_name,
            skip_box=args.skip_box,
            skip_refiner=args.skip_refiner,
            skip_video=args.skip_video,
        )

    print("=" * 80)
    print("[PipelineAll] All pair runs processed.")
    print("=" * 80)


if __name__ == "__main__":
    main()