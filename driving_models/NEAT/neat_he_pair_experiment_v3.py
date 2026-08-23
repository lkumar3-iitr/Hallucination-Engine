"""
neat_he_pair_experiment_v3.py

NEAT <-> HE paired experiment using the production 4320-view
Hallucination Engine renderer.

This file intentionally reuses neat_he_pair_experiment_v2.py unchanged.

Only the HE rendering backend is replaced:

    old:
        angle-only sprite selector
        render_he_actor()

    new:
        4320 view-matrix
        render_he_actor_view_matrix()

NEAT itself, preprocessing, cameras, navigation, control, scenario
execution, safety metrics, and CSV output remain unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


# ============================================================
# Existing validated NEAT experiment
# ============================================================

import neat_he_pair_experiment_v2 as base


# v2 inserts driving_models/common into sys.path during import.
from he_camera_renderer import (
    load_view_matrix_sprite_bank,
    render_he_actor_view_matrix,
)


DEFAULT_VIEW_MATRIX_CSV = Path(
    r"D:\HE_Data\sprite_bank_grabcut"
    r"\tesla_grabcut_view_matrix_full"
    r"\view_matrix.csv"
)


def main():

    # --------------------------------------------------------
    # Parse only v3-specific arguments.
    #
    # Everything else is passed untouched to v2.
    # --------------------------------------------------------

    parser = argparse.ArgumentParser(
        add_help=False
    )

    parser.add_argument(
        "--view-matrix-csv",
        default=str(DEFAULT_VIEW_MATRIX_CSV),
    )

    parser.add_argument(
        "--distance-selection-mode",
        choices=[
            "linear",
            "log",
            "inverse_depth",
        ],
        default="linear",
    )

    parser.add_argument(
        "--he-bottom-y-offset-px",
        type=float,
        default=0.0,
    )

    v3_args, remaining = parser.parse_known_args()

    view_matrix_csv = Path(
        v3_args.view_matrix_csv
    ).resolve()

    if not view_matrix_csv.exists():
        raise FileNotFoundError(
            f"View-matrix CSV not found: "
            f"{view_matrix_csv}"
        )

    # --------------------------------------------------------
    # Production view-matrix configuration.
    #
    # vertical_mode=state_y means camera_height_m is
    # informational; actual camera_tf is supplied separately
    # for front / left / right.
    # --------------------------------------------------------

    production_sprite_bank = {
        "mode": "view_matrix",

        "view_matrix_csvs": [
            str(view_matrix_csv)
        ],

        "target_height_m": 0.75,

        "vertical_mode": "state_y",

        # NEAT native cameras are mounted at z=2.3 m.
        "camera_height_m": 2.3,

        "distance_selection_mode":
            v3_args.distance_selection_mode,
    }

    view_matrix = (
        load_view_matrix_sprite_bank(
            production_sprite_bank
        )
    )

    print()
    print(
        "[NEAT HE v3] production 4320 renderer enabled"
    )
    print(
        f"[NEAT HE v3] view matrix: "
        f"{view_matrix_csv}"
    )
    print(
        f"[NEAT HE v3] distance mode: "
        f"{v3_args.distance_selection_mode}"
    )
    print(
        f"[NEAT HE v3] bottom-y offset: "
        f"{v3_args.he_bottom_y_offset_px:+.3f} px"
    )
    print()

    # --------------------------------------------------------
    # Compatibility renderer.
    #
    # This has exactly the old render_he_actor() call
    # signature expected by neat_he_pair_experiment_v2.py.
    #
    # v2 will call this independently for:
    #
    #   front
    #   left
    #   right
    #
    # Each call supplies that camera's world transform.
    # --------------------------------------------------------

    def render_he_actor_v3(
        base_rgb,
        actor_tf,
        camera_tf,
        dimensions,
        sprite_bank,
        available_angles,
        sprite_cache,
        width,
        height,
        fov,
    ):

        # Old sprite_bank and available_angles are deliberately
        # ignored. They exist only because v2 expects them.

        return render_he_actor_view_matrix(
            base_rgb=base_rgb,

            actor_tf=actor_tf,

            camera_tf=camera_tf,

            dimensions=dimensions,

            sprite_bank=production_sprite_bank,

            view_matrix=view_matrix,

            sprite_cache=sprite_cache,

            width=width,

            height=height,

            fov=fov,

            bottom_y_offset_px=(
                v3_args.he_bottom_y_offset_px
            ),
        )

    # --------------------------------------------------------
    # Monkey-patch ONLY the rendering backend.
    # --------------------------------------------------------

    base.render_he_actor = (
        render_he_actor_v3
    )

    # Prevent v2 from scanning the obsolete angle-only bank.
    #
    # v2 only checks this because its old renderer required it.
    # The returned value is ignored by render_he_actor_v3().
    base.discover_available_sprite_angles = (
        lambda sprite_bank: [0]
    )

    # --------------------------------------------------------
    # Remove v3-specific arguments before v2 parses argv.
    # --------------------------------------------------------

    sys.argv = [
        sys.argv[0]
    ] + remaining

    # --------------------------------------------------------
    # Run validated v2 experiment.
    # --------------------------------------------------------

    base.main()


if __name__ == "__main__":
    main()