import csv
import sys
from pathlib import Path


def main():
    p = Path(sys.argv[1])

    rows = list(
        csv.DictReader(
            open(p, "r", encoding="utf-8")
        )
    )

    suspicious = []

    for r in rows:

        ref_w = float(
            r["he_reference_projected_width_px"]
        )

        vis_w = float(
            r["visible_width_px"]
        )

        ratio = (
            vis_w / ref_w
            if ref_w > 0
            else 999.0
        )

        ids = [
            x
            for x in r[
                "instance_id_pixel_counts"
            ].split(";")
            if x
        ]

        if (
            ratio > 1.5
            or
            len(ids) > 1
        ):
            suspicious.append(
                (ratio, r)
            )

    suspicious.sort(
        key=lambda x: x[0],
        reverse=True,
    )

    print()
    print("=" * 100)
    print(p.parent.name)
    print("=" * 100)

    print(
        "rows       :",
        len(rows),
    )

    print(
        "suspicious :",
        len(suspicious),
    )

    for ratio, r in suspicious:

        print(
            "a=%s d=%s e=%s "
            "ratio=%.3f "
            "visible=%s "
            "ref=%.2f "
            "components=%s "
            "ids=%s "
            "qa=%s "
            "flags=%s "
            "warnings=%s"
            % (
                r["angle_deg"],
                r["distance_m"],
                r["elevation_deg"],
                ratio,
                r["visible_width_px"],
                float(
                    r[
                        "he_reference_projected_width_px"
                    ]
                ),
                r[
                    "connected_component_count"
                ],
                r[
                    "instance_id_pixel_counts"
                ],
                r[
                    "qa_pass"
                ],
                r[
                    "qa_flags"
                ],
                r[
                    "qa_warnings"
                ],
            )
        )


if __name__ == "__main__":
    main()