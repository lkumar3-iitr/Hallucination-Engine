#!/usr/bin/env python3
r"""
make_gtmask_renderer_copy.py

Make a WEB-local snapshot of the CURRENT local production renderer.

Source:
    D:\HallucinationEngine\driving_models\common\he_camera_renderer.py

Output:
    D:\HallucinationEngine\web\he_camera_renderer_gtmask_oracle_v1.py

The production file is never modified.

Why patch the copy?
-------------------
The production renderer lives under:
    repo\driving_models\common\

and uses:
    THIS_FILE.parents[2]

to locate the repository root.

The copied file lives under:
    repo\web\

so the copied file must instead use:
    THIS_FILE.parent.parent

No renderer logic is changed.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(r"D:\HallucinationEngine"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            r"D:\HallucinationEngine\web\he_camera_renderer_gtmask_oracle_v1.py"
        ),
    )
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    source = (
        repo_root
        / "driving_models"
        / "common"
        / "he_camera_renderer.py"
    )

    if not source.is_file():
        raise FileNotFoundError(
            f"Production renderer not found: {source}"
        )

    original = source.read_text(encoding="utf-8")
    source_sha = sha256_file(source)

    # Patch only repo-root lookup expressions in the copied file.
    patched, count = re.subn(
        r"THIS_FILE\s*\.parents\[2\]",
        "THIS_FILE.parent.parent",
        original,
    )

    if count == 0:
        raise RuntimeError(
            "Could not find THIS_FILE.parents[2] in the production renderer. "
            "The local renderer layout may have changed."
        )

    provenance = (
        "\n\n"
        "# ======================================================================\n"
        "# WEB SNAPSHOT PROVENANCE\n"
        "# ======================================================================\n"
        f"# source_path = {source}\n"
        f"# source_sha256 = {source_sha}\n"
        f"# repo_root_lookup_replacements = {count}\n"
        "# production_source_modified = False\n"
        "# ======================================================================\n"
    )

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        patched + provenance,
        encoding="utf-8",
    )

    print("=" * 96)
    print("CURRENT HE RENDERER SNAPSHOT CREATED")
    print("=" * 96)
    print("[source]       ", source)
    print("[source sha256]", source_sha)
    print("[output]       ", output)
    print("[path patches] ", count)
    print("[production]   unchanged")
    print("=" * 96)


if __name__ == "__main__":
    main()
