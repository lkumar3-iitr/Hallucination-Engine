# Experimental ASTRA Source Checkpoint

This checkpoint publishes source, tests, manifests and documentation only.
It does not promote the candidate into the production `he_renderer` backend.
See `RENDERER_LOGIC_AND_OPTIMIZATIONS.md` for the current algorithm and status.

Excluded: generated banks, images/videos, numerical evidence, GPU runtime
packages, compiled caches and frozen local backup ZIPs. Paths into `artifacts/`
in reports refer to local evidence, not files included in this commit.

Existing CARLA/NEAT Python environment and external calibrated native/close
banks are required. `requirements.txt` describes the older base path; the
calibrated path also requires PyTorch. The optional fused experiment uses
workspace-local CuPy/CUDA dependencies documented in `PERFORMANCE_WORK.md`.
It is not automatically selected by the normal renderer.

ASTRA imports repository common recorder/geometry and production renderer
modules. This is not a standalone release or a clean-clone reproducibility
claim: unrelated local modifications to those modules were not included in
this scoped source checkpoint. Reproduction must audit those dependencies
and supply the referenced banks before running CARLA experiments.

Verification in the existing working environment: 29 tests pass with
`python -m unittest discover -s astra -p test_*.py`.
No source bank or prior evidence was deleted.
