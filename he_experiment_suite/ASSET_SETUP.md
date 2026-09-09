# HE Asset Release

Sprite banks are external, versioned experiment data and are not stored in
Git. Point `--asset-root` at the directory containing:

```text
sprite_bank_native_production/
sprite_bank_cartesian_close_targeted/
```

The release profile requires the four accepted native banks, their guarded
distillation tables (`selector_teacher_v1.npz`), and only the close-range banks
listed in `configs/assets.json`. Experimental banks and rejected generation
candidates are not runtime dependencies.

`assets/asset_release_v1.lock.json` records file counts and byte sizes for the
frozen release, plus SHA-256 hashes for metadata, selector tables, and accepted
view manifests. Preflight verifies these small identity-critical files. The
counts and sizes are archival transfer checks and are not rescanned on every
launch.

Validate an installation before recording:

```powershell
python -m he_experiment_suite validate `
  --asset-root D:/HallucinationEngine-asset/HE_v_0.1/assets
```

Use `--profile native-only` only for diagnostics. It permits missing close
banks and selector tables, but near-field quality and speed will not match the
accepted release configuration.
