"""Generate pose-to-sprite labels from the exact visual-hull selector."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from pathlib import Path

import carla
import numpy as np

from he_renderer.renderer import aimed_camera
from he_renderer.selector import Selector


_SELECTOR = None
_ACTOR = None
_INCLUDE_MASKS = False


def _initialize_worker(bank, cache, include_masks):
    global _SELECTOR, _ACTOR, _INCLUDE_MASKS
    _SELECTOR = Selector(bank, cache)
    _SELECTOR.build_hull()
    _ACTOR = carla.Transform()
    _INCLUDE_MASKS = include_masks


def _label_pose(sample):
    bearing, distance, elevation = sample
    bearing_rad = np.deg2rad(bearing)
    elevation_rad = np.deg2rad(elevation)
    horizontal = distance*np.cos(elevation_rad)
    location = _SELECTOR.center + np.array([
        horizontal*np.cos(bearing_rad),
        horizontal*np.sin(bearing_rad),
        distance*np.sin(elevation_rad),
    ])
    camera = carla.Transform(carla.Location(*map(float, location)))
    virtual, _ = aimed_camera(
        _ACTOR, camera, _SELECTOR.center, _SELECTOR.extents
    )
    result = _SELECTOR.select(_ACTOR, virtual, 1280, 720, 90.0)
    label = int(result["index"])
    if _INCLUDE_MASKS:
        return label, np.packbits(result["predicted_mask"], axis=None)
    return label


def stratified_samples(count, seed, minimum_distance, maximum_distance,
                       minimum_elevation, maximum_elevation):
    rng = np.random.default_rng(seed)
    unit = (np.arange(count, dtype=np.float64) + rng.random(count))/count
    bearing = (360*unit[rng.permutation(count)]) % 360
    log_distance = (
        np.log(minimum_distance)
        + unit[rng.permutation(count)]
        * (np.log(maximum_distance)-np.log(minimum_distance))
    )
    elevation = (
        minimum_elevation
        + unit[rng.permutation(count)]*(maximum_elevation-minimum_elevation)
    )
    return np.column_stack((bearing, np.exp(log_distance), elevation))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30000)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--seed", type=int, default=20260908)
    parser.add_argument("--minimum-distance", type=float, default=3.0)
    parser.add_argument("--maximum-distance", type=float, default=100.0)
    parser.add_argument("--minimum-elevation", type=float, default=-20.0)
    parser.add_argument("--maximum-elevation", type=float, default=30.0)
    parser.add_argument("--include-masks", action="store_true")
    args = parser.parse_args()

    bank_identity = Selector(args.bank, args.cache)

    samples = stratified_samples(
        args.samples, args.seed, args.minimum_distance, args.maximum_distance,
        args.minimum_elevation, args.maximum_elevation,
    )
    context = mp.get_context("spawn")
    with context.Pool(
        args.workers,
        initializer=_initialize_worker,
        initargs=(args.bank, args.cache, args.include_masks),
    ) as pool:
        results = list(pool.imap(_label_pose, map(tuple, samples), chunksize=32))
    if args.include_masks:
        labels = np.asarray([result[0] for result in results], dtype=np.int32)
        packed_masks = np.stack([result[1] for result in results])
    else:
        labels = np.asarray(results, dtype=np.int32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema": "he_selector_teacher_v1",
        "bank": str(args.bank.resolve()),
        "samples": len(samples),
        "seed": args.seed,
        "distance_range_m": [args.minimum_distance, args.maximum_distance],
        "elevation_range_deg": [args.minimum_elevation, args.maximum_elevation],
        "bank_fingerprint": bank_identity.fingerprint,
    }
    arrays = {
        "features": samples.astype(np.float32),
        "labels": labels,
        "metadata": json.dumps(metadata, sort_keys=True),
    }
    if args.include_masks:
        arrays["packed_masks"] = packed_masks
    np.savez_compressed(args.output, **arrays)
    print(json.dumps({**metadata, "output": str(args.output),
                      "unique_labels": int(len(np.unique(labels)))}, indent=2))


if __name__ == "__main__":
    main()
