"""Replay the failed campaign job with pose capture; do not suppress errors."""
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'driving_models/common')]
import numpy as np
from he_renderer.calibrated import gpu_native_warp


def main():
    output = ROOT / 'he_renderer/artifacts/contact_diagnosis_v1'
    output.mkdir(parents=True, exist_ok=False)
    original = gpu_native_warp.aimed_camera

    def probe(actor, camera, center, extents=None):
        try:
            return original(actor, camera, center, extents)
        except ValueError as exc:
            local = (np.asarray(actor.get_inverse_matrix()) @
                     np.array([camera.location.x, camera.location.y, camera.location.z, 1]))[:3]
            def pose(tf):
                return {**{k: getattr(tf.location, k) for k in ('x', 'y', 'z')},
                        **{k: getattr(tf.rotation, k) for k in ('yaw', 'pitch', 'roll')}}
            receipt = dict(error=str(exc), actor=pose(actor), camera=pose(camera),
                           center=np.asarray(center).tolist(), extents=np.asarray(extents).tolist(),
                           camera_local=local.tolist(),
                           clearance=float(np.linalg.norm(local-np.clip(local, center-extents, center+extents))))
            (output / 'failure_pose.json').write_text(json.dumps(receipt, indent=2))
            raise

    gpu_native_warp.aimed_camera = probe
    plan = json.loads((ROOT / 'he_renderer/artifacts/runtime_campaign_v1/PLAN.json').read_text())
    command = list(plan['jobs'][45]['command'])
    command[command.index('--output-root') + 1] = str(output / 'run')
    sys.argv = command[1:]
    runpy.run_path(command[1], run_name='__main__')


if __name__ == '__main__':
    main()
