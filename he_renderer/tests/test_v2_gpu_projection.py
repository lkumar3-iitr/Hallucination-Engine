import unittest
from types import SimpleNamespace

import carla
import numpy as np
import torch

from he_renderer.calibrated.gpu_native_outline import GPUNativeOutline
from he_renderer.calibrated.gpu_native_projection import GPUNativeProjection


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ProjectionTests(unittest.TestCase):
    def teacher(self, middle):
        return SimpleNamespace(size=128,faces=np.array([
            [[10,-1,-1],[10,1,-1],[10,1,1],[10,-1,1]],
            [[10,middle,-1],[10,1,-1],[10,1,1],[10,middle,1]],
        ],dtype=np.float32))

    def test_rounding_guard_and_face_replacement(self):
        teacher=self.teacher(0.)
        candidate=GPUNativeProjection(teacher)
        identity=carla.Transform()
        a=GPUNativeOutline(teacher).predicted_mask(identity,identity,400,300,100)
        b=candidate.predicted_mask(identity,identity,400,300,100)
        np.testing.assert_array_equal(a[0],b[0])
        self.assertEqual(a[1],b[1])
        self.assertEqual(candidate.projection_fallbacks,1)
        teacher.faces=self.teacher(.3).faces
        b=candidate.predicted_mask(identity,identity,400,300,100)
        a=GPUNativeOutline(teacher).predicted_mask(identity,identity,400,300,100)
        np.testing.assert_array_equal(a[0],b[0])
        np.testing.assert_allclose(a[1],b[1],atol=1e-9,rtol=0)
        self.assertEqual(candidate.projection_fallbacks,1)

    def test_near_plane_rejection(self):
        teacher=self.teacher(.3)
        teacher.faces[:,:,0]=.005
        candidate=GPUNativeProjection(teacher)
        with self.assertRaisesRegex(ValueError,'near plane'):
            candidate.predicted_mask(carla.Transform(),carla.Transform(),400,300,100)

    def test_fixed_download_matches_compaction(self):
        teacher=self.teacher(.3)
        reference=GPUNativeProjection(teacher)
        candidate=GPUNativeProjection(teacher,fixed_download=True)
        identity=carla.Transform()
        for scale in (1.,0.):
            teacher.faces[:,:,2] *= scale
            # Replace the array to obey the immutable-face cache contract.
            teacher.faces=teacher.faces.copy()
            if scale == 0:
                teacher.faces[1,:,2]=1.
            a=reference.predicted_mask(identity,identity,400,300,100)
            b=candidate.predicted_mask(identity,identity,400,300,100)
            np.testing.assert_array_equal(a[0],b[0])
            self.assertEqual(a[1],b[1])
        self.assertEqual(candidate.projection_fallbacks,0)

    def test_resident_scoring_ties_and_fallback(self):
        identity=carla.Transform()
        for middle in (.3,0.):
            teacher=self.teacher(middle)
            teacher.keys=np.array([[0.,10.,0.],[0.,10.,0.]])
            teacher.rows=['first','second']
            teacher.path=lambda row: row
            teacher.query=lambda *args: np.array([0.,10.,0.])
            teacher.packed_masks=np.full((2,128*128//8),255,np.uint8)
            reference=GPUNativeProjection(teacher,gpu_raster=True)
            candidate=GPUNativeProjection(teacher,gpu_raster=True,resident_scoring=True)
            a=reference.select(identity,identity,400,300,100)
            b=candidate.select(identity,identity,400,300,100)
            np.testing.assert_array_equal(a.pop('predicted_mask'),b.pop('predicted_mask'))
            self.assertEqual(a,b)
            self.assertEqual(b['index'],0)
            public,_=candidate.predicted_mask(identity,identity,400,300,100)
            self.assertIsInstance(public,np.ndarray)
            self.assertEqual(public.dtype,np.bool_)
            self.assertEqual(candidate.projection_fallbacks,2 if middle==0 else 0)

    def test_resident_requires_raster(self):
        with self.assertRaisesRegex(ValueError,'requires GPU rasterization'):
            GPUNativeProjection(self.teacher(.3),resident_scoring=True)


if __name__=='__main__':
    unittest.main()
