import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from he_renderer.calibrated.fused_traversal import cp


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class ConvexFillTests(unittest.TestCase):
    def test_integer_quads_and_degeneracies(self):
        kernel=cp.RawKernel((Path(__file__).resolve().parents[1]/'calibrated'/'gpu_convex_fill.cu').read_text(), 'convex_fill')
        rng=np.random.default_rng(92177)
        polygons=[]
        for _ in range(2000):
            points=rng.integers(0,64,(4,2),dtype=np.int32)
            hull=cv2.convexHull(points).reshape(-1,2)
            while len(hull)<4:
                hull=np.concatenate((hull,hull[-1:]),axis=0)
            polygons.append(hull)
        for i in range(64):
            polygons.extend([np.array([[0,i],[63,i],[63,i],[0,i]],np.int32),
                             np.full((4,2),i,np.int32)])
        polygons=np.asarray(polygons,np.int32)
        expected=np.zeros((len(polygons),64,64),np.int32)
        for dst,p in zip(expected,polygons):
            cv2.fillConvexPoly(dst,p,1)
        actual=cp.zeros(expected.shape,cp.int32)
        kernel(((len(polygons)+127)//128,),(128,), (cp.asarray(polygons),
            cp.ones(len(polygons),cp.uint8),actual,np.int32(len(polygons)),np.int32(64),np.int32(1)))
        np.testing.assert_array_equal(cp.asnumpy(actual),expected)


if __name__=='__main__':
    unittest.main()
