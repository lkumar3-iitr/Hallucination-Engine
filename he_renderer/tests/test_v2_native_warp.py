import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import carla
import numpy as np
import torch

from he_renderer.calibrated.gpu_native_warp import GPUNativeWarp, cp
from he_renderer.calibrated.passby_renderer import reproject_sprite


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class NativeWarpTests(unittest.TestCase):
    def test_device_handoff_preserves_public_output(self):
        sprite=np.full((16,18,4),255,np.uint8)
        background=np.full((30,40,3),37,np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'sprite.png')
            self.assertTrue(cv2.imwrite(path,sprite))
            selector=SimpleNamespace(center=np.zeros(3),extents=np.ones(3),
                select=lambda *args: dict(predicted_box=[8.,4.,31.,27.],sprite_path=path))
            warp=GPUNativeWarp(SimpleNamespace(selector=selector))
            actor=carla.Transform(carla.Location(x=20))
            camera=carla.Transform(carla.Location(z=2))
            expected=warp.render(background,actor,camera,100)
            actual=warp.render_device(background,actor,camera,100)
            self.assertIsInstance(actual[0],cp.ndarray)
            self.assertIsInstance(actual[1],cp.ndarray)
            np.testing.assert_array_equal(expected[0],cp.asnumpy(actual[0]))
            np.testing.assert_array_equal(expected[1],cp.asnumpy(actual[1]))
            self.assertEqual(expected[2],actual[2])

    def test_half_pixel_mapping_clipping_and_texture_cache(self):
        rng=np.random.default_rng(14)
        sprite=rng.integers(0,256,(16,18,4),dtype=np.uint8)
        sprite[:2,:,3]=0
        background=rng.integers(0,256,(30,40,3),dtype=np.uint8)
        box=[-8.,-4.,24.,29.]
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'sprite.png')
            self.assertTrue(cv2.imwrite(path,sprite))
            warp=GPUNativeWarp(SimpleNamespace())
            source=warp.texture(path)
            self.assertIs(source,warp.texture(path))
            self.assertEqual(warp.bytes,source.nbytes)
            for matrix in (np.eye(3),np.array([[1.,.1,0],[.1,1.,0],[.07,0,-.2]])):
                expected,ea=reproject_sprite(background,sprite,box,matrix)
                image,alpha=cp.empty(background.shape,cp.uint8),cp.empty((30,40),cp.float32)
                warp.kernel(((1200+127)//128,),(128,), (source,np.int32(source.shape[1]),np.int32(source.shape[0]),
                    cp.asarray(matrix),*map(np.float64,box),cp.asarray(background),image,alpha,np.int32(40),np.int32(30)))
                self.assertLessEqual(np.abs(cp.asnumpy(image).astype(int)-expected.astype(int)).max(),1)
                np.testing.assert_allclose(cp.asnumpy(alpha),ea,atol=2e-7,rtol=1e-6)


if __name__=='__main__':
    unittest.main()
