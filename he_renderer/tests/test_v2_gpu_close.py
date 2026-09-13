import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

from he_renderer.calibrated.gpu_close_pipeline import GPUClosePipeline, quantized_sampler, cp


class CoverageTests(unittest.TestCase):
    def test_nodes_and_missing_coverage(self):
        gpu = GPUClosePipeline.__new__(GPUClosePipeline)
        gpu.bank = SimpleNamespace(axes=[[0., 2.], [-2., -1., 1., 2.], [0., 1.]],
            index={(x,y,z):i for i,(x,y,z) in enumerate((x,y,z) for x in (0.,2.) for y in (-2.,-1.,1.,2.) for z in (0.,1.))})
        nodes = gpu.nodes(np.array([1., 1.5, .5]))
        self.assertEqual(len(nodes), 8)
        self.assertAlmostEqual(sum(w for _,w in nodes), 1.)
        self.assertIsNone(gpu.nodes(np.array([1., 0., .5])))
        self.assertIsNone(gpu.nodes(np.array([3., 1.5, .5])))
        del gpu.bank.index[(0.,1.,0.)]
        self.assertIsNone(gpu.nodes(np.array([1., 1.5, .5])))


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class GPUSamplingTests(unittest.TestCase):
    def test_float_source_and_zero_border(self):
        source = np.random.default_rng(5).random((12, 13, 4), dtype=np.float32)
        maps = np.random.default_rng(6).uniform(-2, 14, (8, 10, 2)).astype(np.float32)
        points = np.stack((np.ones((8,10), np.float32), maps[:,:,0], -maps[:,:,1]),axis=-1)
        expected = cv2.remap(source,maps[:,:,0],maps[:,:,1],cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT)
        code = (Path(__file__).resolve().parents[1]/'calibrated'/'gpu_close.cu').read_text()
        options = ('--fmad=false',) + (('-DQUANTIZED_LINEAR',) if quantized_sampler() else ())
        sample = cp.RawKernel(code,'source_sample',options=options)
        result = cp.zeros((8,10,4),cp.float32)
        sample((1,),(128,), (cp.asarray(points),cp.ones(80,cp.uint8),cp.eye(4,dtype=cp.float64),cp.asarray(source),
            np.int32(13),np.int32(12),np.float64(1),np.float64(1),np.float64(0),np.float64(0),
            np.float64(0),np.float64(0),np.float32(1),result,np.int32(80)))
        np.testing.assert_allclose(cp.asnumpy(result),expected,atol=2e-7,rtol=1e-6)

    def test_finish_depth_unknown_and_margin(self):
        code = (Path(__file__).resolve().parents[1]/'calibrated'/'gpu_close.cu').read_text()
        finish = cp.RawKernel(code,'finish',options=('--fmad=false',))
        points = cp.asarray([[3.,0,0]]*5,dtype=cp.float32)
        layer = cp.asarray([[100.,100,100,.5]]*5,dtype=cp.float32)
        background = cp.full((5,3),20,cp.uint8)
        scene = cp.asarray([2.,3.,np.nan,0.,2.96],dtype=cp.float64)
        rgb,alpha,status = cp.empty((5,3),cp.uint8),cp.empty(5,cp.float32),cp.empty(5,cp.uint8)
        finish((1,),(128,), (points,cp.ones(5,cp.uint8),cp.eye(4,dtype=cp.float64),layer,background,
            scene,np.int32(1),rgb,alpha,status,np.int32(5)))
        np.testing.assert_array_equal(cp.asnumpy(alpha),[0,.5,.5,.5,.5])
        np.testing.assert_array_equal(cp.asnumpy(rgb[:,0]),[20,110,110,110,110])


if __name__ == '__main__':
    unittest.main()
