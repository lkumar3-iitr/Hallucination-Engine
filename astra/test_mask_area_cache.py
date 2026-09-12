import unittest
import numpy as np
from astra.selector import packed_iou_scores, POPCOUNT_U8


class MaskAreaTests(unittest.TestCase):
    def test_exact_scores(self):
        rng = np.random.default_rng(531)
        for shape in ((7, 9), (64, 64)):
            masks = rng.random((25, *shape)) > .5
            masks[0] = False
            masks[1] = True
            packed = np.packbits(masks.reshape(25, -1), axis=1)
            areas = POPCOUNT_U8[packed].sum(axis=1)
            for target in (np.zeros(shape, bool), np.ones(shape, bool), rng.random(shape)>.5):
                np.testing.assert_array_equal(packed_iou_scores(packed, target),
                                              packed_iou_scores(packed, target, areas))


if __name__ == '__main__':
    unittest.main()
