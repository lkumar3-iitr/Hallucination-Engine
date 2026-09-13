import unittest
from types import SimpleNamespace
import numpy as np
from he_renderer.calibrated.exact_selection_cache import ExactSelectionCache


class CacheTests(unittest.TestCase):
    def test_copy_key_invalidation_and_eviction(self):
        class Teacher:
            faces=object()
            packed_masks=object()
            keys=object()
            size=128
            calls=0
            def select(self,*args):
                self.calls+=1
                return {'mask':np.ones((2,2)), 'nested':{'x':[1]}}
        teacher=Teacher()
        cache=ExactSelectionCache(teacher,1)
        pose=SimpleNamespace(get_matrix=lambda:np.eye(4))
        a=cache.select(pose,pose,400,300,100)
        a['mask'][:]=0
        a['nested']['x'].append(2)
        b=cache.select(pose,pose,400,300,100)
        self.assertEqual(teacher.calls,1)
        self.assertEqual(b['mask'].sum(),4)
        self.assertEqual(b['nested']['x'],[1])
        cache.select(pose,pose,401,300,100)
        self.assertEqual(teacher.calls,2)
        cache.select(pose,pose,400,300,100)
        self.assertEqual(teacher.calls,3)
        teacher.faces=object()
        cache.select(pose,pose,400,300,100)
        self.assertEqual(teacher.calls,4)
        cache.clear()
        self.assertFalse(cache.cache)


if __name__=='__main__':
    unittest.main()
