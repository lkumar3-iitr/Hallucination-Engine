import unittest
from unittest.mock import Mock
from collections import OrderedDict
import carla
from astra.native_lookup_candidate import NativeLookup


class LookupCacheTests(unittest.TestCase):
    def test_keys_copy_and_bound(self):
        lookup = NativeLookup.__new__(NativeLookup)
        lookup.selection_cache = OrderedDict()
        lookup._select_uncached = Mock(return_value={'key': [1, 2, 3]})
        actor, camera = carla.Transform(), carla.Transform()
        first = lookup.select(actor, camera, 400, 300, 100)
        first['key'][0] = 9
        self.assertEqual(lookup.select(actor, camera, 400, 300, 100)['key'][0], 1)
        self.assertEqual(lookup._select_uncached.call_count, 1)
        lookup.select(actor, camera, 400, 300, 90)
        self.assertEqual(lookup._select_uncached.call_count, 2)
        for i in range(20):
            lookup.select(actor, carla.Transform(carla.Location(x=i)), 400, 300, 100)
        self.assertEqual(len(lookup.selection_cache), 16)


if __name__ == '__main__':
    unittest.main()
