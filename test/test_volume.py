import os
import random
import unittest
import shutil, tempfile

import numpy as np

import stk

class Test_Volume(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_constructor(self):
        arr = np.zeros((4, 5, 6)).astype(np.int32)
        vol = stk.Volume(arr)

        np.testing.assert_equal(vol.size, (6, 5, 4))
        np.testing.assert_equal(vol.origin, (0, 0, 0))
        np.testing.assert_equal(vol.spacing, (1, 1, 1))
        np.testing.assert_equal(vol.direction, ((1, 0, 0), (0, 1, 0), (0, 0, 1)))
        np.testing.assert_equal(np.array(vol), arr)

        d = ((10, 0, 0), (0, 11, 0), (0, 0, 12))
        vol = stk.Volume(arr, origin=(1,2,3), spacing=(4,5,6), direction=d)

        np.testing.assert_equal(vol.size, (6, 5, 4))
        np.testing.assert_equal(vol.origin, (1, 2, 3))
        np.testing.assert_equal(vol.spacing, (4, 5, 6))
        np.testing.assert_equal(vol.direction, d)
        np.testing.assert_equal(np.array(vol), arr)

    def test_meta(self):
        vol = stk.Volume()
        vol.origin = (2, 3, 4)
        np.testing.assert_equal(vol.origin, (2, 3, 4))

        vol = stk.Volume()
        vol.spacing = (2, 3, 4)
        np.testing.assert_equal(vol.spacing, (2, 3, 4))

        d = ((10, 0, 0), (0, 11, 0), (0, 0, 12))
        vol = stk.Volume()
        vol.direction = d
        np.testing.assert_equal(vol.direction, d)

    def test_copy_meta_from(self):
        vol1 = stk.Volume()
        vol2 = stk.Volume()

        vol1.origin = (2,3,4)
        vol1.spacing = (5,6,7)
        d = ((10, 0, 0), (0, 11, 0), (0, 0, 12))
        vol1.direction = d

        vol2.copy_meta_from(vol1)

        np.testing.assert_equal(vol2.origin, (2, 3, 4))
        np.testing.assert_equal(vol2.spacing, (5, 6, 7))
        np.testing.assert_equal(vol2.direction, d)
    
    def test_type(self):
        data = np.zeros((5,5,5)).astype(np.float32)
        vol = stk.Volume(data)
        self.assertEqual(vol.type, stk.Type.Float)

        data = np.zeros((5,5,5,3)).astype(np.float32)
        vol = stk.Volume(data)
        self.assertEqual(vol.type, stk.Type.Float3)

    def test_io(self):
        data = np.zeros((5,5,5)).astype(np.float32)
        vol1 = stk.Volume(data, origin=(2,3,4), spacing=(5,6,7))

        f = os.path.join(self.temp_dir, 'test.nrrd')
        stk.write_volume(f, vol1)

        vol2 = stk.read_volume(f)
        np.testing.assert_equal(vol2.origin, vol1.origin)
        np.testing.assert_equal(vol2.spacing, vol1.spacing)
        np.testing.assert_equal(vol2.direction, vol1.direction)
        np.testing.assert_equal(np.array(vol2), np.array(vol1))
