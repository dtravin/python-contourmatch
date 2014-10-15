import unittest
from math import pi as PI
from ca.contour import Contour
from numpy.testing import assert_almost_equal

from tests import *


class TestContourFramework(unittest.TestCase):

    def setUp(self):
        self.c = Contour(points=points_house_90)

    def test_create_contour(self):
        self.assertEqual(6, self.c.count())
        assert_almost_equal(vectors_house_90, self.c.arr)

    def test_contour_abs(self):
        self.assertEqual(72, round(self.c.abs() * self.c.abs()))

    def test_get_points(self):
        points_generated = self.c.get_points((0, 3))
        assert_almost_equal(points_house_90, points_generated)

    def test_scale_2(self):
        c2 = Contour(points=points_house_90)
        c2.scale(2)
        points_generated = c2.get_points((0, 6))
        assert_almost_equal(points_house_90_scaled2, points_generated)

    def test_rotate_180(self):
        c2 = Contour(points=points_house_90)
        c2.rotate(PI)
        points_generated = c2.get_points((6, 3))
        assert_almost_equal(points_house_rotated_minus90, points_generated)

    def test_norm_dot_self(self):
        self.assertEqual(1, self.c.norm_dot(self.c))

    def test_norm_dot_with_180rotation(self):
        c2 = Contour(points=points_house_rotated_minus90)
        self.assertEqual(-1-0j, self.c.norm_dot(c2))
        self.assertTrue(self.c.acf(normalize=True), c2.acf(normalize=True))

    def test_norm_dot_with_shift(self):
        points_shifted_1 = points_house_90[-1:] + points_house_90[:-1]
        c3 = Contour(len(points_shifted_1), points_shifted_1)

        self.assertNotEqual(-1-0j, self.c.norm_dot(c3))
        self.assertTrue(self.c.acf(normalize=True), c3.acf(normalize=True))

    def test_norm_dot_with_scale(self):
        c3 = self.c.copy()
        c3.scale(3)
        self.assertNotEqual(-1-0j, self.c.norm_dot(c3))
        self.assertTrue(self.c.acf(normalize=True), c3.acf(normalize=True))

    '''
    def test_norm_dot_with_self_equalized_up(self):
        c4 = Contour(points=points_house_90)
        c4.equalization(12)
        self.assertEqual(12, c4.count())
        self.assertNotEqual(1+1j, self.c.norm_dot(c4))
        self.assertTrue(self.c.acf(normalize=True), c4.acf(normalize=True))

    def test_norm_dot_with_self_equalized_down(self):
        c4 = Contour(points=points_house_90)
        c4.equalization(12)
        c4.equalization(6)
        self.assertEqual(6, c4.count())
        self.assertNotEqual(1+1j, self.c.norm_dot(c4))
        self.assertTrue(self.c.acf(normalize=True), c4.acf(normalize=True))

    def test_norm_dot_with_rotated180_equalized_up(self):
        c4 = Contour(points=points_house_rotated_minus90)
        c4.equalization(12)
        self.assertEqual(12, c4.count())
        self.assertNotEqual(-1+0j, self.c.norm_dot(c4))
        self.assertTrue(self.c.acf(normalize=True), c4.acf(normalize=True))
    '''

    def test_equalization(self):
        ce = Contour(points=points_unequalized)
        ce.equalization(24)
        self.assertEqual(24, ce.count())
        assert_almost_equal(points_equalized_24, ce.get_points((0,0)))


if __name__ == "__main__":
    unittest.main()