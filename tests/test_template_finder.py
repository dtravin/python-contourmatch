import unittest
from math import pi as PI
from ca.contour import Contour
from ca.templatefinder import TemplateFinder, ContourTemplate
from tests import *


class Test(unittest.TestCase):

    def setUp(self):
        self.c = Contour(points=points_house_90)
        self.tf = TemplateFinder()
        self.tf.add_template(TEMPLATE_HOUSE, points_house_90, 27)
        self.tf.add_template(TEMPLATE_ROMB, points_romb, 36)
        self.tf.add_template(TEMPLATE_SHIP, points_ship, 40.5)

    def test_template_creation(self):
        sample = ContourTemplate('test1', points_house_rotated_minus90, 27, self.tf.template_contour_size)
        self.assertEqual(self.tf.template_contour_size, sample.contour.count())

    def test_find_template(self):
        found = self.tf.find_template(points_house_rotated_minus90, 27, name='house_rotated_m90')

        self.assertIsNotNone(found)
        self.assertEqual(TEMPLATE_HOUSE, found.name)
        self.assertEqual(PI, found.angle)

if __name__ == "__main__":
    unittest.main()