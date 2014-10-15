import cv2
import math
from cmath import phase
from math import sqrt
import numpy as np

from ca.contour import Contour

class ContourTemplate:
    preferredAngleNoMore90 = False

    def __init__(self, name, points, sourceArea, templateSize):
        self.name = name
        self.sourceArea = sourceArea
        self.startPoint = points[0]
        self.contour = Contour(points=points)
        self.contour.equalization(templateSize)
        self.contourNorma = self.contour.abs()
        self.autoCorr = self.contour.acf(True)

    def writeToFile(self, directory):
        points = self.contour.get_points((100, 100))
        mask = np.zeros((500, 500), np.uint8)
        i = 0
        for p in points:
            if i % 3 == 0:
                cv2.circle(mask, (p[0]+150, p[1]+150), 1, (255,0,0), -1)
            elif i % 3 == 1:
                cv2.circle(mask, (p[0]+150, p[1]+150), 2, (255,0,0), -1)
            else:
                cv2.circle(mask, (p[0]+150, p[1]+150), 3, (255,0,0), -1)

            i += 1

        cv2.imwrite('%s/template_%s.png' % (directory, self.name), mask)

class FoundTemplateDesc:

    def __init__(self, foundTemplate, rate, sample, angle):
        self.foundTemplate = foundTemplate
        self.rate = rate
        self.sample = sample
        self.angle = angle
        self.name = foundTemplate.name

    def get_scale(self):
        return sqrt(self.sample.sourceArea / self.template.sourceArea)


class TemplateFinder:
    templates = {}
    template_contour_size = 30
    minACF = 0.96
    minICF = 0.80
    checkICF = True
    checkACF = True
    maxRotateAngle = math.pi
    maxACFDescriptorDeviation = 4

    def add_template(self, name, pixels, area):
        tmpl = ContourTemplate(name, pixels, area, self.template_contour_size)
        self.templates[name] = tmpl
        return tmpl

    def match_template_with_contour(self, template_name, pixels, area, sample_name):
        sample = ContourTemplate(sample_name, pixels, area, self.template_contour_size)
        result = self._match_templates(template_name, sample)
        if result:
            return result
        else:
            sample.writeToFile('/tmp/i')
            self.templates[template_name].writeToFile('/tmp/i')


    def _match_templates(self, template_name, sample):
        rate = 0
        angle = 0
        foundTemplate = None

        template = self.templates[template_name]
        r = 0
        if self.checkACF:
            r = abs(template.autoCorr.norm_dot(sample.autoCorr))

        if self.checkICF:
            interCorr = template.contour.icf(sample.contour).find_max_norma()
            r = abs(interCorr) / (template.contourNorma * sample.contourNorma)
            if r < self.minICF:
                return
            _angle = phase(interCorr)

        if r >= rate:
            rate = r
            foundTemplate = template
            angle = _angle

        return angle, foundTemplate, rate

    def find_template(self, pixels, area, name='unknown'):
        foundTemplate = None
        sample = ContourTemplate(name, pixels, area, self.template_contour_size)
        best_rate = 0
        for template_name, template in self.templates.iteritems():
            result = self._match_templates(template_name, sample)
            if result and result[2] > best_rate:
                angle, foundTemplate, rate = result
                best_rate = rate

        return foundTemplate and FoundTemplateDesc(foundTemplate, rate, sample, angle)


