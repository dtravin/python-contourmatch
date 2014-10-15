import sys

import numpy as np
from math import sqrt
from ca.utils import abs_square, complex_rotate, phase_between


class Contour:

    arr = None
    top_left_point = None

    def __init__(self, capacity=0, points=None, start_idx=0):
        self.init(not isinstance(points, type(None)) and len(points) or capacity, points, start_idx)

    def init(self, count, points, start_idx):
        self.arr = np.empty(count, dtype=complex)
        if not isinstance(points, type(None)):
            minX = points[start_idx][0]
            minY = points[start_idx][1]
            maxX = minX
            maxY = minY
            end_idx = start_idx + count

            for i in range(start_idx, end_idx):
                p1 = points[i]
                if i == end_idx - 1:
                    p2 = points[start_idx]
                else:
                    p2 = points[i+1]
                #p2 = i == end_idx-1 and points[start_idx] or points[i + 1]
                self.arr[i] = complex(p2[0] - p1[0], -p2[1] + p1[1])

                if p1[0] > maxX:
                    maxX = p1[0]
                if p1[0] < minX:
                    minX = p1[0]
                if p1[1] > maxY:
                    maxY = p1[1]
                if p1[1] < minY:
                    minY = p1[1]

            self.boundsX = minX
            self.boundsY = minY
            self.boundsX2 = maxX - minX + 1
            self.boundsY2 = maxY - minY + 1

    def count(self):
        return len(self.arr)

    def copy(self):
        clone = Contour()
        clone.arr = np.copy(self.arr)
        return clone

    def abs(self):
        result = 0
        for i in range(self.count()):
            result += abs_square(self.arr[i])

        return sqrt(result)

    # scalar multiplication
    def dot(self, c, shift=0):
        sumA = 0
        sumB = 0
        ptr1 = 0
        ptr2 = shift
        ptr22 = 0
        ptr3 = c.count() - 1

        p1 = ptr1
        p2 = ptr2
        for i in range(self.count()):
            x1 = self.arr[p1]
            x2 = c.arr[p2]
            sumA += x1.real * x2.real + x1.imag * x2.imag
            sumB += x1.imag * x2.real - x1.real * x2.imag

            p1 += 1
            if p2 == ptr3:
                p2 = ptr22
            else:
                p2 += 1

        return complex(sumA, sumB)

    # Intercorrelcation function (ICF)
    def icf(self, c):
        result = Contour(capacity=self.count())
        for i in range(self.count()):
            result.arr[i] = self.dot(c, i)

        return result

    # Autocorrelation function (ACF)
    def acf(self, normalize):
        count = self.count()/2
        result = Contour(capacity=count)
        maxNormaSq = 0
        for i in range(count):
            result.arr[i] = self.dot(self, i)
            normaSq = abs_square(result.arr[i])
            if normaSq > maxNormaSq:
                maxNormaSq = normaSq

        if normalize:
            maxNormaSq = sqrt(maxNormaSq)
            for i in range(count):
                v = result.arr[i]
                result.arr[i] = complex(v.real/maxNormaSq, v.imag / maxNormaSq)

        return result

    def normalize(self):
        max = abs(self.find_max_norma())
        if max > sys.float_info.min:
            self.scale(1/max)

    def find_max_norma(self):
        max = 0
        res = 0, 0
        for v in self.arr:
            absv = abs(v)
            if absv > max:
                max = absv
                res = v

        return res

    def scale(self, factor):
        for i in range(self.count()):
            self.arr[i] = self.arr[i] * factor

    def rotate(self, angle):
        for i in range(self.count()):
            self.arr[i] = complex_rotate(self.arr[i], angle)

    def norm_dot(self, c):
        sumA = 0
        sumB = 0
        norm1 = 0
        norm2 = 0
        for i in range(self.count()):
            x1 = self.arr[i]
            x2 = c.arr[i]
            sumA += x1.real * x2.real + x1.imag * x2.imag
            sumB += x1.imag * x2.real - x1.real * x2.imag
            norm1 += abs_square(x1)
            norm2 += abs_square(x2)

        k = 1 / sqrt(norm1*norm2)
        return complex(sumA * k, sumB * k)

    def is_positive_orientation(self):
        sum_phase = 0
        curr = self.arr[0]
        for i in range(1, self.count()):
            next_curr = curr + self.arr[i]
            if not abs(next_curr) == 0:
                sum_phase += phase_between(next_curr, curr)
                curr = next_curr
            else:
                break

        return sum_phase > 0

    def distance(self, c):
        n1 = self.abs()
        n2 = c.abs()
        return n1 * n1 + n2 * n2 - 2 * self.dot(c).real

    def equalization(self, newCount):
        abs_phase = np.zeros(self.count(), dtype=float)
        for i in range(self.count()):
            abs_phase[i] = abs(self.arr[i])

        total_abs = sum((x for x in abs_phase))
        step = float(total_abs) / newCount

        result = np.zeros(newCount, dtype=complex)

        new_idx = 0
        to_cover = step
        i = 0

        v = self.arr[0]
        d = abs_phase[i]
        while i < self.count() and new_idx < newCount:
            if d < to_cover:
                to_cover -= d
                result[new_idx] += v
                i += 1
                if i == self.count():
                    break

                v = self.arr[i]
                d = abs_phase[i]
            elif d == to_cover:
                to_cover = step
                result[new_idx] += v
                new_idx += 1
                i += 1
                if i == self.count():
                    break

                v = self.arr[i]
                d = abs_phase[i]
            else:
                #k = to_cover / d
                result[new_idx] += v * to_cover / d
                v *= (d - to_cover) / d
                d *= (d - to_cover) / d
                new_idx += 1
                to_cover = step


        self.arr = result

    def equalizationUp(self, newCount):
        result = np.zeros(newCount, dtype=complex)

        for i in range(newCount):
            index = i * self.count() / newCount
            j = int(index)
            k = index - j
            if j == self.count() - 1:
                result[i] = self.arr[j]
            else:
                result[i] = self.arr[j] * (1 - k) + self.arr[j + 1] * k

        self.arr = result

    def equalizationDown(self, newCount):
        result = np.zeros(newCount, dtype=complex)

        for i in range(newCount):
            index = i * newCount / self.count()
            result[index] += self.arr[i]

        self.arr = result

    def get_points(self, start_point):
        result = np.zeros([self.count(), 2], dtype=np.uint8)
        result[0] = start_point
        sum = start_point
        for i in range(self.count()-1):
            sum = sum[0] + self.arr[i].real, sum[1] - self.arr[i].imag
            result[i + 1] = abs(sum[0]), abs(sum[1])

        return result



