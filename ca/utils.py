import math


def abs_square(cmplx):
    return cmplx.real * cmplx.real + cmplx.imag * cmplx.imag


def complex_offset(cmplx, delta_a, delta_b):
    return cmplx + (delta_a, delta_b)


def complex_rotate(cmplx, angle):
    a = cmplx.real
    b = cmplx.imag
    cosAngle = math.cos(angle)
    sinAngle = math.sin(angle)
    return complex(round(cosAngle*a - sinAngle*b, ndigits=5), round(sinAngle*a + cosAngle*b, ndigits=5))


def phase_between(x, y):
    coss = (x.real * y.real + x.imag * y.imag) / (abs(x)*abs(y))
    return math.acos(coss)