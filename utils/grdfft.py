import math
import numpy


def length(obj):
    return numpy.max(obj.shape)


def grdfft(xf, yf, tf):
    if length(xf) > 1:
        dx = numpy.absolute(xf[2] - xf[1])
    else:
        dx = xf

    if length(yf) > 1:
        dy = numpy.absolute(yf[2] - yf[1])
    else:
        dy = yf

    sp = numpy.fft.fft2(tf)
    sp = numpy.fft.fftshift(sp)

    # First, need to get the frequency scale right.
    # Remember, for even number of points in spectrum, central frequency
    # is not right at the centre, hence frequency spacing is given as follows
    kxmax = ((2 * math.pi) / (dx * 2))
    kymax = ((2 * math.pi) / (dy * 2))
    kxinc = kxmax / ((numpy.size(tf, 1) / 2))
    kyinc = kymax / ((numpy.size(tf, 0) / 2))

    # make grid of correctly scaled frequencies
    kx, ky = numpy.meshgrid(numpy.arange(-kxmax, kxmax, kxinc), numpy.arange(-kymax, kymax, kyinc))

    return kx, ky, sp
