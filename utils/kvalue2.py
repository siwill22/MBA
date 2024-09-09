import math
import numpy


def kvalue2(nx, ny, dx, dy):
    # [kx,ky] = kvalue2 (nx,ny,dx,dy);
    #   Wavenumber coordinates for 2D FFT Spectrum
    # ASSUMES EVEN NUMBERED NX

    # First, need to get the frequency scale right.
    # Remember, for even number of points in spectrum, central frequency 
    # is not right at the centre, hence frequency spacing is given as follows
    kxmax = ((2 * math.pi) / (dx * 2))
    kymax = ((2 * math.pi) / (dy * 2))
    kxinc = kxmax / ((nx / 2))
    kyinc = kymax / ((ny / 2))

    # make grid of correctly scaled frequencies
    kx, ky = numpy.meshgrid(numpy.arange(-kxmax, kxmax, kxinc), numpy.arange(-kymax, kymax, kyinc))

    return kx, ky
