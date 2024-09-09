import math
import numpy

from bigbord import bigbord
from grdfft import grdfft
from grdmask import grdmask


def parkergrav(x, y, z1, z2, M, z0):
    '''
    [GravEffect] = parkergrav(x,y,z1,z2,RHO);
    Synthetic gravity response response for layer bounded by surfaces z1 and z2
    Note that z positive downwards
    see Blakely's potential field theory textbook, p294
    INPUTS:    x = vector containing x coordinates of topography grids
               y = vector containing x coordinates of topography grids
               z1 = grid of depth to upper surface of layer
               z2 = grid of depth to lower surface of layer
               RHO = density contrast at layer surface
               z0 = reference depth
    OUTPUTS: GravEffect = Gravity response of modelled layer

    NB units MUST be metres and g/cc (1000 smaller than kg/m3)
    '''

    # for fast convergence, z0 is set to be midway between minimum 
    # values of z1 and z2. Could cause problems for very thick layers
    # z0 = (max(z1(~isnan(z1)))+min(z1(~isnan(z1))))/2;
    # 
    # z0 = max(max(z1));

    z1 = z1 - z0
    z2 = z2 - z0

    print('z0 set to %s' % z0)

    # Expand grid to avoid edge effects
    xf, yf, z1f = bigbord(x, y, z1)

    z2f = numpy.ones(z1f.shape) * z2

    ## only do this to get spectrum scale
    kx, ky, dummy = grdfft(x, y, z1f)

    # Gravitational Constant
    Gamma = 6.67e-11
    si2mg = 1e5
    Gamma = Gamma * si2mg
    M = M * 1000    # Unit conversion, g/cc to kg/m3

    k = numpy.sqrt(numpy.power(kx, 2) + numpy.power(ky, 2))
    # note next term includes an upward continuation from z0 to z=0;
    # Blakely (p294) doesn't have a minus sign here, but need it if convention
    # is positive downwards
    Const = 2 * math.pi * Gamma * numpy.exp(-k * z0)

    numtaylor = 1
    tol = 8
    NthSum = numpy.zeros(z1f.shape)

    while numtaylor <= tol:
        zf = (numpy.power(z1f, numtaylor)) * M
        kx, ky, Zsp = grdfft(x, y, zf)
        Nthterm = (numpy.power(-k, numtaylor - 1) / math.factorial(numtaylor)) * Zsp

        NthSum = NthSum + (Const * Nthterm)

        print('finished iteration %s ....' % numtaylor)
        numtaylor = numtaylor + 1

    # Trim off the expanded area of grid
    t = numpy.real(numpy.fft.ifft2(numpy.fft.ifftshift(NthSum)))
    t = grdmask(x, y, z1, xf, yf, t)

    return t
