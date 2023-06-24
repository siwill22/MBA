import math
import numpy as np


def grdfft(xf, yf, tf):
    if len(xf) > 1:
        dx = np.absolute(xf[2] - xf[1])
    else:
        dx = xf

    if len(yf) > 1:
        dy = np.absolute(yf[2] - yf[1])
    else:
        dy = yf

    sp = np.fft.fft2(tf)
    sp = np.fft.fftshift(sp)

    # First, need to get the frequency scale right.
    # Remember, for even number of points in spectrum, central frequency
    # is not right at the centre, hence frequency spacing is given as follows
    kxmax = ((2 * np.pi) / (dx * 2))
    kymax = ((2 * np.pi) / (dy * 2))
    kxinc = kxmax / ((np.size(tf, 1) / 2))
    kyinc = kymax / ((np.size(tf, 0) / 2))

    # make grid of correctly scaled frequencies
    kx, ky = np.meshgrid(np.arange(-kxmax, kxmax, kxinc), np.arange(-kymax, kymax, kyinc))

    return kx, ky, sp



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
    kx, ky = np.meshgrid(np.arange(-kxmax, kxmax, kxinc), np.arange(-kymax, kymax, kyinc))

    return kx, ky



def bigbord(x, y, h):
    # function [newx,newy,newh]=bigbord(x,y,h);
    # Add a border out to a power of 2
    # adapted to output indices of input data within expanded grid

    if np.isscalar(h):
        m, n = (1, 1)
    else:
        m, n = h.shape

    # Find mean of perimeter values
    pmean = (np.sum(h[0, :] + h[m - 1, :]) + np.sum(h[1:m - 1, 0] + h[1:m - 1, n - 1])) / (2 * m + 2 * n - 2)
    #print('pmean = %s' % pmean)
    mm = np.max((m, n))
    mm = int(np.power(2, np.ceil(np.log2(np.max((m, n))))))
    #print('New square array dimension %d' % mm)
    newh = pmean * np.ones((mm, mm))
    m1 = int(np.floor((mm - m) / 2) - 1)
    m2 = int(m1 + m)
    n1 = int(np.floor((mm - n) / 2) - 1)
    n2 = int(n1 + n)
    #print('Old array now on indices: %s - %s, %s - %s' % (m1, m2, n1, n2))

    dx = np.absolute(x[1] - x[0])

    newx = np.empty((mm, 1))
    newx[n1:n2] = np.atleast_2d(x).T
    newx[0:n1] = np.expand_dims(x[0] - (np.arange(n1, 0, -1) * dx), axis=1)
    newx[n2:mm] = np.expand_dims(x[-1] + (np.arange(1, mm - n2 + 1) * dx), axis=1)

    newy = np.empty((mm, 1))
    newy[m1:m2] = np.atleast_2d(y).T
    newy[0:m1] = np.expand_dims(y[0] - (np.arange(m1, 0, -1) * dx), axis=1)
    newy[m2:mm] = np.expand_dims(y[-1] + (np.arange(1, mm - m2 + 1) * dx), axis=1)

    newh[m1:m2, n1:n2] = h

    # Linearly interpolate in the border
    for i in np.arange(1, n1 - 1):
        alpha = (i - 1) / (n1 - 1)
        newh[:, i] = (1 - alpha) * newh[:, 0] + alpha * newh[:, n1 - 1]

    for i in np.arange(n2, mm - 1):
        alpha = (i - n2) / (mm - n2)
        newh[:, i] = (1 - alpha) * newh[:, n2 - 1] + alpha * newh[:, mm - 1]

    for i in np.arange(1, m1 - 1):
        alpha = (i - 1) / (m1 - 1)
        newh[i, :] = (1 - alpha) * newh[0, :] + alpha * newh[m1 - 1, :]

    for i in np.arange(m2, mm - 1):
        alpha = (i - m2) / (mm - m2)
        newh[i, :] = (1 - alpha) * newh[m2 - 1, :] + alpha * newh[mm - 1, :]

    m = mm

    return (m1,m2), (n1,n2), newh


def grdmask(x, y, t, xf, yf, tf):
    # [tmask] = grdmask (x,y,t,xf,yf,tf);
    #   Trim and mask the filled grid defined by xf,yf,tf to only cover the area defined 
    #   by the (presumably original) grid x,y,t
    #   NB ASSUMES SAME CELL SIZE FOR BOTH GRIDS

    x0 = np.nonzero(np.absolute(x[0] - xf) == np.min(np.absolute(x[0] - xf)))[0][0]
    y0 = np.nonzero(np.absolute(y[0] - yf) == np.min(np.absolute(y[0] - yf)))[0][0]

    tmask = tf[y0:y0 + len(y), x0:x0 + len(x)]
    tmask[np.isnan(t)] = np.nan

    return tmask



def parkergrav(x, y, z1, z2, rho, z0):
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

    z2f = np.ones(z1f.shape) * z2

    ## only do this to get spectrum scale
    kx, ky, dummy = grdfft(x, y, z1f)

    # Gravitational Constant
    Gamma = 6.67e-11
    si2mg = 1e5
    Gamma = Gamma * si2mg
    rho = rho * 1000    # Unit conversion, g/cc to kg/m3

    k = np.sqrt(np.power(kx, 2) + np.power(ky, 2))
    # note next term includes an upward continuation from z0 to z=0;
    # Blakely (p294) doesn't have a minus sign here, but need it if convention
    # is positive downwards
    Const = 2 * math.pi * Gamma * np.exp(-k * z0)

    numtaylor = 1
    tol = 8
    NthSum = np.zeros(z1f.shape)

    while numtaylor <= tol:
        zf = (np.power(z1f, numtaylor)) * rho
        kx, ky, Zsp = grdfft(x, y, zf)
        Nthterm = (np.power(-k, numtaylor - 1) / math.factorial(numtaylor)) * Zsp

        NthSum = NthSum + (Const * Nthterm)

        print('finished iteration %s ....' % numtaylor)
        numtaylor = numtaylor + 1

    # Trim off the expanded area of grid
    t = np.real(np.fft.ifft2(np.fft.ifftshift(NthSum)))
    t = grdmask(x, y, z1, xf, yf, t)

    return t



def oldenburg(x, y, grv, delta_rho, zref, convergence_criteria=0.02):
    '''

    '''

    xf, yf, grvf = bigbord(x, y, grv)

    kx, ky, grv_spec = grdfft(x, y, grvf)
    k = np.sqrt(np.power(kx, 2) + np.power(ky, 2))

    ## Define constants
    Gamma = 6.67e-11
    si2mg = 1e5
    km2m=1e3
    Gamma = Gamma*si2mg

    Const = (grv_spec*np.exp(-k*zref))/(2*np.pi*Gamma*delta_rho)

    Z = np.real(np.fft.ifft2(np.fft.ifftshift(Const)))

    ## Taylor Series Expansion for summation term
    numtaylor = 1
    tol = 8
    NthSum = np.zeros(np.size(grvf))

    while numtaylor<=tol:
        NthTerm = ((np.power(k,numtaylor-1)) / math.factorial(numtaylor)) * (np.fft.fftshift(np.fft.fft2(Z)))
        
        NthSum = Const - NthTerm
        
        Z = np.real(np.fft.ifft2(np.fft.ifftshift(NthSum)))
        
        #print('finished iteration {:d}....'.format(numtaylor))
        numtaylor+=1

    Z = grdmask(x,y,grv,xf,yf,Z); 

    #Zinv = (Z/km2m)+zref
    Zinv = Z+zref

    return Zinv