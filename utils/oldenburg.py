import math
import numpy as np

from bigbord import bigbord
from grdfft import grdfft
from grdmask import grdmask


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
