import numpy


def nextpow2(n):
    m_f = numpy.log2(n)
    m_i = numpy.ceil(m_f)
    return 2 ** m_i


def bordr3(data, nnx=None, nny=None):
    # BORDR3 Apply border region to input grid
    #
    # Usage: out=bordr3(data,nnx,nny)
    #
    # Maurice Tivey Oct 1995

    ndatay, ndatax = data.shape
    if nnx is None or nny is None:
        # determine next power of two automatically
        nnx = numpy.power(2, nextpow2(ndatax))
        nny = numpy.power(2, nextpow2(ndatay))

    print(' APPLY BORDER FROM SIZE %3d  X %3d TO %3d X %3d' % (ndatax, ndatay, nnx, nny))
    W = numpy.zeros((nny, nnx))
    W[0:ndatay, 0:ndatax] = data
    scale = 1. / (nny - ndatay + 2.)

    A1 = W[ndatay - 1, 0:ndatax]
    A2 = W[0, 0:ndatax]

    for j in numpy.arange(ndatay - 1, nny):
        xi = scale * ((j + 1) - ndatay)
        W[j, 0:ndatax] = A1 * (1. - xi) + A2 * xi

    scale = 1. / (nnx - ndatax + 2)
    A1 = W[0:nny, ndatax - 1]
    A2 = W[0:nny, 0]

    for i in numpy.arange(ndatax - 1, nnx):
        xi = scale * (i + 1 - ndatax)
        W[0:nny, i] = A1 * (1. - xi) + A2 * xi

    return W
