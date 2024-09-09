import numpy


def bigbord(x, y, h):
    # function [newx,newy,newh]=bigbord(x,y,h);
    # Add a border out to a power of 2
    # adapted to output indices of input data within expanded grid

    if numpy.isscalar(h):
        m, n = (1, 1)
    else:
        m, n = h.shape

    # Find mean of perimeter values
    pmean = (numpy.sum(h[0, :] + h[m - 1, :]) + numpy.sum(h[1:m - 1, 0] + h[1:m - 1, n - 1])) / (2 * m + 2 * n - 2)
    #print('pmean = %s' % pmean)
    mm = numpy.max((m, n))
    mm = int(numpy.power(2, numpy.ceil(numpy.log2(numpy.max((m, n))))))
    #print('New square array dimension %d' % mm)
    newh = pmean * numpy.ones((mm, mm))
    m1 = int(numpy.floor((mm - m) / 2) - 1)
    m2 = int(m1 + m)
    n1 = int(numpy.floor((mm - n) / 2) - 1)
    n2 = int(n1 + n)
    #print('Old array now on indices: %s - %s, %s - %s' % (m1, m2, n1, n2))

    dx = numpy.absolute(x[1] - x[0])

    newx = numpy.empty((mm, 1))
    newx[n1:n2] = numpy.atleast_2d(x).T
    newx[0:n1] = numpy.expand_dims(x[0] - (numpy.arange(n1, 0, -1) * dx), axis=1)
    newx[n2:mm] = numpy.expand_dims(x[-1] + (numpy.arange(1, mm - n2 + 1) * dx), axis=1)

    newy = numpy.empty((mm, 1))
    newy[m1:m2] = numpy.atleast_2d(y).T
    newy[0:m1] = numpy.expand_dims(y[0] - (numpy.arange(m1, 0, -1) * dx), axis=1)
    newy[m2:mm] = numpy.expand_dims(y[-1] + (numpy.arange(1, mm - m2 + 1) * dx), axis=1)

    newh[m1:m2, n1:n2] = h

    # Linearly interpolate in the border
    for i in numpy.arange(1, n1 - 1):
        alpha = (i - 1) / (n1 - 1)
        newh[:, i] = (1 - alpha) * newh[:, 0] + alpha * newh[:, n1 - 1]

    for i in numpy.arange(n2, mm - 1):
        alpha = (i - n2) / (mm - n2)
        newh[:, i] = (1 - alpha) * newh[:, n2 - 1] + alpha * newh[:, mm - 1]

    for i in numpy.arange(1, m1 - 1):
        alpha = (i - 1) / (m1 - 1)
        newh[i, :] = (1 - alpha) * newh[0, :] + alpha * newh[m1 - 1, :]

    for i in numpy.arange(m2, mm - 1):
        alpha = (i - m2) / (mm - m2)
        newh[i, :] = (1 - alpha) * newh[m2 - 1, :] + alpha * newh[mm - 1, :]

    m = mm

    return (m1,m2), (n1,n2), newh
