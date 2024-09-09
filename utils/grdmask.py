import numpy


def length(obj):
    return numpy.max(obj.shape)


def grdmask(x, y, t, xf, yf, tf):
    # [tmask] = grdmask (x,y,t,xf,yf,tf);
    #   Trim and mask the filled grid defined by xf,yf,tf to only cover the area defined 
    #   by the (presumably original) grid x,y,t
    #   NB ASSUMES SAME CELL SIZE FOR BOTH GRIDS

    x0 = numpy.nonzero(numpy.absolute(x[0] - xf) == numpy.min(numpy.absolute(x[0] - xf)))[0][0]
    y0 = numpy.nonzero(numpy.absolute(y[0] - yf) == numpy.min(numpy.absolute(y[0] - yf)))[0][0]

    tmask = tf[y0:y0 + len(y), x0:x0 + len(x)]
    tmask[numpy.isnan(t)] = numpy.nan

    return tmask
