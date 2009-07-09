from __future__ import division
import numpy
# from scipy.interpolate import interp2d

def bilinear_interpolate(x, y, bins=None):
    """Returns interpolated density values on points (x, y).
    
    Ref: http://en.wikipedia.org/wiki/Bilinear_interpolation.
    """
    if bins is None:
        bins = int(numpy.sqrt(len(x)))

    z, xedge, yedge = numpy.histogram2d(y, x, bins=[bins, bins], 
                                        range=[(numpy.min(y), numpy.max(y)),
                                               (numpy.min(x), numpy.max(x))]
                                        )
    xfrac, xint = numpy.modf((x - numpy.min(x))/
                             (numpy.max(x)-numpy.min(x))*(bins-1))
    yfrac, yint = numpy.modf((y - numpy.min(y))/
                             (numpy.max(y)-numpy.min(y))*(bins-1))

    xint = xint.astype('i')
    yint = yint.astype('i')

    z1 = numpy.zeros(numpy.array(z.shape)+1)
    z1[:-1,:-1] = z

    # values at corners of square for interpolation
    q11 = z1[yint, xint]
    q12 = z1[yint, xint+1]
    q21 = z1[yint+1, xint]
    q22 = z1[yint+1, xint+1]
    
    return q11*(1-xfrac)*(1-yfrac) + q21*(1-xfrac)*(yfrac) + \
        q12*(xfrac)*(1-yfrac) + q22*(xfrac)*(yfrac)

def trilinear_interpolate(x, y, z, bins=None):
    """Returns interpolated density values on points (x, y, z).
    
    Ref: http://en.wikipedia.org/wiki/Trilinear_interpolation.
    """
    if bins is None:
        bins = int(len(x)**(1/3.0))
        
    vals = numpy.zeros((len(x), 3), 'd')
    vals[:,0] = x
    vals[:,1] = y
    vals[:,2] = z

    h, edges = numpy.histogramdd(vals, 
                                 bins=[bins, bins, bins]
                                 )
    xfrac, xint = numpy.modf((x - numpy.min(x))/
                             (numpy.max(x)-numpy.min(x))*(bins-1))
    yfrac, yint = numpy.modf((y - numpy.min(y))/
                             (numpy.max(y)-numpy.min(y))*(bins-1))
    zfrac, zint = numpy.modf((z - numpy.min(z))/
                             (numpy.max(z)-numpy.min(z))*(bins-1))

    xint = xint.astype('i')
    yint = yint.astype('i')
    zint = zint.astype('i')

    h1 = numpy.zeros(numpy.array(h.shape)+1)
    h1[:-1,:-1,:-1] = h

    # values at corners of cube for interpolation
    q111 = h1[xint, yint, zint]
    q112 = h1[xint+1, yint, zint]
    q122 = h1[xint+1, yint+1, zint]
    q121 = h1[xint, yint+1, zint]
    q211 = h1[xint, yint, zint+1]
    q212 = h1[xint+1, yint, zint+1]
    q222 = h1[xint+1, yint+1, zint+1]
    q221 = h1[xint, yint+1, zint+1]

    i1 = q111*(1-zfrac) + q211*(zfrac)
    i2 = q121*(1-zfrac) + q221*(zfrac)
    j1 = q112*(1-zfrac) + q212*(zfrac)
    j2 = q122*(1-zfrac) + q222*(zfrac)

    w1 = i1*(1-yfrac) + i2*(yfrac)
    w2 = j1*(1-yfrac) + j2*(yfrac)

    return w1*(1-xfrac) + w2*(xfrac)

if __name__ == '__main__':
    import pylab
    import sys
    sys.path.append('../')
    from io import FCSreader

    fcm = FCSreader('../../sample_data/3FITC_4PE_004.fcs').get_FCMdata()
    x = fcm[:,0]
    y = fcm[:,1]

    import time
    start = time.clock()
    z = bilinear_interpolate(x, y)
    print time.clock() - start

    pylab.scatter(x, y, s=1, c=z, edgecolors='none', cmap=pylab.cm.get_cmap())
    pylab.show()
