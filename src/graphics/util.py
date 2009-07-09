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
