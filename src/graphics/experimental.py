from __future__ import division
from enthought.mayavi import mlab
import numpy

def surface(fcm, idx0, idx1):
    """Plots a surface plot for 2D data."""
    x = fcm[:,idx0]
    y = fcm[:,idx1]
    bins = int(numpy.sqrt(len(x)))
    z, xedge, yedge = numpy.histogram2d(y, x, bins=[bins, bins], 
                                        range=[(numpy.min(y), numpy.max(y)),
                                               (numpy.min(x), numpy.max(x))]
                                        )
    
    mlab.figure()
    mlab.surf(xedge, yedge, z, warp_scale='auto')
    mlab.xlabel(fcm.channels[idx0])
    mlab.ylabel(fcm.channels[idx1])
    mlab.zlabel('Density')

def spin(fcm, idx0, idx1, idx2):
    """Plots 3D data as points in space."""
    x = fcm[:,idx0]
    y = fcm[:,idx1]
    z = fcm[:,idx2]
    bins = int(len(x)**(1/3.0))

    xfrac, xint = numpy.modf((x - numpy.min(x))/
                             (numpy.max(x)-numpy.min(x))*(bins-1))
    yfrac, yint = numpy.modf((y - numpy.min(y))/
                             (numpy.max(y)-numpy.min(y))*(bins-1))
    zfrac, zint = numpy.modf((z - numpy.min(z))/
                             (numpy.max(z)-numpy.min(z))*(bins-1))

    xint = xint.astype('i')
    yint = yint.astype('i')
    zint = zint.astype('i')

    # not interpolated - kiv write trilinear_interpolate function
    h, edges = numpy.histogramdd(fcm[:,[idx0, idx1, idx2]], bins=bins)
    v = h[xint, yint, zint]

    mlab.figure()
    mlab.points3d(x, y, z, v, mode='point')
    mlab.xlabel(fcm.channels[idx0])
    mlab.ylabel(fcm.channels[idx1])
    mlab.zlabel(fcm.channels[idx2])

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from io import FCSreader

    fcm = FCSreader('../../sample_data/3FITC_4PE_004.fcs').get_FCMdata()
    
    surface(fcm, 2, 3)

    spin(fcm, 1, 2, 3)

    mlab.show()
