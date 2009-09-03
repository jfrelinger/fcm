"""Collection of common FCM graphical plots."""

from util import bilinear_interpolate
from scipy import histogram
import pylab

def hist(fcms, index, savefile=None, display=True, new=True, **kwargs):
    """Plot overlay histogram.

    fcms is a list of histograms
    index is channel to plot
    """
    figure = pylab.figure()
    for fcm in fcms:
        y = fcm[:, index]
        h, b = histogram(y, bins=200)
        b = (b[:-1] + b[1:])/2.0
        x = pylab.linspace(min(y), max(y), 100)
        pylab.plot(b, h, label=fcm.name)
        pylab.legend()
    
    if display:
        pylab.show()
        
    if savefile:
        pylab.savefig(savefile)

    return figure


def heatmap(fcm, indices, nrows=1, ncols=1, savefile=None, display=True,
            **kwargs):
    """Plot a heatmap.

    indices = list of marker index pairs
    nrows = number of rows to plot
    ncols = number of cols to plot
    nrows * ncols should be >= len(indices)
    display = boolean indicating whether to show plot
    save = filename to save plot (e.g. 'x.png')
    **kwargs are passed on to pylab.scatter
    """

    if nrows==1 and ncols==1:
        ncols = len(indices)

    assert(nrows*ncols >= len(indices))

    figure = pylab.figure(figsize=(ncols*4, nrows*4))

    for i, idx in enumerate(indices):
        pylab.subplot(nrows, ncols, i+1)
        if (idx[0] != idx[1]):
            x = fcm[:,idx[0]]
            y = fcm[:,idx[1]]
            if not kwargs.has_key('c'):
                z = bilinear_interpolate(x, y)
                pylab.scatter(x, y, c=z, **kwargs)
            else:
                pylab.scatter(x, y, **kwargs)
            pylab.xlabel(fcm.channels[idx[0]])
            pylab.ylabel(fcm.channels[idx[1]])
        pylab.xticks([])
        pylab.yticks([])
        pylab.axis('equal')
        
    if display:
        pylab.show()
        
    if savefile:
        pylab.savefig(savefile)

    return figure

def heatmaps(fcm, savefile=None, display=True, **kwargs):
    """PLot scatter matrix of all heatmaps."""
    n = fcm.shape[1]
    indices = [(i,j) for i in range(n) for j in range(n)]
    heatmap(fcm, indices, nrows=n, ncols=n, savefile=savefile,
            display=display, **kwargs)

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from io import FCSreader

    fcm = FCSreader('../../sample_data/3FITC_4PE_004.fcs').get_FCMdata()
    # heatmap(fcm, [(0,1),(2,3)], nrows=1, ncols=2, s=1, edgecolors='none')
    heatmaps(fcm, s=1, edgecolors='none', display=False,
             savefile='3FITC_4PE_004.png', cmap=pylab.cm.hsv)
