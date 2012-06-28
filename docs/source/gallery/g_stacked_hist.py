"""Overlay and stacked histograms."""

import os
import pylab
import glob
import fcm
from matplotlib.colors import LinearSegmentedColormap

bins = 50
fs = glob.glob(os.path.join('..', 'data', 'basics', '*01'))
colors = LinearSegmentedColormap('colormap', pylab.cm.jet._segmentdata.copy(), len(fs))

# stacked histogram
pylab.figure(figsize=(5,15))
for k, f in enumerate(fs):
    path, filename = os.path.split(f)
    name, ext = os.path.splitext(filename)
    data = fcm.loadFCS(f)
    pylab.subplot(5,1,k+1)
    pylab.hist(data[:, 'CD4-PE'], bins, histtype='step', color=colors(k), label=name)
    pylab.ylabel('Counts')
    if k==(len(fs) - 1):
        pylab.xlabel('CD4-PE')
    else:
        pylab.xticks([])
    pylab.legend()

pylab.tight_layout()
pylab.savefig('stacked_hist.png')
pylab.show()
