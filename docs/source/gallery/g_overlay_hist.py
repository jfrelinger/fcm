"""Overlay and stacked histograms."""

import os
import pylab
import glob
import fcm

bins = 50
fs = glob.glob(os.path.join('..', 'data', 'basics', '*01'))

# overlay histogram
for f in fs:
    path, filename = os.path.split(f)
    name, ext = os.path.splitext(filename)
    data = fcm.loadFCS(f)
    pylab.subplot(1,1,1)
    pylab.hist(data[:, 'CD4-PE'], bins, histtype='step', label=filename)
pylab.xlabel('CD4-PE')
pylab.ylabel('Counts')
pylab.legend()
pylab.tight_layout()

pylab.show()
