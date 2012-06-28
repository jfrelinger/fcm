from __future__ import division
import matplotlib
# matplotlib.use('Agg')
import os
import sys
import fcm
import fcm.statistics as stats
import numpy as np
from scipy.interpolate import Rbf
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib import cm
from fcm.graphics import bilinear_interpolate
import scipy.ndimage as ndi

def make_image(data, x0, x1, y0, y1, w, h, border=0):
    kx = (w - 1) / (x1 - x0)
    ky = (h - 1) / (y1 - y0)
    imgw = (w + 2 * border)
    imgh = (h + 2 * border)
    img = np.zeros((imgh,imgw))
    for x, y in data:
        ix = int((x - x0) * kx) + border
        iy = int((y - y0) * ky) + border
        if 0 <= ix < imgw and 0 <= iy < imgh:
            img[iy][ix] += 1
    return img

if __name__ == '__main__':

    data = fcm.loadFCS('../data/basics/10072101.01')
    cols = [2,3]
    xmin = 0
    ymin = 0
    xmax = int(data.notes.text['p%dr' % (1+cols[0])])
    ymax = int(data.notes.text['p%dr' % (1+cols[1])])

    x, y = data[:,cols[0]], data[:,cols[1]]
    xmu, xsd = x.mean(), x.std()
    ymu, ysd = y.mean(), y.std()
    x1 = (x-xmu)/xsd
    y1 = (y-ymu)/ysd

    view_xmin = (xmin-xmu)/xsd
    view_ymin = (xmin-ymu)/ysd
    view_xmax = (xmax-xmu)/xsd
    view_ymax = (ymax-ymu)/ysd
    z0 = np.zeros((1024,1024))

    img = make_image(zip(x1, y1),view_xmin,view_xmax,view_ymin,view_ymax,
                     1024,1024)
    xi = np.linspace(0,1024,1024)
    yi = np.linspace(0,1024,1024)

    plt.figure(figsize=(8,8))
    # Pseudocolor using bilinear interpolation
    plt.subplot(2,2,1)
    
    try:
        z = np.load('density_10072101_01.npy')
    except:
        m = stats.DPMixtureModel(nclusts=32, niter=100, burnin=1000)
        m.ident = True
        r = m.fit(data[:, cols], verbose=True).average()
        z = stats.mixnormpdf(data[:, cols], r.pis(), r.mus(), r.sigmas())
        np.save('density_10072101_02.npy', z)

    idx = np.argsort(z)
    plt.scatter(x[idx], y[idx], s=1, edgecolors='none', c=z[idx])
    plt.xticks([])
    plt.yticks([])
    plt.imshow(z0, origin='lower', extent=[xmin, xmax, ymin, ymax],
               cmap=plt.cm.binary)
    plt.title('Pseudocolor (DPGMM)')

    # Pseudocolor using kernel density estimate
    plt.subplot(2,2,2)
    kde = gaussian_kde([x1, y1])
    z = kde.evaluate([x1, y1])
    plt.scatter(x, y, s=1, edgecolors='none', c=z)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(z0, origin='lower', extent=[xmin, xmax, ymin, ymax],
               cmap=plt.cm.binary)
    plt.title('Pseudocolor (KDE)')

    # Heatmap using Guassian filter smoothing
    plt.subplot(2,2,3)
    z = ndi.gaussian_filter(img, sigma=25)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(z, origin='lower',
               extent=[xmin,xmax,ymin,ymax])
    plt.title('Smoothed heatmap')

    # Contour plot using Guassian filter smoothing
    plt.subplot(2,2,4)
    Z = ndi.gaussian_filter(img, sigma=25)
    X, Y = np.meshgrid(xi, yi)
    plt.contour(X, Y, Z, 30)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(z0, origin='lower', extent=[xmin, xmax, ymin, ymax],
               cmap=plt.cm.binary)
    plt.title('Smoothed contour plot')

plt.tight_layout()
plt.savefig('dotplots.png')
plt.show()
