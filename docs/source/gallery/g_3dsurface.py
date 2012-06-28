import os
import sys
import glob
import fcm
import fcm.statistics as stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
from matplotlib.colors import LinearSegmentedColormap
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

    data = fcm.loadFCS('../data/basics/10072101.02')
    cols = [3,4]
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
    Z = ndi.gaussian_filter(img, sigma=25)

    xi = np.linspace(0,1024,1024)
    yi = np.linspace(0,1024,1024)
    X, Y = np.meshgrid(xi, yi)
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=8, cstride=8, cmap=cm.jet,
                           linewidth=0, antialiased=False)
    ax.set_xlabel('CD8-PE')
    ax.set_ylabel('CD45-PerCP')
    fig.colorbar(surf, shrink=0.5, aspect=5)

plt.savefig('3d_surface.png')
plt.show()

