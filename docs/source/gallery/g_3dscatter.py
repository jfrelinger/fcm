from __future__ import division
import os
import sys
import glob
import fcm
import fcm.statistics as stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import LinearSegmentedColormap

if __name__ == '__main__':

    data = fcm.loadFCS('../data/basics/10072101.02')
    cols = [2,3,4]

    x, y, z = data[:,cols[0]], data[:,cols[1]], data[:, cols[2]]
    try:
        labels = np.load('labels_10072101_02.npy')
    except:
        m = stats.DPMixtureModel(nclusts=32, burnin=1000, niter=100)
        m.ident = True
        r = m.fit(data[:, cols], verbose=10)
        rav = r.average()
        c = rav.make_modal()
        labels = c.classify(data[:, cols])
        np.save('labels_10072101_02.npy', labels)

    colors = LinearSegmentedColormap('colormap', cm.jet._segmentdata.copy(), np.max(labels))
    cs = [colors(i) for i in labels]
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.gca(projection='3d')
    scatter = ax.scatter(x, y, z, s=5, c=cs, edgecolors='none')
    ax.set_xlabel('CD3-FITC')
    ax.set_ylabel('CD8-PE')
    ax.set_zlabel('CD45-PerCP')
    azim = 135
    elev = 25
    ax.view_init(elev, azim) 

plt.savefig('3d_scatter.png')
plt.show()

