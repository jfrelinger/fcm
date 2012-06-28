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

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

fs = glob.glob(os.path.join('..', 'data', 'basics', '*01'))
colors = LinearSegmentedColormap('colormap', cm.jet._segmentdata.copy(), len(fs))

verts = []
for f in fs:
    cd4 = fcm.loadFCS(f)[:, 'CD4-PE']
    z, edges = np.histogram(cd4, bins=50)
    y = 0.5*(z[1:] + z[:-1])
    x = np.arange(len(y))
    verts.append(zip(x, y))

verts = np.array(verts)
n, p, d = verts.shape
maxz = np.max(verts[:, 0])

poly = PolyCollection(verts, facecolors = [colors(i) for i in range(n)])
poly.set_alpha(0.7)
ax.add_collection3d(poly, zs=np.arange(n), zdir='y')

ax.set_xlabel('CD4-PE')
ax.set_xlim3d(0, p)
ax.set_ylabel('Sample')
ax.set_ylim3d(0,n)
ax.set_zlabel('Counts')
ax.set_zlim3d(0, 1.2*maxz)

plt.savefig('3d_hist.png')
plt.show()

    
