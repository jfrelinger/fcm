import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mp
from matplotlib.ticker import NullFormatter
from matplotlib.ticker import MultipleLocator
from fcm import loadFCS

# Load FCS file using loadFCS from fcm
data = loadFCS("3FITC_4PE_004.fcs")
x = data[:,'FSC-H']
y = data[:,'SSC-H']

# definitions for the axes 
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
bottom_h = left_h = left+width+0.02

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom_h, width, 0.2]
rect_histy = [left_h, bottom, 0.2, height]

plt.figure(1, figsize=(10,10))


axScatter = plt.axes(rect_scatter)
axHistx = plt.axes(rect_histx)
axHistx.xaxis.set_major_formatter(NullFormatter())
axHistx.xaxis.set_minor_formatter(NullFormatter())
axHisty = plt.axes(rect_histy)
axHisty.yaxis.set_major_formatter(NullFormatter())
axHisty.yaxis.set_minor_formatter(NullFormatter())

axHistx.xaxis.set_major_locator( MultipleLocator(10) )
axHistx.xaxis.set_minor_locator( MultipleLocator(10) )
axHisty.yaxis.set_major_locator( MultipleLocator(10) )
axHisty.yaxis.set_minor_locator( MultipleLocator(10) )

# the scatter plot:
axScatter.scatter(x, y, edgecolors='none',s=1)


# now determine nice limits by hand:
binwidth = 0.5
xymax = np.max( [np.max(np.fabs(x)), np.max(np.fabs(y))] )
lim = ( int(xymax/binwidth) + 1) * binwidth

axScatter.set_xlim( (0, lim) )
axScatter.set_ylim( (0, lim) )

bins = np.arange(0, lim + binwidth, binwidth)
axHistx.hist(x, bins=bins, facecolor='blue', edgecolor='blue', histtype='stepfilled')
axHisty.hist(y, bins=bins, orientation='horizontal', facecolor='blue', edgecolor='blue',  histtype='stepfilled')

axHistx.set_xlim( axScatter.get_xlim() )
axHistx.set_xticks(())
axHistx.set_yticks(())
axHisty.set_ylim( axScatter.get_ylim() )
axHisty.set_xticks(())
axHisty.set_yticks(())

plt.show()
