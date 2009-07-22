from io import loadFCS
fcm = loadFCS('../sample_data/3FITC_4PE_004.fcs')
fcm.channels
fcm.shape
subsample = fcm[::10,:]
subsample
subsample.shape

from gui import Gate
from matplotlib import pyplot as plt
gate = Gate(fcm, [2,3], ax)
fig = plt.figure()
ax = fig.add_subplot(111)
gate = Gate(fcm, [2,3], ax)
plt.show()

fcm.view().shape
fcm.tree.current()
fcm.tree.parent()

fcm.tree.visit(fcm.tree.parent())
fcm.view().shape
fcm.tree.visit('root')
fcm.view()
fcm.visit(fcm.tree.children()[0])
fcm.tree.visit(fcm.tree.children()[0])
fcm.tree.visit(fcm.tree.children()[0])
fcm.view().shape
