import fcm
import matplotlib.pyplot as plt

#load FCS data
data = fcm.loadFCS('../../sample_data/3FITC_4PE_004.fcs')

#define a gate
gate1 = fcm.PolyGate([(400,100),(400,300),(600,300),(600,100)], (0,1))

#apply the gate
gate1.gate(data)

# outputs:
# root
#  g1
print data.tree.pprint()

# g1 isn't and informative name, so lets rename it events
current_node = data.current_node
data.tree.rename_node(current_node.name, 'events')
# outputs:
# root
#  events
print data.tree.pprint()

#return to the root node and plot
data.tree.visit('root')
plt.scatter(data[:,0],data[:,1], s=1, edgecolors='none', c='grey')

#and visit the subset of interest to plot
data.tree.visit(current_node)
plt.scatter(data[:,0],data[:,1], s=1, edgecolors='none', c='blue')
plt.show()
