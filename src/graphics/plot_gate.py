from util import bilinear_interpolate
from fcm import PolyGate, IntervalGate, ThresholdGate

def plot_gate(data, gate, ax, chan=None, name=None, *args, **kwargs):
    if chan is None:
        # TODinterval and threshold gates.chan member don't make sense for 2d plots
        chan = gate.chan 
    ax.scatter(data[:,chan[0]],data[:,chan[1]], c='grey', s=1, edgecolor='none')
    
    gate.gate(data, chan=chan, name=name)
    z = bilinear_interpolate(data[:, chan[0]], data[:, chan[1]])
    ax.scatter(data[:,chan[0]],data[:,chan[1]], c=z, s=1, edgecolor='none')
    
    if isinstance(gate,PolyGate):
        ax.fill(gate.vert.T[0], gate.vert.T[1], edgecolor='black', facecolor='none') 
    
if __name__ == '__main__':
    import fcm
    import numpy
    import matplotlib
    import matplotlib.pyplot as plt
    x = fcm.loadFCS('../../sample_data/3FITC_4PE_004.fcs')
    g = PolyGate(numpy.array([[0,0],[500,0],[500,500],[0,500]]), [0,1])
    g2 = PolyGate(numpy.array([[0,0],[500,0],[500,500],[0,500]]), [2,3])
    fig = plt.figure()
    ax = fig.add_subplot(1,2,1)
    plot_gate(x,g,ax, name="firstgate")
    ax = fig.add_subplot(1,2,2)
    plot_gate(x,g2,ax, name="secondgate")
    print x.tree.pprint()
    plt.show()
    