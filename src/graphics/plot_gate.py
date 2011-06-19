from util import bilinear_interpolate
from fcm import PolyGate, IntervalGate, ThresholdGate

def plot_gate(data, gate, ax, chan=None, name=None, **kwargs):
    """
    wrapper around several plots aimed at plotting gates (and gating at the same time)
    see the wrapped functions plot_ploy_gate, plot_threshold_gate, plot_threshold_hist
    for more information
    """
    if isinstance(gate,PolyGate):
        plot_poly_gate(data, gate, ax, chan, name, **kwargs)
    elif isinstance(gate,ThresholdGate):
        if isinstance(chan,int) or len(chan) == 1:
            plot_threshold_hist(data, gate, ax, chan, name, **kwargs)
        else:
            plot_threshold_gate(data, gate, ax, chan, name, **kwargs)
            
    
def plot_threshold_gate(data, gate, ax, chan=None, name=None, **kwargs):
    pass


def plot_threshold_hist(data, gate, ax, chan=None, name=None, **kwargs):
    pass


def plot_poly_gate(data, gate, ax, chan=None, name=None, **kwargs):
    if chan is None:
        chan = gate.chan
        
    if 'bgc' in kwargs:
        bgc = kwargs['bgc']
        del kwargs['bgc']
    else:
        bgc = 'grey'
        
    if 'bgalpha' in kwargs:
        bga = kwargs['bgalpha']
        del kwargs['bgalpha']
    else:
        bga = 1

    if 'c' in kwargs:
        z = kwargs['c']
        del kwargs['c'] # needed incase bgc is used
        calc_z = False
    else:
        calc_z = True
        
    if 'alpha' in kwargs:
        alpha = kwargs['alpha']
        del kwargs['alpha']
    else:
        alpha = 1 # is there a way to look up what this defaults to incase this is changed in .matplotlibrc?

    # add in support for specifing size
    
    ax.scatter(data[:,chan[0]],data[:,chan[1]], c=bgc, s=1, edgecolor='none', alpha=bga)
    gate.gate(data, chan=chan, name=name)

    #has to be set after gating...

    if calc_z:
        z = bilinear_interpolate(data[:, chan[0]], data[:, chan[1]])  

    ax.scatter(data[:,chan[0]],data[:,chan[1]], c=z, s=1, edgecolor='none', alpha=alpha, **kwargs)
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
    plot_gate(x,g,ax, name="firstgate", alpha=.5, bgalpha=.5)
    ax = fig.add_subplot(1,2,2)
    plot_gate(x,g2,ax, name="secondgate", alpha=.5, bgc='red', c='green')
    print x.tree.pprint()
    plt.show()
    