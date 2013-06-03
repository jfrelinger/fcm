'''
graphical routines for plotting or aiding in plotting flow data
'''
from fcm.graphics.plot import pair_plot, pseudocolor, pseudocolors, hist, contour
from fcm.graphics.util import bilinear_interpolate, trilinear_interpolate, set_logicle
from fcm.graphics.plot_gate import plot_gate

__all__ = [
        'pair_plot',
        'pseudocolor',
        'hist',
        'pseudocolors',
        'contour',
        'bilinear_interpolate',
        'trilinear_interpolate',
        'set_logicle',
        'plot_gate',
        ]
