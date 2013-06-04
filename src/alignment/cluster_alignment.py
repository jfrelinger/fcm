from fcm.alignment.generate_cost import mean_distance, classification_distance, kldiv_distance
from fcm.alignment.munkres import munkres, max_cost_munkres
import numpy as np

distfunc = {'mean': mean_distance,
            'class': classification_distance,
            'kldiv': kldiv_distance}

  
    
class AlignMixture(object):
    def __init__(self, mx, dtype='kldiv', max_cost=None):
        self.mx = mx
        self.dtype = distfunc[dtype]
        self.max_cost = max_cost

    def align(self, my, min_unused=None, *args, **kwargs):
        cost = self.dtype(self.mx, my, *args, **kwargs)
        if self.max_cost is not None:
            munk = max_cost_munkres(cost, self.max_cost)
        else:            
            munk = munkres(cost)
        print munk, cost
        translate = {}
        if min_unused is None:
            min_unused = len(self.mx)
        for i, j in enumerate(munk.T):
            if np.any(j):
                translate[i] = np.arange(len(self.mx))[j].squeeze()
            else:
                translate[i] = min_unused
                min_unused +=1

        return my.reorder(translate)
        
        
        
        