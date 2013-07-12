from fcm.alignment.generate_cost import mean_distance, classification_distance, kldiv_distance
from fcm.alignment.munkres import munkres, max_cost_munkres
import numpy as np

distfunc = {'mean': mean_distance,
            'class': classification_distance,
            'kldiv': kldiv_distance}

class AlignMixture(object):
    '''
    find alignment map between two mixture models
    '''
    def __init__(self, mx, model='kldiv'):
        self.mx = mx
        self.dtype = distfunc[model]

    def get_cost(self, my, *args, **kwargs):
        '''
        generate cost matrix between reference set and my
        '''
        return self.dtype(self.mx, my, *args, **kwargs)
    
    def align(self, my, max_cost = None, min_unused=None, *args, **kwargs):
        '''
        generate alignment map for my and convert to an ordered mixture model
        '''
        cost = self.get_cost(my, *args, **kwargs)
        if max_cost is not None:
            munk = max_cost_munkres(cost, max_cost)
        else:            
            munk = munkres(cost)
        translate = {}
        if min_unused is None:
            min_unused = cost.shape[0]
        for i, j in enumerate(munk):
            if np.any(j):
                translate[i] = np.arange(cost.shape[0])[j].squeeze()
            else:
                translate[i] = min_unused
                min_unused +=1

        return my.reorder(translate)
        
        
        
        