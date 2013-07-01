'''
Created on Aug 27, 2009

@author: jolly
'''
from fcm.core.tree import SubsampleNode, DropChannelNode
import fcm
from fcm.statistics import mixnormpdf
import numpy as np
import numpy.random as npr



class Subsample(object):
    '''
    Takes a slice object and performs subsampling on the fcm object.
    '''


    def __init__(self, slicing):
        '''
        slicing = tuple of slices
        '''
        self.samp = slicing

    def subsample(self, fcs):
        """D(<fcmdata>) -> leads to a new fcm view of subsampled data"""
        node = SubsampleNode("", fcs.get_cur_node(), self.samp)
        fcs.add_view(node)
        return fcs

class _SubsampleFactory(object):
    '''
    factory generator of subsample objects
    '''

    def __init__(self):
        pass

    def __getitem__(self, item):
        return Subsample(item)

SubsampleFactory = _SubsampleFactory()

class RandomSubsample(Subsample):
    '''
    randomly subsample events
    '''
    def __init__(self, n):
        '''
        n = number of events to sample
        '''
        self.n = n
        
    def subsample(self, fcs):
        x = fcs[:]
        samp = npr.choice(np.arange(x.shape[0]), self.n)
        if isinstance(fcs, fcm.FCMdata):
            node = SubsampleNode("", fcs.get_cur_node(), samp)
            fcs.add_view(node)
            return fcs
        else:
            return x[samp]
    
    
class AnomalySubsample(Subsample):
    def __init__(self, n, neg):
        self.n = n
        self.neg = neg
    
    def subsample(self, fcs):
        x = fcs[:]
        p = mixnormpdf(x, self.neg.pis, self.neg.mus, self.neg.sigmas)
        p = 1/p
        p = p/np.sum(p)
    
        samp = npr.choice(np.arange(x.shape[0]), size=self.n, replace=False, p=p)
        if isinstance(fcs, fcm.FCMdata):
            node = SubsampleNode("", fcs.get_cur_node(), samp)
            fcs.add_view(node)
            return fcs
        else:
            return x[samp]
    
class BiasSubsample(Subsample):
    def __init__(self, n, pos, neg):
        self.n = n
        self.pos = pos
        self.neg = neg
        
    def subsample(self, fcs,*args, **kwargs):
        x = fcs[:]
        neg_py = mixnormpdf(x, self.neg.pis, self.neg.mus, self.neg.sigmas,
                            logged=True, *args, **kwargs)
        pos_py = mixnormpdf(x, self.pos.pis, self.pos.mus, self.pos.sigmas,
                            logged=True,*args, **kwargs)

        diff = pos_py - neg_py

        probs = np.exp(diff)
        probs = probs / np.sum(probs)
        samp = npr.choice(np.arange(x.shape[0]), size=self.n,
                                replace=False, p=probs)
        
        if isinstance(fcs, fcm.FCMdata):
            node = SubsampleNode("", fcs.get_cur_node(), samp)
            fcs.add_view(node)
            return fcs
        else:
            return x[samp]
    

class DropChannel(object):
    """
    Drop channels by name from a fcm view
    """
    def __init__(self, idxs):
        self.idxs = idxs

    def drop(self, fcs):
        """D(<fcmdata>) -> create a new view in the fcm object missing the specified channels"""
        channels = fcs.get_cur_node().channels[:]
        idxs = [fcs.name_to_index(i) if isinstance(i, str) else i for i in self.idxs] 
        left = []
        for i in range(len(channels)):
            if i not in idxs:
                left.append(i)

        node = DropChannelNode("", fcs.get_cur_node(), left, [channels[i] for i in left])
        fcs.add_view(node)
        return fcs
