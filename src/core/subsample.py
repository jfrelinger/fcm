'''
Created on Aug 27, 2009

@author: jolly
'''
from fcm.core.tree import SubsampleNode, DropChannelNode
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

    def subsample(self, fcm):
        """D(<fcmdata>) -> leads to a new fcm view of subsampled data"""
        node = SubsampleNode("", fcm.get_cur_node(), self.samp)
        fcm.add_view(node)
        return fcm

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
        
    def subsample(self, fcm):
        x = fcm[:]
        samp = npr.choice(np.arange(x.shape[0]), self.n)
        node = SubsampleNode("", fcm.get_cur_node(), samp)
        fcm.add_view(node)
        return fcm
    
    
class AnomalySubsample(Subsample):
    def __init__(self, n, neg):
        self.n = n
        self.neg = neg
    
    def subsample(self, fcm):
        x = fcm[:]
        p = mixnormpdf(x, self.neg.pis, self.neg.mus, self.neg.sigmas)
        p = 1/p
        p = p/np.sum(p)
    
        samp = npr.choice(np.arange(x.shape[0]), size=self.n, replace=False, p=p)
        node = SubsampleNode("", fcm.get_cur_node(), samp)
        fcm.add_view(node)
        return fcm
    
class BiasSubsample(Subsample):
    def __init__(self, n, pos, neg):
        self.n = n
        self.pos = pos
        self.neg = neg
        
    def subsample(self, fcm,*args, **kwargs):
        x = fcm[:]
        neg_py = mixnormpdf(x, self.neg.pis, self.neg.mus, self.neg.sigmas, logged=True, *args, **kwargs)
        pos_py = mixnormpdf(x, self.pos.pis, self.pos.mus, self.pos.sigmas, logged=True, *args, **kwargs)

        diff = pos_py - neg_py

        probs = np.exp(diff)
        probs = probs / np.sum(probs)
        samp = npr.choice(np.arange(x.shape[0]), size=self.n,
                                replace=False, p=probs)
        node = SubsampleNode("", fcm.get_cur_node(), samp)
        fcm.add_view(node)
        return fcm
    

class DropChannel(object):
    """
    Drop channels by name from a fcm view
    """
    def __init__(self, idxs):
        self.idxs = idxs

    def drop(self, fcm):
        """D(<fcmdata>) -> create a new view in the fcm object missing the specified channels"""
        channels = fcm.channels[:]

        left = []
        for j,i in enumerate(fcm.channels):
            if i in self.idxs:
                channels.remove(i)
            elif j in self.idxs:
                channels.remove(i)
            else:
                if isinstance(i, str):
                    left.append(fcm.name_to_index(i))
                else:
                    left.append(i)

        node = DropChannelNode("", fcm.get_cur_node(), left, channels)
        fcm.add_view(node)
        return fcm
