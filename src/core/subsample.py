'''
Created on Aug 27, 2009

@author: jolly
'''
from util import SubsampleNode

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
    