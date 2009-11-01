'''
Created on Oct 30, 2009

@author: jolly
'''

from enthought.traits.api import HasTraits
from numpy import zeros

from cdp_wrapper import cdpcluster

class DPMixtureModel(HasTraits):
    '''
    Fits a DP Mixture model to a fcm dataset.
    
    '''


    def __init__(self,fcmdata, nclusts, itter=1000, burnin= 100, last= 5):
        '''
        DPMixtureModel(fcmdata, nclusts, itter=1000, burnin= 100, last= 5)
        fcmdata = a fcm data object
        nclusts = number of clusters to fit
        itter = number of mcmc itterations
        burning = number of mcmc burnin itterations
        last = number of mcmc itterations to draw samples from
        
        '''
        pnts = fcmdata.view()
        self.m = pnts.mean(0)
        self.s = pnts.std(0)
        self.data = (pnts-self.m)/self.s
        
        self.nclusts = nclusts
        self.itter = itter
        self.burnin = burnin
        self.last = last
        
        self.d = self.data.shape[1]
        self.pi = zeros((nclusts*last))
        self.mus = zeros((nclusts*last, self.d))
        self.sigmas = zeros((nclusts*last, self.d, self.d))
        
        
    def fit(self, verbose=False):
        self.cdp = cdpcluster(self.data)
        self.cdp.setT(1)
        self.cdp.setJ(self.nclusts)
        self.cdp.setBurnin(self.burnin)
        self.cdp.setIter(self.itter-self.last)
        if verbose:
            self.cdp.setVerbose(True)
        self.cdp.run()
        
        for i in range(self.last):
            for j in range(self.nclusts):
                pass
            
        
        self._run = True #we've fit the mixture model
        
    
        
        
        
        