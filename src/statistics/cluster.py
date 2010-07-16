'''
Created on Oct 30, 2009

@author: jolly
'''

from enthought.traits.api import HasTraits
from numpy import zeros, outer, sum

from cdp import cdpcluster
from dp_cluster import DPCluster, DPMixture

class DPMixtureModel(HasTraits):
    '''
    Fits a DP Mixture model to a fcm dataset.
    
    '''


    def __init__(self,fcmdata, nclusts, iter=1000, burnin= 100, last= 5):
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
        self.iter = iter
        self.burnin = burnin
        self.last = last
        
        self.n, self.d = self.data.shape
        
        self.pi = zeros((nclusts*last))
        self.mus = zeros((nclusts*last, self.d))
        self.sigmas = zeros((nclusts*last, self.d, self.d))
        self.cdp = cdpcluster(self.data)
        self._run = False
        
    def fit(self, verbose=False):
        """
        fit the mixture model to the data
        use get_results() to get the fitted model
        """
        self.cdp.setT(self.nclusts)
        self.cdp.setJ(1)
        self.cdp.setBurnin(self.burnin)
        self.cdp.setIter(self.iter-self.last)
        if verbose:
            self.cdp.setVerbose(True)
        self.cdp.run()
        
        
        self._run = True #we've fit the mixture model
        
        idx = 0
        n = self.burnin+self.iter-self.last+1
        for i in range(self.last):
            for j in range(self.nclusts):
                self.pi[idx] = self._getpi(j)
                self.mus[idx,:] = self._getmu(j)
                self.sigmas[idx,:,:] = self._getsigma(j)
                idx+=1
            self.cdp.step()
            if verbose:
                print "it = %d" % (n+i)
        if verbose:
            print "Done"
                
    def _getpi(self, idx):
        return self.cdp.getp(idx)
    
    def _getmu(self,idx):
        tmp = zeros(self.d)
        for i in range(self.d):
            tmp[i] = self.cdp.getMu(idx,i)
            
        return tmp

    def _getsigma(self, idx):
        tmp = zeros((self.d,self.d))
        for i in range(self.d):
            for j in range(self.d):
                tmp[i,j] = self.cdp.getSigma(idx,i,j)   
        
        return tmp
        
        
        
    def get_results(self):
        """
        get the results of the fitted mixture model
        """
        
        if self._run:
            self.pi = self.pi/sum(self.pi)
            rslts = []
            for i in range(self.last * self.nclusts):
                tmp = DPCluster(self.pi[i],(self.mus[i]*self.s) + self.m, self.sigmas[i]*outer(self.s,self.s))
                tmp.nmu = self.mus[i]
                tmp.nsigma = self.sigmas[i]
                rslts.append(tmp)
            tmp = DPMixture(rslts, self.m, self.s)
            return tmp
        else:
            return None # TODO raise exception
        
    def get_class(self):
        """
        get the last classification from the model
        """
        
        if self._run:
            return self.cdp.getK(self.n)
        else:
            return None # TODO raise exception
            