'''
Created on Oct 30, 2009

@author: Jacob Frelinger
'''

from distributions import mvnormpdf
from numpy import array
from component import Component
from util import modesearch
from enthought.traits.api import HasTraits, List, Float, Array, Dict, Int


class DPCluster(HasTraits, Component):
    '''
    Single component cluster in mixture model
    '''
    pi = Float()
    mu = Array()
    sigma = Array()
    def __init__(self, pi, mu, sig):
        '''
        DPCluster(pi,mu,sigma)
        pi = cluster weight
        mu = cluster mean
        sigma = cluster variance/covariance
        '''
        self.pi = pi
        self.mu = mu
        self.sigma = sig
        
    def prob(self, x):
        '''
        DPCluster.prob(x):
        returns probability of x beloging to this mixture compoent
        '''
        return self.pi * mvnormpdf(x, self.mu, self.sigma)
    


class DPMixture(HasTraits):
    '''
    collection of compoents that describe a mixture model
    '''
    clusters = List(DPCluster)
    def __init__(self, clusters):
        '''
        DPMixture(clusters)
        cluster = list of DPCluster objects
        '''
        self.clusters = clusters
        
    def prob(self, x):
        '''
        DPMixture(x)
        returns a list of probabilities of x being in each component of the mixture
        '''
        return array([i.prob(x) for i in self.clusters])
    
    def classify(self, x):
        '''
        DPMixture.classify(x):
        returns the classification (which mixture) x is a member of
        '''
        probs = self.prob(x)
        return probs.argmax(0)
    
    def mus(self):
        '''
        DPMixture.mus():
        returns an array of all cluster means
        '''
        return array([i.mu for i in self.clusters])
    
    def sigmas(self):
        '''
        DPMixture.sigmas():
        returns an array of all cluster variance/covariances
        '''
        return array([i.sigma for i in self.clusters])
    
    def pis(self):
        '''
        DPMixture.pis()
        return an array of all cluster weights/proportions
        '''
        return array([i.pi for i in self.clusters])
    
    def make_modal(self, tol=1e-5, maxiter=20):
        modes,cmap = modesearch(self.pis(), self.mus(), self.sigmas(), tol, maxiter)
        return ModalDPMixture(self.clusters, cmap, modes)
        
        
    

class ModalDPMixture(DPMixture, HasTraits):
    '''
    collection of modal compoents that describe a mixture model
    '''
    clusters = List(DPCluster)
    cmap = Dict(Int, List(Int))
    modes = Dict(Int, Array)
    def __init__(self, clusters, cmap, modes):
        '''
        DPMixture(clusters)
        cluster = list of DPCluster objects
        cmap = map of modal clusters to component clusters
        '''
        self.clusters = clusters
        self.cmap = cmap
        self.modemap = modes

        
    def prob(self,x):
        rslt = []
        for j in self.cmap.keys():
            rslt.append(sum([self.clusters[i].prob(x) for i in self.cmap[j]]))
            
        return array(rslt)
    
    def modes(self):
        lst = []
        for i in self.modemap.itervalues():
            lst.append(i)
        return array(lst)        

