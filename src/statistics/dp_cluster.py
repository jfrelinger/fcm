'''
Created on Oct 30, 2009

@author: Jacob Frelinger
'''

from distributions import mvnormpdf
from numpy import array
from enthought.traits.api import HasTraits, List, Float, Array


class DPCluster(HasTraits):
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
        return probs.argmax(1)
    