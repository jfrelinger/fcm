'''
Created on Oct 30, 2009

@author: Jacob Frelinger
'''

from distributions import mvnormpdf
from enthought.traits.api import HasTraits, ListClass, Float, Array

class DPCluster(HasTraits):
    pi = Float()
    mu = Array()
    sigma = Array()
    def __init__(self, pi, mu, sig):
        self.pi = pi
        self.mu = mu
        self.sigma = sig
        
    def prob(self, x):
        return self.pi * mvnormpdf(x, self.mu, self.sigma)
    


class DPMixture(HasTraits):
    clusters = ListClass(DPCluster)
    def __init__(self, clusters):
        self.clusters = clusters
    def prob(self, x):
        return [i.prob(x) for i in self.clusters]