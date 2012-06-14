'''
Created on Oct 30, 2009

@author: Jacob Frelinger
'''

from distributions import compmixnormpdf
from numpy import array, log, sum, zeros, concatenate, mean, exp
from numpy.random import multivariate_normal as mvn
from numpy.random import multinomial
from component import Component
from util import modesearch
from warnings import warn


class DPCluster(Component):
    '''
    Single component cluster in mixture model
    '''

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

    def prob(self, x, logged=False, **kwargs):
        '''
        DPCluster.prob(x):
        returns probability of x beloging to this mixture compoent
        '''
        #return self.pi * mvnormpdf(x, self.mu, self.sigma)
        return compmixnormpdf(x, self.pi, self.mu, self.sigma, logged=logged, **kwargs)

    def draw(self, n=1):
        '''
        draw a random sample of size n form this cluster
        '''
        return mvn(self.mu, self.sigma, n)


class DPMixture(object):
    '''
    collection of components that describe a mixture model
    '''

    def __init__(self, clusters, niter=1, m=False, s=False, identified=False):
        '''
        DPMixture(clusters)
        cluster = list of DPCluster objects
        '''
        self.clusters = clusters
        self.niter = niter
        self.ident = identified
        if m is not False:
            self.m = m
        if s is not False:
            self.s = s

    def prob(self, x, logged=False, **kwargs):
        '''
        DPMixture.prob(x)
        returns an array of probabilities of x being in each component of the mixture
        '''
        #return array([i.prob(x) for i in self.clusters])
        return compmixnormpdf(x, self.pis(), self.mus(), self.sigmas(), logged=logged, **kwargs)

    def classify(self, x, **kwargs):
        '''
        DPMixture.classify(x):
        returns the classification (which mixture) x is a member of
        '''
        probs = self.prob(x, logged=True, **kwargs)
        try:
            unused_n, unused_j = x.shape
            #return array([i.argmax(0) for i in probs])
            return probs.argmax(1)
        except ValueError:
            return probs.argmax(0)

    def mus(self, normed=False):
        '''
        DPMixture.mus():
        returns an array of all cluster means
        '''
        if normed:
            return array([i.nmu for i in self.clusters])
        else:
            return array([i.mu for i in self.clusters])

    def sigmas(self, normed=False):
        '''
        DPMixture.sigmas():
        returns an array of all cluster variance/covariances
        '''
        if normed:
            return array([i.nsigma for i in self.clusters])
        else:
            return array([i.sigma for i in self.clusters])

    def pis(self):
        '''
        DPMixture.pis()
        return an array of all cluster weights/proportions
        '''
        return array([i.pi for i in self.clusters])

    def make_modal(self, tol=1e-5, maxiter=20):
        """
        find the modes and return a modal dp mixture
        """
        try:
            modes, cmap = modesearch(self.pis(), self.mus(True), self.sigmas(True), tol, maxiter)
            return ModalDPMixture(self.clusters, cmap, modes, self.m, self.s)

        except AttributeError:
            warn("trying to make modal of a mixture I'm not sure is normalized.\nThe mode finding algorithm is designed for normalized data.\nResults may be unexpected")
            modes, cmap = modesearch(self.pis(), self.mus(), self.sigmas(), tol, maxiter)
            return ModalDPMixture(self.clusters, cmap, modes)

    def log_likelihood(self, x):
        '''
        return the log liklihood of x belonging to this mixture
        '''
        
        return sum(log(sum(self.prob(x), axis=0)))

    def draw(self, n):
        '''
        draw n samples from the represented mixture model
        '''
        
        d = multinomial(n, self.pis())
        results = None
        for index, count in enumerate(d):
            if count > 0:
                try:
                    results = concatenate((results, self.clusters[index].draw(count)), 0)
                except ValueError:
                    results = self.clusters[index].draw(count)

        return results

    def average(self):
        '''
        average over mcmc draws to try and find the 'average' weights, means, and covariances
        '''
        if not self.ident:
            warn("model wasn't run with ident=True, therefor these averages are likely"
                 + "meaningless")
        
        k = len(self.clusters)/self.niter
        rslts = []
        for i in range(k):
            mu_avg = []
            sig_avg = []
            pi_avg = []
            for j in range(self.niter):
                mu_avg.append(self.clusters[j*k+i].mu)
                sig_avg.append(self.clusters[j*k+i].sigma)
                pi_avg.append(self.clusters[j*k+i].pi)
            
            rslts.append(DPCluster(mean(pi_avg, 0), mean(mu_avg, 0), mean(sig_avg, 0)))
            
        return DPMixture(rslts)
            
    def last(self, n=1):
        '''
        return the last n (defaults to 1) mcmc draws
        '''
        if n > self.niter:
            raise ValueError('n=%d is larger than niter (%d)' % (n, self.niter))
        rslts = []
        k = len(self.clusters)/self.niter
        for j in range(n):
            for i in range(k):
                rslts.append(self.clusters[-1*((i+(j*k))+1)])
        
        return DPMixture(rslts[::-1])
                             
                            



class ModalDPMixture(DPMixture):
    '''
    collection of modal components that describe a mixture model
    '''

    def __init__(self, clusters, cmap, modes, m=False, s=False):
        '''
        DPMixture(clusters)
        cluster = list of DPCluster objects
        cmap = map of modal clusters to component clusters
        '''
        self.clusters = clusters
        self.cmap = cmap
        self.modemap = modes

        if m is not False:
            self.m = m
        if s is not False:
            self.s = s

    def prob(self, x, logged=False, **kwargs):
        '''
        ModalDPMixture.prob(x)
        returns  an array of probabilities of x being in each mode of the modal mixture
        '''
        probs = compmixnormpdf(x, self.pis(), self.mus(), self.sigmas(), logged=logged, **kwargs)
        
        #can't sum in log prob space
        if logged:
            probs = exp(probs)
        try:

                
            n, j = x.shape # check we're more then 1 point
            rslt = zeros((n, len(self.cmap.keys())))
            for j in self.cmap.keys():
                rslt[:, j] = sum([probs[:, i] for i in self.cmap[j]], 0)
                


        except ValueError:
            #single point
            rslt = zeros((len(self.cmap.keys())))
            for j in self.cmap.keys():
                rslt[j] = sum([self.clusters[i].prob(x) for i in self.cmap[j]])

        #return to log prob space
        if logged:
            rslt = log(rslt)

        return rslt

    def modes(self):
        '''
        ModalDPMixture.modes():
        return an array of mode locations
        '''
        lst = []
        for i in self.modemap.itervalues():
            try:
                lst.append((array(i) * self.s) + self.m)
            except AttributeError:
                lst.append(i)
        return array(lst)

    def classify(self, x, **kwargs):
        '''
        ModalDPMixture.classify(x):
        returns the classification (which mixture) x is a member of
        '''
        probs = self.prob(x, logged=True, **kwargs)
        try:
            unused_n, unused_j = x.shape
            #return array([i.argmax(0) for i in probs])
            return probs.argmax(1)
        except ValueError:
            return probs.argmax(0)


