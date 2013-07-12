'''
Created on Oct 30, 2009

@author: Jacob Frelinger
'''

from fcm.statistics.distributions import compmixnormpdf
from fcm.statistics.component import Component
from numpy import array, log, sum, zeros, concatenate, mean, exp, ndarray, dot
from numpy import outer
import numpy as np
from scipy.misc import logsumexp
from numpy.random import multivariate_normal as mvn
from numpy.random import multinomial
from numbers import Number
from util import modesearch
from warnings import warn
from copy import deepcopy

from modelresult import ModelResult


class DPCluster(Component):
    '''
    Single component cluster in mixture model
    '''

    __array_priority__ = 10

    def __init__(self, pi, mu, sig, centered_mu=None, centered_sigma=None):
        '''
        DPCluster(pi,mu,sigma)
        pi = cluster weight
        mu = cluster mean
        sigma = cluster variance/covariance
        '''
        self.pi = pi
        self.mu = mu
        self.sigma = sig
        if centered_mu is not None:
            self._centered_mu = centered_mu
        else:
            self._centered_mu = None
        if centered_sigma is not None:
            self._centered_sigma = centered_sigma
        else:
            self._centered_sigma = None
    
    @property
    def centered_mu(self):
        if self._centered_mu is None:
            raise AttributeError
        else:
            return self._centered_mu

    @centered_mu.setter
    def centered_mu(self, x):
        self._centered_mu = x

    @property
    def centered_sigma(self):
        if self._centered_sigma is None:
            raise AttributeError
        else:
            return self._centered_sigma

    @centered_sigma.setter
    def centered_sigma(self, s):
        self._centered_sigma = s

    def prob(self, x, logged=False, **kwargs):
        '''
        DPCluster.prob(x):
        returns probability of x belonging to this mixture component
        '''
        #return self.pi * mvnormpdf(x, self.mu, self.sigma)
        d = self.mu.shape[0]
        return compmixnormpdf(x, self.pi, self.mu.reshape(1,-1), self.sigma.reshape(1,d,d), logged=logged, **kwargs)

    def draw(self, n=1):
        '''
        draw a random sample of size n form this cluster
        '''
        # cast n to a int incase it's a numpy.int
        n = int(n)
        return mvn(self.mu, self.sigma, n)

    def __add__(self, k):
        new_mu = self.mu + k
        return DPCluster(self.pi, new_mu, self.sigma)

    def __radd__(self, k):
        new_mu = k + self.mu
        return DPCluster(self.pi, new_mu, self.sigma)

    def __sub__(self, k):
        new_mu = self.mu - k
        return DPCluster(self.pi, new_mu, self.sigma)

    def __rsub__(self, k):
        new_mu = k - self.mu
        return DPCluster(self.pi, new_mu, self.sigma)

    def __mul__(self, k):

        if isinstance(k, Number):
            new_mu = self.mu * k
            new_sigma = k * k * self.sigma
        elif isinstance(k, ndarray):
            new_mu = dot(self.mu, k)
            new_sigma = dot(dot(k, self.sigma), k.T)
        else:
            raise TypeError('unsupported type: %s' % type(k))

        return DPCluster(self.pi, new_mu, new_sigma)

    def __rmul__(self, k):
        if isinstance(k, Number):
            new_mu = self.mu * k
            new_sigma = k * k * self.sigma
        elif isinstance(k, ndarray):
            new_mu = dot(k, self.mu)
            new_sigma = dot(dot(k, self.sigma), k.T)
        else:
            raise TypeError('unsupported type: %s' % type(k))

        return DPCluster(self.pi, new_mu, new_sigma)


class DPMixture(ModelResult):
    '''
    collection of components that describe a mixture model
    '''

    __array_priority__ = 10

    def __init__(self, clusters, niter=1, m=None, s=None, identified=False):
        '''
        DPMixture(clusters)
        cluster = list of DPCluster objects
        '''
        self.clusters = clusters
        self.niter = niter
        self.ident = identified
        self.m = m
        self.s = s


    def __add__(self, k):
        new_clusters = [i + k for i in self.clusters]
        return DPMixture(new_clusters, self.niter, self.m, self.s, self.ident)

    def __radd__(self, k):
        new_clusters = [k + i for i in self.clusters]
        return DPMixture(new_clusters, self.niter, self.m, self.s, self.ident)

    def __sub__(self, k):
        new_clusters = [i - k for i in self.clusters]
        return DPMixture(new_clusters, self.niter, self.m, self.s, self.ident)

    def __rsub__(self, k):
        new_clusters = [k - i for i in self.clusters]
        return DPMixture(new_clusters, self.niter, self.m, self.s, self.ident)

    def __mul__(self, a):
        new_clusters = [i * a for i in self.clusters]
        return DPMixture(new_clusters, self.niter, self.m, self.s, self.ident)

    def __rmul__(self, a):
        new_clusters = [a * i for i in self.clusters]
        return DPMixture(new_clusters, self.niter, self.m, self.s, self.ident)

    def __len__(self):
        return len(self.clusters)

    def __getitem__(self, s):
        return self.clusters.__getitem__(s)

    def __setitem__(self, s, values):
        return self.clusters.__setitem__(s, values)

    def prob(self, x, logged=False, **kwargs):
        '''
        DPMixture.prob(x)
        returns an array of probabilities of x being in each component of the
        mixture
        '''
        #return array([i.prob(x) for i in self.clusters])
        return compmixnormpdf(x, self.pis, self.mus, self.sigmas, logged=logged, **kwargs)

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

    @property
    def mus(self):
        '''
        DPMixture.mus():
        returns an array of all cluster means
        '''
        return array([i.mu for i in self.clusters])

    @property
    def centered_mus(self):
        return array([i.centered_mu for i in self.clusters])

    @property
    def sigmas(self):
        '''
        DPMixture.sigmas():
        returns an array of all cluster variance/covariances
        '''
        return array([i.sigma for i in self.clusters])

    @property
    def centered_sigmas(self):
        return array([i.centered_sigma for i in self.clusters])

    @property
    def pis(self):
        '''
        DPMixture.pis()
        return an array of all cluster weights/proportions
        '''
        return array([i.pi for i in self.clusters])

    def make_modal(self, **kwargs):
        """
        find the modes and return a modal dp mixture
        """
        try:
            modes, cmap = modesearch(self.pis, self.centered_mus, self.centered_sigmas, **kwargs)
            return ModalDPMixture(self.clusters, cmap, modes, self.niter, self.m, self.s, self.ident)

        except AttributeError:
            modes, cmap = modesearch(self.pis, self.mus, self.sigmas, **kwargs)
            return ModalDPMixture(self.clusters, cmap, modes, self.m, self.s)

    def log_likelihood(self, x):
        '''
        return the log likelihood of x belonging to this mixture
        '''

        return sum(log(sum(self.prob(x), axis=0)))

    def draw(self, n):
        '''
        draw n samples from the represented mixture model
        '''

        d = multinomial(n, self.pis)
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
        average over mcmc draws to try and find the 'average' weights, means,
        and covariances
        '''
        if not self.ident:
            warn("model wasn't run with ident=True, therefor these averages "
                 "are likely meaningless")

        k = len(self.clusters) / self.niter
        rslts = []
        for i in range(k):
            mu_avg = []
            sig_avg = []
            pi_avg = []
            for j in range(self.niter):
                mu_avg.append(self.clusters[j * k + i].mu)
                sig_avg.append(self.clusters[j * k + i].sigma)
                pi_avg.append(self.clusters[j * k + i].pi)

            rslts.append(DPCluster(mean(pi_avg, 0), mean(mu_avg, 0), mean(sig_avg, 0)))

        return DPMixture(rslts)

    def last(self, n=1):
        '''
        return the last n (defaults to 1) mcmc draws
        '''
        if n > self.niter:
            raise ValueError('n=%d is larger than niter (%d)' % (n, self.niter))
        rslts = []
        k = len(self.clusters) / self.niter
        for j in range(n):
            for i in range(k):
                rslts.append(self.clusters[-1 * ((i + (j * k)) + 1)])

        return DPMixture(rslts[::-1])
    
    def get_submodel(self, idxs):
        '''
        return a sub model of only specific clusters
        '''
        if isinstance(idxs, Number):
            idxs = [idxs]
        rslts = [deepcopy(self.clusters[i]) for i in idxs]
        norm = sum([i.pi for i in rslts])
        for i in rslts:
            i.pi = i.pi/norm
        return DPMixture(rslts,1,self.m, self.s, self.ident)
    
    def get_iteration(self, iters):
        '''
        return a sub model of specific iterations
        x.get_iteration(0) returns a DPMixture of the first iteration
        x.get_iteration([0,2,4,6]) returs a DPMixture of iterations 0,2,4,6,
            eg. poor mans thinning.
        '''
        if isinstance(iters, Number):
            iters = [iters]
        for i, j in enumerate(iters):
            if abs(i) > self.niter:
                raise IndexError('index error out of range: %d' % i)
            if i < 0:
                iters[j] = self.niter - i

        stride = len(self.clusters) / self.niter
        keep = []
        for i in iters:
            for j in range(stride):
                keep.append(deepcopy(self.clusters[(i * stride) + j]))

        pi_adjust = sum(i.pi for i in keep)
        for i in keep:
            i.pi = i.pi / pi_adjust

        return DPMixture(keep, len(iters), self.m, self.s, self.ident)

    def get_marginal(self, margin):
        if isinstance(margin, Number):
            margin = array([margin])
        elif not isinstance(margin, np.ndarray):
            margin = array([margin]).reshape(-1)
        try:
            d = self.mus.shape[1]
        except:
            d = 1

        newd = margin.shape[0]
        rslts = []
        x = zeros(d, dtype=np.bool)

        for i in margin:
            x[i] = True

        y = outer(x, x)

        for i in self.clusters:
            try:
                rslts.append(DPCluster(i.pi, i.mu[x], i.sigma[y].reshape(newd, newd), i.centered_mu[x], i.centered_sigma[y].reshape(newd, newd)))
            except AttributeError:
                rslts.append(DPCluster(i.pi, i.mu[x], i.sigma[y].reshape(newd, newd)))

        return DPMixture(rslts, self.niter, self.m, self.s, self.ident)

    def enumerate_clusters(self):
        '''
        enumerate through clusters
        '''
        for i in range(len(self.clusters)):
            yield i, self.clusters[i]

    def enumerate_pis(self):
        for i in range(len(self.clusters)):
            yield i, self.clusters[i].pi

    def enumerate_mus(self):
        '''
        enumerate through the clusters means
        '''
        for i in range(len(self.clusters)):
            yield i, self.clusters[i].mu

    def enumerate_sigmas(self):
        '''
        enumerate through the cluster covariances
        '''
        for i in range(len(self.clusters)):
            yield i, self.clusters[i].sigma

    def reorder(self, lookup):
        '''
        add an order to a DPMixture
        '''
        return OrderedDPMixture(self.clusters, lookup, self.niter, self.m, self.s, self.ident)

class OrderedDPMixture(DPMixture):
    '''
    a ordered/identified DPMixture
    '''
    def __init__(self, clusters, lookup, niter=1, m=None, s=None, identified=False):
        self.lookup = lookup
        super(OrderedDPMixture, self).__init__(clusters, niter, m, s, identified)

    def __add__(self, k):
        return super(OrderedDPMixture, self).__add__(k).reorder(self.lookup)

    def __radd__(self, k):
        return super(OrderedDPMixture, self).__radd__(k).reorder(self.lookup)

    def __sub__(self, k):
        return super(OrderedDPMixture, self).__sub__(k).reorder(self.lookup)

    def __rsub__(self, k):
        return super(OrderedDPMixture, self).__rsub__(k).reorder(self.lookup)

    def __mul__(self, k):
        return super(OrderedDPMixture, self).__mul__(k).reorder(self.lookup)

    def __rmul__(self, k):
        return super(OrderedDPMixture, self).__rmul__(k).reorder(self.lookup)

    def enumerate_clusters(self):
        '''
        enumerate through clusters
        '''
        for i in range(len(self.clusters)):
            yield self.lookup[i], self.clusters[i]

    def enumerate_pis(self):
        for i in range(len(self.clusters)):
            yield self.lookup[i], self.clusters[i].pi

    def enumerate_mus(self):
        '''
        enumerate through the clusters means
        '''
        for i in range(len(self.clusters)):
            yield self.lookup[i], self.clusters[i].mu

    def enumerate_sigmas(self):
        '''
        enumerate through the cluster covariances
        '''
        for i in range(len(self.clusters)):
            yield self.lookup[i], self.clusters[i].sigma

    def classify(self, x, **kwargs):
        z = super(OrderedDPMixture, self).classify(x, **kwargs)
        lut = np.array([self.lookup[i] for i in range(len(self))])
        return lut[z]
        

class ModalDPMixture(DPMixture):
    '''
    collection of modal components that describe a mixture model
    '''

    def __init__(self, clusters, cmap, modes, niter=1, m=False, s=False, ident=False):
        '''
        ModalDPMixture(clusters)
        cluster = list of DPCluster objects
        cmap = map of modal clusters to component clusters
        modes = array of mode locations
        '''
        self.clusters = clusters
        self.cmap = cmap
        self.modemap = modes
        self.niter = niter
        self.ident = ident
        
        if m is not False:
            self.m = m
        else:
            self.m = 0
        if s is not False:
            self.s = s
        else:
            self.s = 1

    def __add__(self, k):
        new_clusters = [i + k for i in self.clusters]
        new_modes = {}
        for i in self.modemap:
            new_modes[i] = self.modemap[i]+k
        return ModalDPMixture(new_clusters, self.cmap, new_modes, self.niter, self.m, self.s, self.ident)

    def __radd__(self, k):
        new_clusters = [k + i for i in self.clusters]
        new_modes = {}
        for i in self.modemap:
            new_modes[i] = k+self.modemap[i]
        return ModalDPMixture(new_clusters, self.cmap, new_modes, self.niter, self.m, self.s, self.ident)

    def __sub__(self, k):
        new_clusters = [i - k for i in self.clusters]
        new_modes = {}
        for i in self.modemap:
            new_modes[i] = self.modemap[i]-k
        return ModalDPMixture(new_clusters, self.cmap, new_modes, self.niter, self.m, self.s, self.ident)

    def __rsub__(self, k):
        new_clusters = [k - i for i in self.clusters]
        new_modes = {}
        for i in self.modemap:
            new_modes[i] = k-self.modemap[i]
        return ModalDPMixture(new_clusters, self.cmap, new_modes, self.niter, self.m, self.s, self.ident)

    def __mul__(self, a):
        new_clusters = [i * a for i in self.clusters]
        new_modes = {}
        for i in self.modemap:
            if isinstance(a,Number):
                new_modes[i] = self.modemap[i]*a
            else:
                new_modes[i] = np.dot(self.modemap[i],a)
        return ModalDPMixture(new_clusters, self.cmap, new_modes, self.niter, self.m, self.s, self.ident)

    def __rmul__(self, a):
        new_clusters = [a * i for i in self.clusters]
        new_modes = {}
        for i in self.modemap:
            if isinstance(a,Number):
                new_modes[i] = a*self.modemap[i]
            else:
                new_modes[i] = np.dot(a,self.modemap[i])
        return ModalDPMixture(new_clusters, self.cmap, new_modes, self.niter, self.m, self.s, self.ident)


    def __len__(self):
        return len(self.modemap)

    def prob(self, x, logged=False, **kwargs):
        '''
        ModalDPMixture.prob(x)
        returns  an array of probabilities of x being in each mode of the modal
        mixture
        '''
        probs = compmixnormpdf(x, self.pis, self.mus, self.sigmas, logged=logged, **kwargs)

        #can't sum in log prob space
        
        try:
            n, j = x.shape  # check we're more then 1 point
            rslt = zeros((n, len(self.cmap.keys())))
            for j in self.cmap.keys():
                if logged:
                    rslt[:,j] = logsumexp([probs[:,i] for i in self.cmap[j]], 0)
                else:
                    rslt[:, j] = sum([probs[:, i] for i in self.cmap[j]], 0)
        except ValueError:
            #single point
            rslt = zeros((len(self.cmap.keys())))
            for j in self.cmap.keys():
                if logged:
                    rslt[j] = logsumexp([self.clusters[i].prob(x, logged=logged) for i in self.cmap[j]])
                else:
                    rslt[j] = sum([self.clusters[i].prob(x) for i in self.cmap[j]])


        return rslt

    @property
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
    
    @property
    def centered_modes(self):
        return array([i for i in self.modemap.itervalues()])
    
    def enumerate_modes(self):
        for i in range(len(self.modes)):
            yield i, self.modes[i]



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

    def reorder(self, lookup):
        return OrderedModalDPMixture(self.clusters, self.cmap, self.modemap, lookup, self.m, self.s)

class OrderedModalDPMixture(ModalDPMixture):
    '''
    an ordered Modal DP Mixture
    '''
    def __init__(self, clusters, cmap, modes, lookup, m=False, s=False):
        '''
        clusters, cmap, modes, lookup, m=False, s=False
        '''
        super(OrderedModalDPMixture, self).__init__(clusters, cmap, modes, m, s)
        self.lookup = lookup

    def __add__(self, k):
        return super(OrderedModalDPMixture, self).__add__(k).reorder(self.lookup)

    def __radd__(self, k):
        return super(OrderedModalDPMixture, self).__radd__(k).reorder(self.lookup)

    def __sub__(self, k):
        return super(OrderedModalDPMixture, self).__sub__(k).reorder(self.lookup)

    def __rsub__(self, k):
        return super(OrderedModalDPMixture, self).__rsub__(k).reorder(self.lookup)

    def __mul__(self, k):
        return super(OrderedModalDPMixture, self).__mul__(k).reorder(self.lookup)

    def __rmul__(self, k):
        return super(OrderedModalDPMixture, self).__rmul__(k).reorder(self.lookup)


    def enumerate_modes(self):
        for i in range(len(self.modes)):
            yield self.lookup[i], self.modes[i]



    def classify(self, x, **kwargs):
        z = super(OrderedModalDPMixture, self).classify(x, **kwargs)
        lut = np.array([self.lookup[i] for i in range(len(self))])
        return lut[z]

class HDPMixture(Component):
    '''
    a 'collection' of DPMixtures with common means and variances
    '''

    __array_priority__ = 10

    def __init__(self, pis, mus, sigmas, niter=1, m=0, s=1, identified=False):
        self.pis = pis.squeeze()
        self.mus = mus.squeeze()
        self.sigmas = sigmas.squeeze()
        self.niter = niter
        self.ident = identified
        self.m = m
        self.s = s

    def __len__(self):
        return self.pis.shape[0]

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self[ii] for ii in xrange(*key.indices(len(self)))]
        elif isinstance(key, int):
            #Handle negative indices
            if key < 0:
                key += len(self)
            if key >= len(self):
                raise IndexError("The index (%d) is out of range." % key)
            return self._getData(key)
        else:
            raise TypeError("Invalid argument type.")

    def _getData(self, key):
        pis = self.pis[key, :]
        mus = (self.mus - self.m) / self.s
        sigmas = (self.sigmas) / outer(self.s, self.s)
        clsts = [DPCluster(i, j, k, l, m) for i, j, k, l, m in
                 zip(pis, self.mus, self.sigmas, mus, sigmas)]
        return DPMixture(clsts, self.niter, self.m, self.s, self.ident)

    def __add__(self, x):
        return HDPMixture(self.pis, self.mus + x, self.sigmas, self.niter, self.m, self.s, self.ident)

    def __radd__(self, x):
        return HDPMixture(self.pis, x + self.mus, self.sigmas, self.niter, self.m, self.s, self.ident)

    def __sub__(self, x):
        return HDPMixture(self.pis, self.mus - x, self.sigmas, self.niter, self.m, self.s, self.ident)

    def __rsub__(self, x):
        return HDPMixture(self.pis, x + self.mus, self.sigmas, self.niter, self.m, self.s, self.ident)

    def __mul__(self, k):

        if isinstance(k, Number):
            new_mu = self.mus * k
            new_sigma = k * k * self.sigmas
        elif isinstance(k, ndarray):
            new_mu = dot(self.mus, k)
            new_sigma = np.array([dot(dot(k, i), k.T) for i in self.sigmas])
        else:
            raise TypeError('unsupported type: %s' % type(k))

        return HDPMixture(self.pis, new_mu, new_sigma, self.niter, self.m, self.s, self.ident)

    def __rmul__(self, k):
        if isinstance(k, Number):
            new_mu = self.mus * k
            new_sigma = k * k * self.sigmas
        elif isinstance(k, ndarray):
            new_mu = dot(k, self.mus)
            new_sigma = np.array([dot(dot(k, i), k.T) for i in self.sigmas])
        else:
            raise TypeError('unsupported type: %s' % type(k))

        return HDPMixture(self.pi, new_mu, new_sigma, self.niter, self.m, self.s, self.ident)

    def prob(self, x, **kwargs):
        return array([i.prob(x, kwargs) for i in self])

    def classify(self, x, **kwargs):
        return array([i.classify(x, **kwargs) for i in self])

    def average(self):
        offset = self.mus.shape[0] / self.niter
        d = self.mus.shape[1]
        new_mus = self.mus.reshape(self.niter, offset, d).mean(0).squeeze()
        new_sigmas = self.sigmas.reshape(self.niter, offset, d, d).mean(0).squeeze()
        new_pis = self.pis.reshape(len(self), self.niter, offset).mean(1).squeeze()

        return HDPMixture(new_pis, new_mus, new_sigmas, 1, self.m, self.s, self.ident)

    def make_modal(self, **kwargs):
        # set reference cmap by averaging
        r_consensus = self._getData(0)
        pis = sum(self.pis, 0)
        pis /= sum(pis)
        for i in range(len(r_consensus.clusters)):
            r_consensus.clusters[i].pi = pis[i]

        # merge aggressively
        c_consensus = r_consensus.make_modal(**kwargs)
        ref_cmap = c_consensus.cmap
        ref_modemap = c_consensus.modemap

        return ModalHDPMixture(self.pis, self.mus, self.sigmas, ref_cmap, ref_modemap, self.niter, self.m, self.s)

    def reorder(self, lookup):
        return OrderedHDPMixture(self.pis, self.mus, self.sigmas, lookup, self.niter, self.m, self.s, self.ident)

class OrderedHDPMixture(HDPMixture):
    def __init__(self, pis, mus, sigmas, lookup, niter=1, m=0, s=1, identified=False):
        super(OrderedHDPMixture, self).__init__(pis, mus, sigmas, niter, m, s, identified)
        self.lookup = lookup

    def __add__(self, k):
        return super(OrderedHDPMixture, self).__add__(k).reorder(self.lookup)

    def __radd__(self, k):
        return super(OrderedHDPMixture, self).__radd__(k).reorder(self.lookup)

    def __sub__(self, k):
        return super(OrderedHDPMixture, self).__sub__(k).reorder(self.lookup)

    def __rsub__(self, k):
        return super(OrderedHDPMixture, self).__rsub__(k).reorder(self.lookup)

    def __mul__(self, k):
        return super(OrderedHDPMixture, self).__mul__(k).reorder(self.lookup)

    def __rmul__(self, k):
        return super(OrderedHDPMixture, self).__rmul__(k).reorder(self.lookup)

    def _getData(self, key):
        return super(OrderedHDPMixture, self)._getData(key).reorder(self.lookup)

    def classify(self, x, **kwargs):
        z = super(OrderedHDPMixture, self).classify(x, **kwargs)
        lut = np.array([self.lookup[i] for i in range(len(self))])
        return lut[z]

class ModalHDPMixture(HDPMixture):
    def __init__(self, pis, mus, sigmas, cmap, modemap, niter=1, m=None, s=None):
        '''
        ModalHDPMixture(clusters)
        cluster = HDPMixture object
        cmap = map of modal clusters to component clusters
        modes = array of mode locations
        '''
        self.pis = pis.copy()
        self.mus = mus.copy()
        self.sigmas = sigmas.copy()
        self.cmap = cmap
        self.modemap = modemap
        self.niter = niter

        if m is not None:
            self.m = m
        else:
            self.m = 0
        if s is not None:
            self.s = s
        else:
            self.s = 1

    def __add__(self,k):
        return ModalHDPMixture(self.pis, self.mus+k, self.sigmas, self.cmap, self.modemap, self.niter, self.m, self.s)

    def __radd__(self, k):
        return ModalHDPMixture(self.pis, k+self.mus, self.sigmas, self.cmap, self.modemap, self.niter, self.m, self.s)

    def __sub__(self, k):
        return ModalHDPMixture(self.pis, self.mus-k, self.sigmas, self.cmap, self.modemap, self.niter, self.m, self.s)

    def __rsub__(self,k):
        return ModalHDPMixture(self.pis, k-self.mus, self.sigmas, self.cmap, self.modemap, self.niter, self.m, self.s)

    def __mul__(self, k):
        if isinstance(k, Number):
            new_mu = self.mus * k
            new_sigma = k * k * self.sigmas
        elif isinstance(k, ndarray):
            new_mu = dot(self.mus, k)
            new_sigma = np.array([dot(dot(k, i), k.T) for i in self.sigmas])
        else:
            raise TypeError('unsupported type: %s' % type(k))

        return ModalHDPMixture(self.pis, new_mu, new_sigma, self.cmap, self.modemap, self.niter, self.m, self.s)

    def __rmul__(self, x):
        if isinstance(k, Number):
            new_mu = self.mus * k
            new_sigma = k * k * self.sigmas
        elif isinstance(k, ndarray):
            new_mu = dot(k, self.mus)
            new_sigma = np.array([dot(dot(k, i), k.T) for i in self.sigmas])
        else:
            raise TypeError('unsupported type: %s' % type(k))

        return ModalHDPMixture(self.pis, new_mu, new_sigma, self.cmap, self.modemap, self.niter, self.m, self.s)
        
    def _getData(self, key):
        pis = self.pis[key, :]
        mus = (self.mus - self.m) / self.s
        sigmas = (self.sigmas) / outer(self.s, self.s)
        clsts = [DPCluster(i, j, k, l, m) for i, j, k, l, m in
                 zip(pis, self.mus, self.sigmas, mus, sigmas)]
        return ModalDPMixture(clsts, self.cmap, self.modemap, self.niter, self.m, self.s)

    @property
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

    def prob(self, x):
        return array([r.prob(x) for r in self])

    def classify(self, x, **kwargs):
        return array([r.classify(x, **kwargs) for r in self])

    def reorder(self, lookup):
        return OrderedModalHDPMixture(self.pis, self.mus, self.sigmas, self.cmap, self.modemap, lookup, self.niter, self.m, self.s)

    def enumerate_modes(self):
        for i in range(len(self.modes)):
            yield i, self.modes[i]


class OrderedModalHDPMixture(ModalHDPMixture):
    def __init__(self, pis, mus, sigmas, cmap, modemap, lookup, niter, m=None, s=None):
        super(OrderedModalHDPMixture, self).__init__(pis, mus, sigmas, cmap, modemap, niter, m, s)
        self.lookup = lookup

    def __add__(self, k):
        return super(OrderedModalHDPMixture, self).__add__(k).reorder(self.lookup)

    def __radd__(self, k):
        return super(OrderedModalHDPMixture, self).__radd__(k).reorder(self.lookup)

    def __sub__(self, k):
        return super(OrderedModalHDPMixture, self).__sub__(k).reorder(self.lookup)

    def __rsub__(self, k):
        return super(OrderedModalHDPMixture, self).__rsub__(k).reorder(self.lookup)

    def __mul__(self, k):
        return super(OrderedModalHDPMixture, self).__mul__(k).reorder(self.lookup)

    def __rmul__(self, k):
        return super(OrderedModalHDPMixture, self).__rmul__(k).reorder(self.lookup)
    
    def _getData(self, key):
        return super(OrderedModalHDPMixture, self)._getData(key).reorder(self.lookup)

    def enumerate_modes(self):
        for i in range(len(self.modes)):
            yield self.lookup[i], self.modes[i]

    def classify(self, x, **kwargs):
        z = super(OrderedModalHDPMixture, self).classify(x, **kwargs)
        lut = np.array([self.lookup[i] for i in range(len(self))])
        return lut[z]
