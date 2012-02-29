'''
Created on Oct 30, 2009

@author: jolly
'''

from warnings import warn
from numpy import zeros, outer, sum, eye, array
from numpy.random import multivariate_normal as mvn
from scipy.cluster import vq

from dpmix import DPNormalMixture, _has_gpu
if _has_gpu:
    import os
from dp_cluster import DPCluster, DPMixture
from kmeans import KMeans


class DPMixtureModel(object):
    '''
    Fits a DP Mixture model to a fcm dataset.
    
    '''


    def __init__(self, nclusts, iter=1000, burnin=100, last=5):
        '''
        DPMixtureModel(fcmdata, nclusts, iter=1000, burnin= 100, last= 5)
        fcmdata = a fcm data object
        nclusts = number of clusters to fit
        itter = number of mcmc itterations
        burning = number of mcmc burnin itterations
        last = number of mcmc itterations to draw samples from
        
        '''
        

        self.nclusts = nclusts
        self.iter = iter
        self.burnin = burnin
        self.last = last
        

        self.alpha0=1
        self.nu0=None
        self.Phi0=None
        self.prior_mu=None
        self.prior_sigma=None
        self.prior_pi=None
        self.alpha_a0=1
        self.alpha_b0=1
        
        self._prior_mu = None
        self._prior_pi = None
        self._prior_sigma = None

    def load_mu(self, mu):
        if len(mu.shape) > 2:
            raise ValueError('Shape of Mu is wrong')
        if len(mu.shape) == 2:
            (n, d) = mu.shape
        else:
            n = 1
            d = mu.shape[0]
        if n > self.nclusts:
            raise ValueError('number of proposed Mus grater then number of clusters')

        
        self.prior_mu = mu
        self.mu_d = d
        self._load_mu = True

    def _load_mu_at_fit(self):
        (n,d) = self.prior_mu.shape
        if d != self.d:
            raise ValueError('Dimension mismatch between Mus and Data')
        
        elif n < self.nclusts:
            self._prior_mu = zeros((self.nclusts, self.d))
            self._prior_mu[0:n, :] = (self.prior_mu.copy() - self.m) / self.s
            self._prior_mu[n:, :] = mvn(zeros((self.d,)), eye(self.d), self.nclusts - n)
        else:
            self._prior_mu = (self.prior_mu.copy() - self.m) / self.s
        
    def load_sigma(self, sigma):
        n, d = sigma.shape[0:2]
        if len(sigma.shape) > 3:
            raise ValueError('Shape of Sigma is wrong')

        if len(sigma.shape) == 2:
            sigma = array(sigma)

        if sigma.shape[1] != sigma.shape[2]:
            raise ValueError("Sigmas must be square matricies")
        
        
        if n > self.nclusts:
            raise ValueError('number of proposed Sigmass grater then number of clusters')

        self._load_sigma = True
        self.prior_sigma = sigma
    
    def _load_sigma_at_fit(self):
        n, d = self.prior_sigma.shape[0:2]

        if d != self.d:
            raise ValueError('Dimension mismatch between Sigmas and Data')

        elif n < self.nclusts:
            self._prior_sigma = zeros((self.nclusts, self.d, self.d))
            self._prior_sigma[0:n, :, :] = (self.prior_sigma.copy()) / outer(self.s, self.s)
            for i in range(n, self.nclusts):
                self._prior_sigma[i, :, :] = eye(self.d)
        else:
            self._prior_sigma = (self.prior_sigma.copy()) / outer(self.s, self.s)
        

    def load_pi(self, pi):
        tmp = array(pi)
        if len(tmp.shape) != 1:
            raise ValueError("Shape of pi is wrong")
        n = tmp.shape[0]
        if n > self.nclusts:
            raise ValueError('number of proposed Pis grater then number of clusters')

        if sum(tmp) > 1:
            raise ValueError('Proposed Pis sum to more then 1')
        if n < self.nclusts:
            self._prior_pi = zeros((self.nclusts))
            self._prior_pi[0:n] = tmp
            left = (1.0 - sum(tmp)) / (self.nclusts - n)
            for i in range(n, self.nclusts):
                self._prior_pi[i] = left
        else:
            self._prior_pi = tmp

        self._prior_pi = self._prior_pi.reshape((1, self.nclusts))

        self._load_pi = True

    def load_ref(self, ref):
        raise Exception("load_ref not implemented yet")

    
    def fit(self, fcmdata,  verbose=False):
        """
        fit the mixture model to the data
        use get_results() to get the fitted model
        """
        pnts = fcmdata.view()
        self.m = pnts.mean(0)
        self.s = pnts.std(0)
        self.data = (pnts - self.m) / self.s
        
        if len(self.data.shape) == 1:
            self.data = self.data.reshape((self.data.shape[0], 1))

        if len(self.data.shape) != 2:
            raise ValueError("pnts is the wrong shape")
        self.n, self.d = self.data.shape
        
        
        if self.prior_mu is not None:
            self._load_mu_at_fit()
        if self.prior_sigma is not None:
            self._load_sigma_at_fit()
        
        #TODO move hyperparameter settings here
        try:
            if _has_gpu and self.device is not None:
                os.environ['CUDA_DEVICE'] = self.device
        except AttributeError:
            pass
        self.cdp = DPNormalMixture(self.data,ncomp=self.nclusts,
                                    alpha0=self.alpha0, nu0=self.nu0,
                                    Phi0=self.Phi0,
                                    mu0=self._prior_mu, Sigma0=self._prior_sigma,
                                    weights0=self._prior_pi, alpha_a0=1,
                                    alpha_b0=self.alpha_b0, gpu=None)
        

        
        
        self.pi = zeros((self.nclusts * self.last))
        self.mus = zeros((self.nclusts * self.last, self.d))
        self.sigmas = zeros((self.nclusts * self.last, self.d, self.d))
        
        
        
        self.cdp.sample(self.iter, self.burnin, 1)

        self._run = True #we've fit the mixture model

        
        
        return self.get_results()

    def step(self, verbose=False):
        raise Exception("With the new CDP stepping is not supported")


    
    def get_results(self):
        """
        get the results of the fitted mixture model
        """
        
        if self._run:
            #pi = self.cdp.weights[-self.last] / sum(self.cdp.weight[-self.last])
            rslts = []
            for i in range(self.last):
                for j in range(self.nclusts):
                    tmp = DPCluster(self.cdp.weights[-i,j], (self.cdp.mu[-i,j] * self.s) + self.m, self.cdp.Sigma[-i,j] * outer(self.s, self.s))
                    tmp.nmu = self.cdp.mu[-i,j]
                    tmp.nsigma = self.cdp.Sigma[-i,j]
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



class KMeansModel(object):
    '''
    KmeansModel(data, k, iter=20, tol=1e-5)
    kmeans clustering model
    '''
    def __init__(self, data, k, iter=20, tol=1e-5):
        self.data = data.view()
        self.k = k
        self.iter = iter
        self.tol = tol

    def fit(self):
        self.r = vq.kmeans(self.data, self.k, iter=self.iter, thresh=self.tol)


    def get_results(self):
        return KMeans(self.r[0])
