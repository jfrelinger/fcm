'''
Created on Oct 30, 2009

@author: jolly
'''

from warnings import warn
from numpy import zeros, outer, sum, eye, array
from numpy.random import multivariate_normal as mvn
from scipy.cluster import vq

from cdp import cdpcluster
from dp_cluster import DPCluster, DPMixture
from kmeans import KMeans


class DPMixtureModel(object):
    '''
    Fits a DP Mixture model to a fcm dataset.
    
    '''


    def __init__(self, fcmdata, nclusts, iter=1000, burnin=100, last=5):
        '''
        DPMixtureModel(fcmdata, nclusts, iter=1000, burnin= 100, last= 5)
        fcmdata = a fcm data object
        nclusts = number of clusters to fit
        itter = number of mcmc itterations
        burning = number of mcmc burnin itterations
        last = number of mcmc itterations to draw samples from
        
        '''
        pnts = fcmdata.view()
        self.m = pnts.mean(0)
        self.s = pnts.std(0)
        self.data = (pnts - self.m) / self.s

        self.nclusts = nclusts
        self.iter = iter
        self.burnin = burnin
        self.last = last
        if len(self.data.shape) == 1:
            self.data = self.data.reshape((self.data.shape[0], 1))

        if len(self.data.shape) != 2:
            raise ValueError("pnts is the wrong shape")
        self.n, self.d = self.data.shape

        self.cdp = cdpcluster(self.data)
        try:
            self.cdp.getdevice()
            # if the above passed we're cuda enabled...
            if self.nclusts % 16:
                tmp = self.nclusts + (16 - (self.nclusts % 16))
                warn("Number of clusters, %d, is not a multiple of 16, increasing it to %d" % (self.nclusts, tmp))
                self.nclusts = tmp
        except RuntimeError:
            pass
        self.pi = zeros((nclusts * last))
        self.mus = zeros((nclusts * last, self.d))
        self.sigmas = zeros((nclusts * last, self.d, self.d))

        # self.cdp.setphi0(0.5)
        # self.cdp.setgamma(5)
        # self.cdp.setaa(5)

        #setup samplers...
        self.samplem = True
        self.samplePhi = True
        self.samplew = True
        self.sampleq = True
        self.samplealpha0 = True
        self.sample_mu = True
        self.sample_sigma = True
        self.samplek = True
        self.sample_pi = True
        self.samplealpha = True

        #load hyper paramters
        self.lambda0 = self.cdp.getlambda0()
        self.phi0 = self.cdp.getphi0()
        self.nu0 = self.cdp.getnu0()
        #self.gamma = self.cdp.getgamma()
        # per cliburn: override the default gamma to be 3
        self.gamma = 3
        self.nu = self.cdp.getnu()
        self.e0 = self.cdp.gete0()
        self.f0 = self.cdp.getf0()
        self.ee = self.cdp.getee()
        self.aa = self.cdp.getaa()
        self.ff = self.cdp.getff()



        self._prerun = False
        self._run = False
        self._load_mu = False
        self._load_sigma = False
        self._load_pi = False
        self._load_ref = False


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
        elif d != self.d:
            raise ValueError('Dimension mismatch between Mus and Data')
        elif n < self.nclusts:
            self._prior_mu = zeros((self.nclusts, self.d))
            self._prior_mu[0:n, :] = (mu.copy() - self.m) / self.s
            self._prior_mu[n:, :] = mvn(zeros((self.d,)), eye(self.d), self.nclusts - n)
        else:
            self._prior_mu = (mu.copy() - self.m) / self.s

        self._load_mu = True


    def load_sigma(self, sigma):
        if len(sigma.shape) > 3:
            raise ValueError('Shape of Sigma is wrong')

        if len(sigma.shape) == 2:
            sigma = array(sigma)

        if sigma.shape[1] != sigma.shape[2]:
            raise ValueError("Sigmas must be square matricies")

        n, d = sigma.shape[0:2]

        if n > self.nclusts:
            raise ValueError('number of proposed Sigmass grater then number of clusters')

        if d != self.d:
            raise ValueError('Dimension mismatch between Sigmas and Data')

        elif n < self.nclusts:
            self._prior_sigma = zeros((self.nclusts, self.d, self.d))
            self._prior_sigma[0:n, :, :] = (sigma.copy()) / outer(self.s, self.s)
            for i in range(n, self.nclusts):
                self._prior_sigma[i, :, :] = eye(self.d)
        else:
            self._prior_sigma = (sigma.copy()) / outer(self.s, self.s)

        self._load_sigma = True

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
        ref = array(ref)
        if len(ref.shape) != 1:
            raise ValueError("reference assignments are the wrong shape")
        if ref.shape[0] != self.data.shape[0]:
            raise ValueError("Reference assignments are not the same as the number of points")

        self._ref = ref
        self._load_ref = True

    def _setup(self, verbose):
        if not self._prerun:
            self.cdp.setT(self.nclusts)
            self.cdp.setJ(1)
            self.cdp.setBurnin(self.burnin)
            self.cdp.setIter(self.iter - self.last)

            self.cdp.samplem(self.samplem)
            self.cdp.samplePhi(self.samplePhi)
            self.cdp.samplew(self.samplew)
            self.cdp.sampleq(self.sampleq)
            self.cdp.samplealpha0(self.samplealpha0)
            self.cdp.samplemu(self.sample_mu)
            self.cdp.sampleSigma(self.sample_sigma)
            self.cdp.samplek(self.samplek)
            self.cdp.samplep(self.sample_pi)
            self.cdp.samplealpha(self.samplealpha)

            self.cdp.setlambda0(self.lambda0)
            self.cdp.setphi0(self.phi0)
            self.cdp.setnu0(self.nu0)
            self.cdp.setgamma(self.gamma)
            self.cdp.setnu(self.nu)
            self.cdp.sete0(self.e0)
            self.cdp.setf0(self.f0)
            self.cdp.setee(self.ee)
            self.cdp.setaa(self.aa)
            self.cdp.setff(self.ff)



            self.cdp.makeResult()

            if verbose:
                self.cdp.setVerbose(True)
            if self._load_mu:
                self.cdp.loadMu(self._prior_mu)

            if self._load_sigma:
                self.cdp.loadSigma(self._prior_sigma)

            if self._load_pi:
                self.cdp.loadp(self._prior_pi)

            if self._load_ref:
                print "loading ref"
                print self._ref
                print self._ref.shape
                self.cdp.loadref(self._ref)


            self._prerun = True


    def fit(self, verbose=False):
        """
        fit the mixture model to the data
        use get_results() to get the fitted model
        """
        self._setup(verbose)
        self.cdp.run()


        self._run = True #we've fit the mixture model

        idx = 0
        n = self.burnin + self.iter - self.last + 1
        for i in range(self.last):
            for j in range(self.nclusts):
                self.pi[idx] = self._getpi(j)
                self.mus[idx, :] = self._getmu(j)
                self.sigmas[idx, :, :] = self._getsigma(j)
                idx += 1
            self.cdp.step()
            if verbose:
                print "it = %d" % (n + i)
        if verbose:
            print "Done"

    def step(self, verbose=False):
        self._setup(verbose)
        tpi = zeros((self.nclusts))
        tmus = zeros((self.nclusts, self.d))
        tsigmas = zeros((self.nclusts, self.d, self.d))
        self.cdp.step()
        for j in range(self.nclusts):
                tpi[j] = self._getpi(j)
                tmus[j, :] = self._getmu(j)
                tsigmas[j, :, :] = self._getsigma(j)

        rslts = []
        for i in range(self.nclusts):
            tmp = DPCluster(tpi[i], (tmus[i] * self.s) + self.m, tsigmas[i] * outer(self.s, self.s))
            tmp.nmu = tmus[i]
            tmp.nsigma = tsigmas[i]
            rslts.append(tmp)
        tmp = DPMixture(rslts, self.m, self.s)
        return tmp


    def _getpi(self, idx):
        return self.cdp.getp(idx)

    def _getmu(self, idx):
        tmp = zeros(self.d)
        for i in range(self.d):
            tmp[i] = self.cdp.getMu(idx, i)

        return tmp

    def _getsigma(self, idx):
        tmp = zeros((self.d, self.d))
        for i in range(self.d):
            for j in range(self.d):
                tmp[i, j] = self.cdp.getSigma(idx, i, j)

        return tmp



    def get_results(self):
        """
        get the results of the fitted mixture model
        """

        if self._run:
            self.pi = self.pi / sum(self.pi)
            rslts = []
            for i in range(self.last * self.nclusts):
                tmp = DPCluster(self.pi[i], (self.mus[i] * self.s) + self.m, self.sigmas[i] * outer(self.s, self.s))
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
