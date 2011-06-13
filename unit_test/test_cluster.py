import sys
#sys.path.append("/home/jolly/MyPython") 

from fcm.statistics import DPMixtureModel, DPMixture
import unittest
from numpy import array, eye, all
from numpy.random import multivariate_normal
from fcm.statistics.dp_cluster import DPMixture


class DPMixtureModel_TestCase(unittest.TestCase):
    def setUp(self):
        self.mu = array([0,0])
        self.sig = eye(2)
        self.pnts = multivariate_normal(self.mu, self.sig, 1000)
        self.k = 16
        self.niter = 10
        self.model = DPMixtureModel(self.k,self.niter,0,1)
        

    def testModel(self):
        r = self.model.fit(self.pnts, verbose=False)
        assert(isinstance(r, DPMixture))
        mus = r.mus()
        assert(mus.shape == (16,2))
        
    def testModel_prior(self):
        self.model.load_mu(self.mu.reshape(1,2))
        self.model.load_sigma(self.sig.reshape(1,2,2))
        r = self.model.fit(self.pnts, verbose=False)
        assert(isinstance(r, DPMixture))
        mus = r.mus()
        assert(mus.shape == (16,2))
        
if __name__ == '__main__':
    unittest.main()