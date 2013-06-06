import unittest
import fcm
from fcm.statistics import DPMixture, DPCluster
from fcm.alignment import DiagonalAlignData, CompAlignData
import numpy as np
import numpy.testing as npt

class DiagAlignTestCase(unittest.TestCase):
    def setUp(self):
        self.mu1 = np.array([0, 0, 0])
        self.sig = 2*np.eye(3)
        self.mu2 = np.array([5, 5, 5])

        self.clust1 = DPCluster(.5, self.mu1, self.sig)
        self.clust2 = DPCluster(.5, self.mu2, self.sig)
        self.clusters = [self.clust1, self.clust2]
        self.x = DPMixture(self.clusters,niter=1,identified=True)
       
        self.Diag = DiagonalAlignData(self.x, size=100000)
        self.Comp = CompAlignData(self.x, size=100000)

        
    def testDiagAlign(self):
        y = self.x+np.array([1,-1,1])
        a,b = self.Diag.align(y, xtol=0.1, maxiter=100, maxfun=100)
        npt.assert_array_almost_equal(a, np.eye(3), decimal=1)
        npt.assert_array_almost_equal(b, np.array([-1,1,-1]), decimal=1)

    def testCompAlign(self):
        m = np.array([[1,0,.2],[0,1,0],[0,0,1]])
        y = self.x*m
        print y.mus
        a,b = self.Comp.align(y, x0=np.array([0,-.1,0,0,0,0,]))
        npt.assert_array_almost_equal(a, np.linalg.inv(m), decimal=1)
        npt.assert_array_almost_equal(b, np.array([0,0,0]), decimal=1)
        
if __name__ == '__main__':
    suite1 = unittest.makeSuite(DiagAlignTestCase,'test')

    unittest.main()