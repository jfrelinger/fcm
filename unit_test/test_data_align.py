import unittest
import fcm
from fcm.statistics import DPMixture, DPCluster
from fcm.alignment import DiagonalAlignData, CompAlignData, FullAlignData
import numpy as np
import numpy.testing as npt

class DiagAlignTestCase(unittest.TestCase):
    def setUp(self):
        self.mu1 = np.array([0, 0, 0])
        self.sig = 2 * np.eye(3)
        self.mu2 = np.array([5, 5, 5])

        self.clust1 = DPCluster(.5, self.mu1, self.sig)
        self.clust2 = DPCluster(.5, self.mu2, self.sig)
        self.clusters = [self.clust1, self.clust2]
        self.x = DPMixture(self.clusters, niter=1, identified=True)

        self.Diag = DiagonalAlignData(self.x, size=100000)
        self.Comp = CompAlignData(self.x, size=100000)
        self.Full = FullAlignData(self.x, size=200000)


    def testDiagAlign(self):
        y = self.x + np.array([1, -1, 1])
        lb = np.array([0.1,0.1,0.1,-np.inf,-np.inf,-np.inf])
        #a, b = self.Diag.align(y, method='TNC', bounds=np.array([(0.5, 2), (None, None)]), tol=1e-8, options={'disp': False})
        a, b, f, s, m = self.Diag.align(y, solver='ralg', stencil=3)
        assert s, 'failed to converge'
        npt.assert_array_almost_equal(a, np.eye(3), decimal=1)
        npt.assert_array_almost_equal(b, np.array([-1, 1, -1]), decimal=1)

    def testCompAlign(self):
        m = np.array([[1, 0, .2], [0, 1, 0], [0, 0, 1]])
        lb = np.ones(6)*-.2
        ub = np.ones(6)*.2
        y = self.x * m
        a, b, f, s, _ = self.Comp.align(y, lb=lb, ub=ub, fEnough=0, ftol=1e-16, xtol=1e-16)
        assert s, 'failed to converge'
        npt.assert_array_almost_equal(a, np.linalg.inv(m), decimal=1)
        npt.assert_array_almost_equal(b, np.array([0, 0, 0]), decimal=1)
        npt.assert_array_almost_equal((y * a).mus, self.x.mus, decimal=1)

    def testFullAlign(self):
        m = np.array([[1, 0, .2], [0, 1, 0], [0, 0, 1]])
        y = self.x * m
        a, b, f, s, _ = self.Full.align(y, ftol=1e-10, gtol=1e-10, xtol=1e-10)
        assert s, 'failed to converge'
        npt.assert_array_almost_equal(a, np.linalg.inv(m), decimal=1)
        npt.assert_array_almost_equal(b, np.array([0, 0, 0]), decimal=1)
        npt.assert_array_almost_equal(((y * a) + b).mus, self.x.mus, decimal=1)

    def testFullAlignExclude(self):
        m = np.array([[1, 0, .2], [0, 1, 0], [0, 0, 1]])
        y = self.x * m
        x0 = np.hstack((np.eye(3).flatten(), np.zeros(3)))
        Full = FullAlignData(self.x, size=200000, exclude=[0])
        a, b, f, s, _ = Full.align(y, x0=x0, ftol=1e-8)
        npt.assert_array_almost_equal(a[0], np.array([1,0,0]), decimal=1)
        npt.assert_array_almost_equal(a[:,0], np.array([1,0,0]), decimal=1)


if __name__ == '__main__':
    suite1 = unittest.makeSuite(DiagAlignTestCase, 'test')

    unittest.main()
