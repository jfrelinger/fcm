from fcm.alignment.kldiv import eKLdiv, true_kldiv

import unittest
import fcm
from fcm.statistics import DPMixture, DPCluster
import numpy as np

class KLdivTestCase(unittest.TestCase):
    def setUp(self):
        self.mux = np.array([0, 0, 0])
        self.sigx = 2 * np.eye(3)

        self.muy = np.array([0, .5, .5])
        self.sigy = np.eye(3)

        self.clustx = DPCluster(1, self.mux, self.sigx)
        self.clustersx = [self.clustx]

        self.clusty = DPCluster(1, self.muy, self.sigy)
        self.clustersy = [self.clusty]

        self.x = DPMixture(self.clustersx, niter=1, identified=True)
        self.y = DPMixture(self.clustersy, niter=1, identified=True)

    def testTrueKLdiv(self):
        f = eKLdiv(self.x, self.y, 3000000)
        g = true_kldiv(self.mux, self.muy, self.sigx, self.sigy)
        self.assertAlmostEqual(f, g, 2, 'fialed to generate simialr ansers, f:%f, g:%f' % (f, g))

    def testRandomTrueKLdiv(self):
        y= self.y + np.random.uniform(-1,1,3)
        f = eKLdiv(self.x, y, 3000000)
        g = true_kldiv(self.mux, y.mus[0], self.sigx, y.sigmas[0])
        self.assertAlmostEqual(f, g, 2, 'fialed to generate simialr ansers, f:%f, g:%f' % (f, g))
