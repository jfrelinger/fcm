import unittest
from fcm.statistics import DPCluster, ModalHDPMixture, ModalDPMixture
from numpy import array, eye, all, vstack
import numpy


class ModalHDp_clusterTestCase(unittest.TestCase):


    def setUp(self):
        self.mu1 = array([0, 0, 0])
        self.sig = eye(3)
        self.mu2 = array([5, 5, 5])
        self.mus = array([self.mu1, self.mu2])
        self.sigmas = array([self.sig, self.sig])
        self.pis = array([[.5, .5], [.4, .6]])
        self.cmap = {0: [0], 1: [1]}
        self.modes = {0: self.mu1, 1: self.mu2}


        self.clst1 = DPCluster(.5, self.mu1, self.sig)
        self.clst2 = DPCluster(.5, self.mu2, self.sig)
        self.clst3 = DPCluster(.4, self.mu1, self.sig)
        self.clst4 = DPCluster(.6, self.mu2, self.sig)
        self.clsts = [self.clst1, self.clst2, self.clst3, self.clst4]

        self.mix = ModalHDPMixture(self.pis, self.mus, self.sigmas, self.cmap, self.modes)


    def tearDown(self):
        pass

    def testModes(self):
        assert all(self.mix.modes()[0] == self.mu1)
        assert all(self.mix.modes()[1] == self.mu2)

    def testmixprob(self):
        pnt = array([1, 1, 1])

        r = self.mix.prob(pnt)
        self.assertEqual(r.shape, (2, 2), 'prob return wrong shape')
        r = r.flatten()
        for i, j in enumerate(self.clsts):
            self.assertEqual(j.prob(pnt), r[i], 'prob calculation failed')

    def testgetitem(self):
        r = self.mix[0]
        self.assertIsInstance(r, ModalDPMixture, 'get item returned wrong type')
        numpy.testing.assert_array_equal(r.modes()[0], self.mix.modes()[0], 'modes changed under getitem')

    
    def testclassify(self):
        pnt = array([self.mu1, self.mu2])
        
        numpy.testing.assert_array_equal(self.mix.classify(array([self.mu1, self.mu2, self.mu1, self.mu2, self.mu1, self.mu2])), array([[0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1]]), 'classify not working')
        numpy.testing.assert_array_equal(self.mix.classify(pnt)[0], array([0, 1]), 'classify classifys mu1 as belonging to something else')
        numpy.testing.assert_array_equal(self.mix.classify(pnt)[1], array([0, 1]), 'classify classifys m21 as belonging to something else')


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
