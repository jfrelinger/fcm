import unittest
from fcm.statistics import DPCluster, ModalDPMixture
from numpy import array, eye, all, ndarray, dot
from numpy.testing import assert_array_equal
from numpy.testing.utils import assert_equal
from scipy.misc import logsumexp


class ModalDp_clusterTestCase(unittest.TestCase):


    def setUp(self):
        self.mu1 = array([0,0,0])
        self.sig = eye(3)
        self.mu2 = array([5,5,5])
        
        self.clst1 = DPCluster(.25, self.mu1, self.sig)
        self.clst2 = DPCluster(.25, self.mu1, self.sig)
        self.clst3 = DPCluster(.5, self.mu2, self.sig)
        self.mix = ModalDPMixture([self.clst1, self.clst2, self.clst3], {0 : [0,1], 1 : [2]}, {0: self.mu1, 1:self.mu2})


    def tearDown(self):
        pass

    def testLen(self):
        self.assertEqual(len(self.mix), 2, 'length wrong')
        
    def testEnumerateClusters(self):
        for i,j in self.mix.enumerate_modes():
            self.assertIsInstance(i, int, 'wrong return value')
            self.assertIsInstance(j, ndarray, 'wrong return value')
            
    def testModes(self):
        assert all(self.mix.modes[0] == self.mu1)
        assert all(self.mix.modes[1] == self.mu2) 

    def testprob(self):
        pnt = array([1,1,1])

        for i in [self.clst1, self.clst2]:
            assert i.prob(pnt) <= 1, 'prob of clst %s is > 1' % i
            assert i.prob(pnt) >= 0, 'prob of clst %s is < 0' % i

        
    def testmixprob(self):
        pnt = array([1,1,1])
        assert self.mix.prob(pnt)[0] == (self.clst1.prob(pnt)+self.clst2.prob(pnt)), 'mixture generates different prob then compoent 1'
        assert self.mix.prob(pnt)[1] == self.clst3.prob(pnt), 'mixture generates different prob then compoent 2'
    
    def testmixproblogged(self):
        pnt = array([1,1,1])
        assert self.mix.prob(pnt, logged=True)[0] == logsumexp([self.clst1.prob(pnt, logged=True),self.clst2.prob(pnt, logged=True)]), 'mixture generates different prob then compoent 1'
        assert self.mix.prob(pnt)[1] == self.clst3.prob(pnt), 'mixture generates different prob then compoent 2'    
    
    def testclassify(self):
        pnt = array([self.mu1, self.mu2])
        assert self.mix.classify(array([self.mu1, self.mu2, self.mu1, self.mu2, self.mu1, self.mu2])).tolist() == [0,1,0,1,0,1], 'classify not working'
        assert self.mix.classify(pnt)[0] == 0, 'classify classifys mu1 as belonging to something else'
        assert self.mix.classify(pnt)[1] == 1, 'classify classifys m21 as belonging to something else'

    def testDraw(self):
        x = self.mix.draw(10)
        assert x.shape[0] == 10, "Number of drawed rows is wrong"
        assert x.shape[1] == 3, "number of drawed columns is wrong"

    def testarith(self):
        adder = 3
        array_adder = array([1, 2, 3])
        mat_adder = eye(3)

        # add
        b = self.mix + adder
        self.assertIsInstance(b, ModalDPMixture, 'integer addition return wrong type')
        assert_equal(b.mus[0], self.mix.mus[0] + adder,
                     'integer addition returned wrong value')
        assert_equal(b.modes[0], self.mix.modes[0] + adder)

        c = self.mix + array_adder
        self.assertIsInstance(c, ModalDPMixture, 'array addition return wrong type')
        assert_array_equal(c.mus[0], self.mix.mus[0] + array_adder,
                     'array addition returned wrong value')
        assert_equal(c.modes[0], self.mix.modes[0] + array_adder)

        # radd
        b = adder + self.mix
        self.assertIsInstance(b, ModalDPMixture, 'integer addition return wrong type')
        assert_array_equal(b.mus[0], adder + self.mix.mus[0],
                     'integer addition returned wrong value')

        c = array_adder + self.mix
        self.assertIsInstance(c, ModalDPMixture, 'array addition return wrong type')
        assert_array_equal(c.mus[0], array_adder + self.mix.mus[0],
                     'array addition returned wrong value')

        # sub
        b = self.mix - adder
        self.assertIsInstance(b, ModalDPMixture, 'integer subtraction return wrong type')
        assert_array_equal(b.mus[0], self.mix.mus[0] - adder,
                     'integer subtraction returned wrong value')

        c = self.mix - array_adder
        self.assertIsInstance(c, ModalDPMixture, 'array subtraction return wrong type')
        assert_array_equal(c.mus[0], self.mix.mus[0] - array_adder,
                     'array subtraction returned wrong value')

        # rsub
        b = adder - self.mix
        self.assertIsInstance(b, ModalDPMixture, 'integer subtraction return wrong type')
        assert_array_equal(b.mus[0], adder - self.mix.mus[0],
                     'integer subtraction returned wrong value')

        c = array_adder - self.mix
        self.assertIsInstance(c, ModalDPMixture, 'array subtraction return wrong type')
        assert_array_equal(c.mus[0], array_adder - self.mix.mus[0],
                     'array subtraction returned wrong value')
        # mul
        b = self.mix * adder
        self.assertIsInstance(b, ModalDPMixture, 'integer multiplication return wrong type')
        assert_array_equal(b.mus[0], self.mix.mus[0] * adder,
                     'integer multiplication returned wrong value')

        c = self.mix * array_adder
        self.assertIsInstance(c, ModalDPMixture, 'array multiplicaton return wrong type')
        assert_array_equal(c.mus[0], dot(self.mix.mus[0], array_adder),
                     'array multiplication returned wrong value')

        d = self.mix * mat_adder
        self.assertIsInstance(d, ModalDPMixture, 'array multiplicaton return wrong type')
        assert_array_equal(d.mus[0], dot(self.mix.mus[0], mat_adder),
                     'array multiplication returned wrong value')
        

        # rmul
        b = adder * self.mix
        self.assertIsInstance(b, ModalDPMixture, 'integer multiplication return wrong type')
        assert_array_equal(b.mus[0], adder * self.mix.mus[0],
                     'integer multiplication returned wrong value')
        assert_array_equal(b.modes[0], self.mix.modes[0] * adder)
        
        c = array_adder * self.mix
        self.assertIsInstance(c, ModalDPMixture, 'array multiplication return wrong type')
        assert_array_equal(c.mus[0], dot(array_adder, self.mix.mus[0]),
                     'array multiplication returned wrong value')
        assert_array_equal(c.modes[0], dot(array_adder,self.mix.modes[0]))
        
        d = mat_adder * self.mix
        self.assertIsInstance(d, ModalDPMixture, 'array multiplicaton return wrong type')
        assert_array_equal(d.mus[0], dot(mat_adder, self.mix.mus[0]),
                     'array multiplication returned wrong value')
        assert_array_equal(d.modes[0], dot(mat_adder,self.mix.modes[0]))
        assert_array_equal(d.sigmas[0], dot(mat_adder,dot(self.mix.sigmas[0], mat_adder)),
                           'array multiplcation failed')
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()