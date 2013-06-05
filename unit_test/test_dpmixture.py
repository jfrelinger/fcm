'''
Created on Oct 30, 2009

@author: Jacob Frelinger
'''
import unittest
from fcm.statistics import DPCluster, DPMixture
from numpy import array, eye, all, dot
from numpy.testing import assert_array_equal
from numpy.testing.utils import assert_equal


class Dp_mixtureTestCase(unittest.TestCase):


    def setUp(self):
        self.mu1 = array([0, 0, 0])
        self.sig = eye(3)
        self.mu2 = array([5, 5, 5])

        self.clust1 = DPCluster(.5/3, self.mu1, self.sig)
        self.clust2 = DPCluster(.5/3, self.mu2, self.sig)
        self.clusters = [self.clust1, self.clust2, self.clust1, self.clust2,
                         self.clust1, self.clust2]
        self.mix = DPMixture(self.clusters,niter=3,identified=True)


    def tearDown(self):
        pass


    def testprob(self):
        pnt = array([[1, 1, 1]])
        print pnt.shape
        for i in [self.clust1, self.clust2]:
            print i.prob(pnt)
            assert i.prob(pnt) <= 1, 'prob of clst %s is > 1' % i
            assert i.prob(pnt) >= 0, 'prob of clst %s is < 0' % i


    def testmixprob(self):
        pnt = array([1, 1, 1])
        assert self.mix.prob(pnt)[0] == self.clust1.prob(pnt), 'mixture generates different prob then compoent 1'
        assert self.mix.prob(pnt)[1] == self.clust2.prob(pnt), 'mixture generates different prob then compoent 2'

    def testclassify(self):
        pnt = array([self.mu1, self.mu2])
        assert self.mix.classify(pnt)[0] == 0, 'classify classifys mu1 as belonging to something else'
        assert self.mix.classify(pnt)[1] == 1, 'classify classifys m21 as belonging to something else'

    def testMakeModal(self):

        modal = self.mix.make_modal()
#        modal = ModalDPMixture([self.clst1, self.clst2],
#                                { 0: [0], 1: [1]},
#                                [self.mu1, self.mu2])
        pnt = array([self.mu1, self.mu2])
        assert modal.classify(array([self.mu1, self.mu2, self.mu1, self.mu2, self.mu1, self.mu2])).tolist() == [0, 1, 0, 1, 0, 1], 'classify not working'
        assert self.mix.classify(self.mu1) == modal.classify(self.mu1), 'derived modal mixture is wrong'
        assert self.mix.classify(pnt)[0] == modal.classify(pnt)[0], 'derived modal mixture is wrong'
        assert self.mix.classify(pnt)[1] == modal.classify(pnt)[1], 'derived modal mixture is wrong'
        
        modal = self.mix.make_modal(delta=9)
        assert modal.classify(array([self.mu1, self.mu2, self.mu1, self.mu2, self.mu1, self.mu2])).tolist() == [0, 0, 0, 0, 0, 0], 'classify not working'
        

    def testAverage(self):
        clst1 = DPCluster(0.5, self.mu1, self.sig)
        clst3 = DPCluster(0.5, self.mu1, self.sig)
        clst5 = DPCluster(0.5, self.mu1, self.sig)
        clst7 = DPCluster(0.5, self.mu1, self.sig)
        clst2 = DPCluster(0.5, self.mu2, self.sig)
        clst4 = DPCluster(0.5, self.mu2, self.sig)
        clst6 = DPCluster(0.5, self.mu2, self.sig)
        clst8 = DPCluster(0.5, self.mu2, self.sig)

        mix = DPMixture([clst1, clst2, clst3, clst4, clst5, clst6, clst7, clst8], niter=4)
        avg = mix.average()

        assert len(avg.clusters) == 2
        assert all(avg.mus[0] == self.mu1)
        assert all(avg.mus[1] == self.mu2)
        assert all(avg.sigmas[0] == self.sig)
        assert all(avg.sigmas[1] == self.sig)
        assert avg.pis[0] == 0.5, 'pis should be 0.5 but is %f' % avg.pis()[0]
        assert avg.pis[1] == 0.5, 'pis should be 0.5 but is %f' % avg.pis()[0]

    def testLast(self):
        clst1 = DPCluster(0.5, self.mu1, self.sig)
        clst3 = DPCluster(0.5, self.mu1 + 3, self.sig)
        clst5 = DPCluster(0.5, self.mu1 + 5, self.sig)
        clst7 = DPCluster(0.5, self.mu1 + 7, self.sig)
        clst2 = DPCluster(0.5, self.mu2 + 2, self.sig)
        clst4 = DPCluster(0.5, self.mu2 + 4, self.sig)
        clst6 = DPCluster(0.5, self.mu2 + 6, self.sig)
        clst8 = DPCluster(0.5, self.mu2 + 8, self.sig)

        mix = DPMixture([clst1, clst2, clst3, clst4, clst5, clst6, clst7, clst8], niter=4)

        new_r = mix.last()
        assert len(new_r.clusters) == 2
        assert all(new_r.clusters[0].mu == clst7.mu)
        assert all(new_r.clusters[1].mu == clst8.mu)

        new_r = mix.last(2)
        assert len(new_r.clusters) == 4
        assert all(new_r.clusters[0].mu == clst5.mu)
        assert all(new_r.clusters[1].mu == clst6.mu)
        assert all(new_r.clusters[2].mu == clst7.mu)
        assert all(new_r.clusters[3].mu == clst8.mu)

        try:
            new_r = mix.last(10)
        except ValueError:
                pass

    def testDraw(self):
        x = self.mix.draw(10)
        assert x.shape[0] == 10, "Number of drawed rows is wrong"
        assert x.shape[1] == 3, "number of drawed columns is wrong"

    def testarith(self):
        adder = 3
        array_adder = array([1, 2, 3])
        mat_adder = 2*eye(3)

        # add
        b = self.mix + adder
        self.assertIsInstance(b, DPMixture, 'integer addition return wrong type')
        assert_equal(b.mus[0], self.mix.mus[0] + adder,
                     'integer addition returned wrong value')

        c = self.mix + array_adder
        self.assertIsInstance(c, DPMixture, 'array addition return wrong type')
        assert_array_equal(c.mus[0], self.mix.mus[0] + array_adder,
                     'array addition returned wrong value')


        # radd
        b = adder + self.mix
        self.assertIsInstance(b, DPMixture, 'integer addition return wrong type')
        assert_array_equal(b.mus[0], adder + self.mix.mus[0],
                     'integer addition returned wrong value')

        c = array_adder + self.mix
        self.assertIsInstance(c, DPMixture, 'array addition return wrong type')
        assert_array_equal(c.mus[0], array_adder + self.mix.mus[0],
                     'array addition returned wrong value')

        # sub
        b = self.mix - adder
        self.assertIsInstance(b, DPMixture, 'integer subtraction return wrong type')
        assert_array_equal(b.mus[0], self.mix.mus[0] - adder,
                     'integer subtraction returned wrong value')

        c = self.mix - array_adder
        self.assertIsInstance(c, DPMixture, 'array subtraction return wrong type')
        assert_array_equal(c.mus[0], self.mix.mus[0] - array_adder,
                     'array subtraction returned wrong value')

        # rsub
        b = adder - self.mix
        self.assertIsInstance(b, DPMixture, 'integer subtraction return wrong type')
        assert_array_equal(b.mus[0], adder - self.mix.mus[0],
                     'integer subtraction returned wrong value')

        c = array_adder - self.mix
        self.assertIsInstance(c, DPMixture, 'array subtraction return wrong type')
        assert_array_equal(c.mus[0], array_adder - self.mix.mus[0],
                     'array subtraction returned wrong value')
        # mul
        b = self.mix * adder
        self.assertIsInstance(b, DPMixture, 'integer multiplication return wrong type')
        assert_array_equal(b.mus[0], self.mix.mus[0] * adder,
                     'integer multiplication returned wrong value')

        c = self.mix * array_adder
        self.assertIsInstance(c, DPMixture, 'array multiplicaton return wrong type')
        assert_array_equal(c.mus[0], dot(self.mix.mus[0], array_adder),
                     'array multiplication returned wrong value')

        d = self.mix * mat_adder
        self.assertIsInstance(d, DPMixture, 'array multiplicaton return wrong type')
        assert_array_equal(d.mus[0], dot(self.mix.mus[0], mat_adder),
                     'array multiplication returned wrong value')
        

        # rmul
        b = adder * self.mix
        self.assertIsInstance(b, DPMixture, 'integer multiplication return wrong type')
        assert_array_equal(b.mus[0], adder * self.mix.mus[0],
                     'integer multiplication returned wrong value')

        c = array_adder * self.mix
        self.assertIsInstance(c, DPMixture, 'array multiplication return wrong type')
        assert_array_equal(c.mus[0], dot(array_adder, self.mix.mus[0]),
                     'array multiplication returned wrong value')
        
        d = mat_adder * self.mix
        self.assertIsInstance(d, DPMixture, 'array multiplicaton return wrong type')
        assert_array_equal(d.mus[0], dot(mat_adder, self.mix.mus[0]),
                     'array multiplication returned wrong value')
        
        assert_array_equal(d.sigmas[0], dot(mat_adder,dot(self.mix.sigmas[0], mat_adder)),
                           'array multiplcation failed')

    def testgetitem(self):
        assert_equal(self.mu1, self.mix[0].mu, 'getitem failed')
        self.mix[0] = self.clust2
        assert_equal(self.mu2, self.mix[0].mu, 'getitem failed')
        self.mix[0] = self.clust1
        
    def testgetiteration(self):
        self.assertIsInstance(self.mix.get_iteration(2), DPMixture, 
                              'get_iteration failed')
        self.assertEqual(len(self.mix.get_iteration(2).clusters), 2, 
                         'get_iteration return wrong number of clusters')
        self.assertIsInstance(self.mix.get_iteration([0,2]), DPMixture, 
                              'get_iteration failed')
        self.assertEqual(len(self.mix.get_iteration([0,2]).clusters), 4, 
                         'get_iteration return wrong number of clusters')
        
    def testEnumerateClusters(self):
        for i,j in self.mix.enumerate_clusters():
            self.assertIsInstance(i, int)
            self.assertIs(j, self.mix[i], 'fialed to return the right cluster when enumerating')
    
    def testEnumeratePis(self):
        for i,j in self.mix.enumerate_pis():
            self.assertIsInstance(i, int)
            self.assertIs(j, self.mix[i].pi, 'fialed to return the right pi when enumerating')
    
    def testEnumerateMus(self):
        for i,j in self.mix.enumerate_mus():
            self.assertIsInstance(i, int)
            self.assertIs(j, self.mix[i].mu, 'fialed to return the right mean when enumerating')
            
    def testEnumerateSigmas(self):
        for i,j in self.mix.enumerate_sigmas():
            self.assertIsInstance(i, int)
            self.assertIs(j, self.mix[i].sigma, 'fialed to return the right covariance when enumerating')
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()

