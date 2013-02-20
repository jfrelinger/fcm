import unittest
from fcm.statistics import HDPMixture, DPMixture, DPCluster
from numpy import array, eye, dot, array
import numpy.testing
class HDPMixtureTestCase(unittest.TestCase):


    def setUp(self):
        self.pis = array([[.1,.1,.8], [.1,.8,.1],[.8,.1,.1]])
        self.mus = array([[2,2],[3,3],[4,4]])
        self.sigmas = array([eye(2),eye(2),eye(2)])
        
        self.mix = HDPMixture(self.pis,self.mus,self.sigmas)
        
    def test_arith(self):
        numpy.testing.assert_array_equal((self.mix+2).mus, self.mix.mus+2,
                                    "Failed addition")
        numpy.testing.assert_array_equal((self.mix*2).mus, self.mix.mus*2,
                                    "Failed addition")
        numpy.testing.assert_array_equal((self.mix*2).sigmas, self.mix.sigmas*4,
                                    "multiplication addition")
        
        numpy.testing.assert_array_equal((self.mix+2).pis, self.mix.pis,
                                    "addition changes pi values")
    def test_getitem(self):
        single = self.mix[0]
        self.assertIsInstance(single, DPMixture, 
                              'get item didn\'t return a DPMixture')
        
        sliced = self.mix[0:2]
        
        self.assertIsInstance(sliced, list,
                              'get slice didn\'t return a list')
        
        numpy.testing.assert_array_equal(self.mix[0].pis, self.pis[0,:], "returned pis wrong")
        
    def test_len(self):
        self.assertEqual(len(self.mix), 3, 'len returned the wrong value')
    
    def test_iter(self):
        count = 0
        for i in self.mix:
            self.assertIsInstance(i, DPMixture, 
                              'get itterator didn\'t return a DPMixture')
            count += 1
        self.assertEqual(count, 3, 'itterator didn\'t return all the elements')
    
    def test_stats(self):
        prob = self.mix.prob(array([1,2]))
        self.assertEqual(prob.shape, (3,3), 'probability return wrong shape')
        self.assertLessEqual(prob.max(), 1, 'probability returnd a value > 1')
        
        classified = self.mix.classify(array([1,2]))
        self.assertEqual(classified.shape, (3,), 'probability return wrong shape,')
        
    def test_average(self):
        avg = self.mix.average()
        numpy.testing.assert_array_equal(avg.mus, self.mix.mus, 'averaging got means wrong')
        numpy.testing.assert_array_equal(avg.sigmas, self.mix.sigmas, 'averaging got variance wrong')
        self.assertIsInstance(avg, HDPMixture, 'average returned wrong type of object')
        self.assertEqual(len(avg),3, 'average return the wrong number of mixtures, %d' % len(avg))
        
        
       
if __name__ == '__main__':
    unittest.main()