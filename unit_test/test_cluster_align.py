import unittest
from fcm.statistics import DPCluster, DPMixture, OrderedDPMixture
from fcm.alignment.cluster_alignment import AlignMixture
import numpy as np


class ClusterAlignTestCase(unittest.TestCase):
    def setUp(self):
        mus = np.array([[0, 0], [0, 10], [10, 0], [10,10]])
        sigmas = np.array([np.eye(2),np.eye(2),np.eye(2), np.eye(2)])
        pis = np.array([1.0/3, 1.0/3, 1.0/3, 1.0/3])
        clusters = [DPCluster(i,j,k) for i,j,k in zip(pis,mus,sigmas)]
        
        self.mx = DPMixture(clusters[0:])
        self.my = DPMixture(clusters[1:])
        
    def test_kldiv(self):
        a = AlignMixture(self.mx, 'kldiv')
        r = a.align(self.my, max_cost=1)
        self.assertIsInstance(r, OrderedDPMixture, 'failed to return an ordered mixture')
        for i in r.lookup:
            self.assertEqual(i+1, r.lookup[i], 'assignment failed %d : %d' % (i, r.lookup[i]))
    
    def test_mean(self):
        a = AlignMixture(self.mx, 'mean')
        r = a.align(self.my, max_cost=1)
        self.assertIsInstance(r, OrderedDPMixture, 'failed to return an ordered mixture')
        for i in r.lookup:
            self.assertEqual(i+1, r.lookup[i], 'assignment failed %d : %d' % (i, r.lookup[i]))
        
    def test_class(self):
        a = AlignMixture(self.mx, 'class',)
        r = a.align(self.my, max_cost=90000)
        self.assertIsInstance(r, OrderedDPMixture, 'failed to return an ordered mixture')
        for i in r.lookup:
            self.assertEqual(i+1, r.lookup[i], 'assignment failed %d : %d' % (i, r.lookup[i]))

if __name__ == '__main__':
    suite1 = unittest.makeSuite(ClusterAlignTestCase,'test')

    unittest.main()