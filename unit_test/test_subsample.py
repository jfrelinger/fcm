'''
Created on Aug 27, 2009

@author: jolly
'''
import unittest
from fcm import FCMdata
from fcm import Gate
from fcm.core import SubsampleFactory
from numpy import array

class SubsampleTestCase(unittest.TestCase):
    def setUp(self):
        self.pnts = array([[0,1,2],[3,4,5]])
        self.fcm = FCMdata('test_fcm', self.pnts, ['fsc','ssc','cd3'], [0,1])
        self.samp = SubsampleFactory[:,1]

    def testSubSample(self):
        self.samp.subsample(self.fcm)
        assert self.fcm[0] == 1, 'subsample via subsample failed'
        assert self.fcm[1] == 4, 'subsample via subsample failed'
        self.fcm.visit('root')
        self.fcm.subsample(self.samp)
        assert self.fcm[0] == 1, 'subsample via fcm failed'
        assert self.fcm[1] == 4, 'subsample via fcm failed'
        


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testSubSample']
    unittest.main()