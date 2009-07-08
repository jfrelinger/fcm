import unittest
import numpy
from core import productlog

class FCMtransformTestCase(unittest.TestCase):
    def setUp(self):
        pass
        
    def testProductLog(self):
        x = numpy.array([0,1,10,100,1000,10000], 'd')
        ans = numpy.array([0., 0.567143, 1.74553, 3.38563, 5.2496, 7.23185])
        self.assert_(numpy.all(numpy.abs(productlog(x) - ans) < 0.1))

if __name__ == '__main__':
    suite1 = unittest.makeSuite(FCMtransformTestCase, 'test')

    unittest.main()
