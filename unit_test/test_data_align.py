import unittest
import fcm
from fcm.alignment import DiagonalAlignDataS, DiagonalAlignData
import numpy as np
import numpy.testing as npt

class DiagAlignTestCase(unittest.TestCase):
    def setUp(self):
        self.x = fcm.FCSreader('../sample_data/3FITC_4PE_004.fcs').get_FCMdata()[:5000,:]
        self.y = fcm.FCSreader('../sample_data/3FITC_4PE_004.fcs').get_FCMdata()[:5000,:]+10
        
        
        self.alignS = DiagonalAlignDataS(self.x, k=8, size=100)
        self.align = DiagonalAlignData(self.x, k=8, size=100)
        self.align.m.parallel=True
        #self.align.m.niter=100
        #self.align.m.burnin=100
        
    def testDiagAlign(self):
        a,b = self.align.align(self.y)
        npt.assert_array_almost_equal(a, np.eye(4), decimal=2)
        npt.assert_array_almost_equal(b, -10*np.ones(4), decimal=2)
    
    def testDiagAlignS(self):
        a,b = self.alignS.align(self.y)
        npt.assert_array_almost_equal(a, np.eye(4), decimal=2)
        npt.assert_array_almost_equal(b, -10*np.ones(4), decimal=2)

if __name__ == '__main__':
    suite1 = unittest.makeSuite(DiagAlignTestCase,'test')

    unittest.main()