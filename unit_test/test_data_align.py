import unittest
import fcm
from fcm.alignment import DiagonalAlignData
import numpy as np
import numpy.testing as npt

class DiagAlignTestCase(unittest.TestCase):
    def setUp(self):
        self.x = fcm.FCSreader('../sample_data/3FITC_4PE_004.fcs').get_FCMdata()[:10000,:]
        self.y = fcm.FCSreader('../sample_data/3FITC_4PE_004.fcs').get_FCMdata()[:10000,:]
        
        
        self.align = DiagonalAlignData(self.x, k=8)
        self.align.m.parallel=True
        
    def testDiagAlign(self):
        a,b = self.align.align(self.y)
        npt.assert_array_almost_equal(a, np.eye(4), decimal=2)
        npt.assert_array_almost_equal(b, np.zeros(4), decimal=2)
if __name__ == '__main__':
    suite1 = unittest.makeSuite(DiagAlignTestCase,'test')

    unittest.main()