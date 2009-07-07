import unittest
import sys
sys.path.append('../src')
from readfcs import FCSreader

class FCSreaderTestCase(unittest.TestCase):
    def setUp(self):
        self.fcm = FCSreader('../sample_data/3FITC_4PE_004.fcs').get_FCMdata()
        
    def testGetPnts(self):
        self.assertEqual(self.fcm.pnts.shape, (94569, 4))

if __name__ == '__main__':
    suite1 = unittest.makeSuite(FCSreaderTestCase,'test')

    unittest.main()
