import unittest
from io import FCSreader

class FCSreaderTestCase(unittest.TestCase):
    def setUp(self):
        self.fcm = FCSreader('/Volumes/HD2/data/flow/cvc_ics/Lab04/ICS2D1S1R1lab4.LMD').get_FCMdata()
        
    def testGetPnts(self):
        self.assertEqual(self.fcm.shape, (94569, 4))

    def testGetNotes(self):
        self.assertEqual(self.fcm.notes.text['cyt'], 'FACScan')

if __name__ == '__main__':
    suite1 = unittest.makeSuite(FCSreaderTestCase,'test')

    unittest.main()
