import unittest
from fcm import FCSreader

class FCSreaderTestCase(unittest.TestCase):
    def setUp(self):
        self.fcm = FCSreader('../sample_data/3FITC_4PE_004.fcs').get_FCMdata()
        
    def testGetPnts(self):
        self.assertEqual(self.fcm.shape[0], int(self.fcm.notes.text['tot']))

    def testGetNotes(self):
        self.assertEqual(self.fcm.notes.text['cyt'], 'FACScan')

if __name__ == '__main__':
    suite1 = unittest.makeSuite(FCSreaderTestCase,'test')

    unittest.main()
