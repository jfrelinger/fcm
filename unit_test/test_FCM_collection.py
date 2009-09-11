import unittest
from numpy import array, all, equal, sum
from random import randint

from fcm import FCMdata
from fcm import FCMcollection

class FCMcollectionTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def testCheckNames(self):
        pnts = array([])
        fcm1 = FCMdata('test_fcm1', pnts, ['fsc','ssc','cd3'], [0,1])
        fcm2 = FCMdata('test_fcm2', pnts, ['fsc','ssc','cd3'], [0,1])
        fcm3 = FCMdata('test_fcm3', pnts, ['fsc','ssc','cd4'], [0,1])

        fcms1 = FCMcollection('fcms1', [fcm1, fcm2])
        fcms2 = FCMcollection('fcms2', [fcm1, fcm2, fcm3, fcms1])
        
        check1 = fcms1.check_names()
        assert check1[fcms1.name] == [True, True, True]
        check2 = fcms2.check_names()
        assert check2[fcms2.name] == [check1, True, True, False]
        
if __name__ == '__main__':
    suite1 = unittest.makeSuite(FCMcollectionTestCase,'test')

    unittest.main()
