"""
Unit test framework
"""

import unittest
from numpy import array
from core import FCMdata
from core import Annotation
from random import randint


class FCMdataTestCase(unittest.TestCase):
    def setUp(self):
        self.pnts = array([[0,1,2],[3,4,5]])
        self.fcm = FCMdata(self.pnts, ['fsc','ssc','cd3'], [0,1])
        
    def testGetPnts(self):
        a = randint(0,1)
        b = randint(0,2)
        assert self.fcm.pnts[a,b] == self.pnts[a,b], "Data not consistent with inital data"
            
    def testGetChannelByName(self):
        assert self.fcm.get_channel_by_name(['fsc'])[0] == 0, 'incorrect first column'
        assert self.fcm.get_channel_by_name(['fsc'])[1] == 3, 'incorrect first column'
        
    def testGetMarkers(self):
        assert self.fcm.markers == [2], 'Marker CD3 not picked up'
    
    def testGetItem(self):
        a = randint(0,1)
        b = randint(0,2)
        assert type(self.fcm[a]) == type(self.pnts[a]), "__getitem__ failed to return array"
        assert self.fcm[a,b] == self.pnts[a,b], '__getitem__ returned wrong value'
        assert self.fcm['fsc','ssc'][a,0] == self.pnts[a,0], '__getitem__ with multiple strings failed'
                    
                    
class FCMannotationTestCase(unittest.TestCase):
    def setUp(self):
        self.test = {'foo': 'bar'}
        self.ann = Annotation(self.test)
    
    def testFlatName(self):
        assert self.ann.foo == 'bar', 'flat name lookup failed'
        assert self.ann['foo'] == 'bar', 'index lookup failed'
        assert self.ann.foo == self.ann['foo'], 'flat lookup isnt index lookup'
    
    def testFlatAssign(self):
        self.ann.spam = 'eggs'
        assert self.ann['spam'] == 'eggs', 'assignment lookup by index failed'
        assert self.ann.spam == 'eggs', 'assignment lookup by flat failed'
        
    def testIndexAssign(self):
        self.ann['spam'] = 'eggs'
        assert self.ann['spam'] == 'eggs', 'assignment lookup by index failed'
        assert self.ann.spam == 'eggs', 'assignment lookup by flat failed'
    
    def testAnnDeleg(self):
        assert self.ann.keys()[0] == self.test.keys()[0], 'delegation of keys() failed'

if __name__ == "__main__":
    suite1 = unittest.makeSuite(FCMdataTestCase,'test')
    suite2 = unittest.makeSuite(FCMannotationTestCase,'test')
    alltests = unittest.TestSuite((suite1, suite2))

    unittest.main()

        