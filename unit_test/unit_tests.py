"""
Unit test framework
"""

import unittest
from test_Annotation import FCMAnnotationTestCase
from test_FCM_data import FCMdataTestCase
from test_load_lmd import FCSreaderLMDTestCase
from test_transforms import FCMtransformTestCase
from test_load_fcs import FCSreaderTestCase
from test_tree  import TreeTestCase
from test_subsample import SubsampleTestCase
from test_dpmixture import Dp_mixtureTestCase
from test_dpcluster import Dp_clusterTestCase
from test_mdpmixture import ModalDp_clusterTestCase
from test_mvnpdf import mvnpdfTestCase
from test_cluster import DPMixtureModel_TestCase
from test_hdp import HDPMixtureModel_TestCase
from test_hdpmixture import HDPMixtureTestCase
from test_mhdpmixture import ModalHDp_clusterTestCase
from test_data_align import DiagAlignTestCase
from test_ordereddpmixture import OrderedDp_mixtureTestCase
from test_cluster_align import ClusterAlignTestCase

if __name__ == "__main__":
    suite1 = unittest.makeSuite(FCMdataTestCase,'test')
    suite2 = unittest.makeSuite(FCMAnnotationTestCase,'test')
    suite3 = unittest.makeSuite(FCSreaderLMDTestCase, 'test')
    suite4 = unittest.makeSuite(FCMtransformTestCase, 'test')
    suite5 = unittest.makeSuite(FCSreaderTestCase,'test')
    suite6 = unittest.makeSuite(TreeTestCase, 'test')
    suite7 = unittest.makeSuite(SubsampleTestCase, 'test')
    suite8 = unittest.makeSuite(Dp_clusterTestCase, 'test')
    suite9 = unittest.makeSuite(ModalDp_clusterTestCase, 'test')
    suite10 = unittest.makeSuite(mvnpdfTestCase, 'test')
    suite11 = unittest.makeSuite(DPMixtureModel_TestCase, 'test')
    suite12 = unittest.makeSuite(HDPMixtureModel_TestCase, 'test')
    suite13 = unittest.makeSuite(Dp_mixtureTestCase, 'test')
    suite14 = unittest.makeSuite(HDPMixtureTestCase, 'test')
    suite15 = unittest.makeSuite(ModalHDp_clusterTestCase, 'test')
    suite16 = unittest.makeSuite(DiagAlignTestCase, 'test')
    suite17 = unittest.makeSuite(OrderedDp_mixtureTestCase, 'test')
    suite18 = unittest.makeSuite(ClusterAlignTestCase, 'test')
    alltests = unittest.TestSuite((suite1, suite2, suite3, suite4, suite5,
                                    suite6, suite7, suite8,suite10, suite11,
                                    suite12, suite13, suite14, suite15,
                                    suite16, suite17, suite18))

    unittest.main()

        