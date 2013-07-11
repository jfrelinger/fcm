'''
Created on May 6, 2013

@author: Jacob Frelinger <jacob.frelinger@duke.edu>
'''
import numpy as np
from scipy.spatial.distance import cdist
import fcm.statistics as stats
from fcm.alignment.munkres import _get_cost
from fcm.alignment.kldiv import eKLdiv as kldiv
from fcm.statistics.dp_cluster import ModalDPMixture


def mean_distance(ref, test, use_means=None):
    '''
    calculate cost matrix using mean distance between mixture models
    optional argument use_means controls overiding default use of modes if
    available
    '''
    if not use_means:
        try:
            x = ref.modes
            y = test.modes
        except AttributeError:
            x = ref.mus
            y = test.mus
    else:
        x = ref.mus
        y = test.mus

    return cdist(x, y)


def classification_distance(ref, test, test_data=None, ndraw=100000):
    '''
    generate cost matrix using miss classification as distance
    '''
    if test_data is None:
        test_data = ref.draw(ndraw)

    t_x = test.classify(test_data)
    r_x = ref.classify(test_data)

    cost = test_data.shape[0] * np.ones((len(test.clusters), len(ref.clusters)), dtype=np.int)

    _get_cost(t_x, r_x, cost)

    #return (cost / test_data.shape[0]).T.copy()
    return cost.T.copy().astype(np.double)/test_data.shape[0]


def kldiv_distance(ref, test, use_means=None, ndraws=100000):
    '''
    generate cost matrix using kl-divergence
    '''
    if isinstance(ref, ModalDPMixture) and isinstance(test, ModalDPMixture):
        xs = [ref.get_submodel(j) for j in ref.cmap]
        ys = [test.get_submodel(j) for j in test.cmap]
    else:
        xs = [ref.get_submodel(j) for j in range(len(ref))]
        ys = [test.get_submodel(j) for j in range(len(test))]
        
    cost = np.zeros((len(xs),len(ys)))
    for i,j in enumerate(xs):
        for k,l in enumerate(ys):
            cost[i,k] = kldiv(j,l)
            
    return cost

if __name__ == '__main__':
    cluster1 = stats.DPCluster(.005, np.array([0, 0]), np.eye(2))
    cluster2 = stats.DPCluster(.995, np.array([0, 4]), np.eye(2))
    cluster3 = stats.DPCluster(.25, np.array([0, 0]), np.eye(2))
    cluster4 = stats.DPCluster(.25, np.array([4, 0]), np.eye(2))
    cluster5 = stats.DPCluster(.5, np.array([0, 4]), np.eye(2))
    A = stats.DPMixture([cluster1, cluster2])
    B = stats.DPMixture([cluster3, cluster4, cluster5])
    from munkres import munkres
    print 'Ref has means', A.mus, 'with weights', A.pis
    print 'Test has means', B.mus, 'with weights', B.pis
#    print 'mean distance'
#    print mean_distance(A, B)
#    print munkres(mean_distance(A, B))
    mA = A.make_modal()
    mB = B.make_modal()
#    print 'modal distance'
#    print mean_distance(mA, mB)
#    print munkres(mean_distance(mA, mB))
#    print 'modal using means'
#    print mean_distance(mA, mB, use_means=True)
#    print munkres(mean_distance(mA, mB, use_means=True))

    print 'classification'
    print classification_distance(A, B)
    print classification_distance(A, B).mean(), classification_distance(A, B).sum()
    print munkres(classification_distance(A, B))
    print 'modal classification'
    print classification_distance(mA, mB)
    print classification_distance(mA, mB).mean()
    print munkres(classification_distance(mA, mB))

    print 'kldiv'
    print kldiv_distance(A, B)
    print munkres(kldiv_distance(A, B))

    print 'modal kldiv'
    print kldiv_distance(mA, mB)
    print munkres(kldiv_distance(mA, mB))
