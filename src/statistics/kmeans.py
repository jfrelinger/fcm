
from enthought.traits.api import HasTraits, List, Array
from scipy.cluster import vq

#class Centroid(HasTraits):
#    '''Kmeans centroid'''
#    cent = Array()
#    def __init__(self, cent):
#        self.cent = cent
        
class KMeans(HasTraits):
    '''
    K means model
    '''
    centroids = Array()
    def __init__(self, centroids):
        self.centroids = centroids
        
    def classify(self, x):
        return vq.vq(x, self.centroids)[0]
    
    
