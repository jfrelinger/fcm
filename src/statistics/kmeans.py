
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
    cent = Array()
    def __init__(self, centroids):
        self.cent = centroids
        
    def classify(self, x):
        return vq.vq(x, self.centroids)[0]
    
    def mus(self):
        return self.cent
    
    def centroids(self):
        return self.mus()
    
