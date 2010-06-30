from Bio.Cluster import kcluster
from partition import Partition
from component import Component
from numpy import identity, zeros

# we should probably go thorugh and implement this so it works

class Centroid(Component):
    def __init__(self, pos):
        self.pos = pos

    
def kmeans(fcm, nclusters):
    data = fcm.view()[:]
    z, error, found = kcluster(data, nclusters=nclusters)
    
    centroids = []
    p = zeros((data.shape[0], max(z)+1))
    #print p.shape
    rfn = identity(max(z)+1)
    for i in range(max(z)+1):
        tmp = data[z==i].mean(0)
        centroids.append(Centroid(tmp))
        p[z==i] = rfn[i]
    
    return Partition(fcm, centroids, p, z)
        
    
    
if __name__ == '__main__':
    from fcm import loadFCS
    foo = loadFCS('../../sample_data/3FITC_4PE_004.fcs')
    
    cluster = kmeans(foo, 10)
    
    print cluster.z
    print cluster.get_k(0, .5)
    
    
