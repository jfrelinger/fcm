"""
perform dime analysis on a set of data points and a set of pi, mu, and sigmas
for the clusters

"""

from numpy import zeros, sum, log
from distributions import mvnormpdf, mixnormpdf

class DiME(object):
    """
    DiME analysis object
    
    """
    
    def __init__(self, x, pi, mu, sigma, cmap=None):
        """
        DiME(pi, mu, sigma, cmap=None):
        x: data points
        pi: mixing proportions
        mu: means of distributions
        sigma: covariances
        cmap: a dictionary of modal cluster to component clusters, defaults to
            None. If none perform analysis on component cluster level.
        """
        
        self.pi = pi
        self.mu = mu
        self.sigma = sigma
        self.data = x
        self.k = x.shape[1] # dimension
        self.n = x.shape[0] # number of points
        if cmap == None:
            self.c = len(pi) # number of clusters
        else:
            self.c = len(cmap.keys())
        self.cmap = cmap
        
        self.dc0 = self.run([])
        
    def run(self, drop):
        """
        Perform Dime calculation and return a list of information in each
        channel
        
        run(drop): channel or list of channels to drop (indexed by 0)
        
        """

        ids = []
        if type(drop) is type(1): # are we dropping single col?
            for i in range(self.k):
                if i != drop:
                    ids.append(i)
        else: # we're dropping a list
            for i in range(self.k):
                if i not in drop:
                    ids.append(i)
        
        x = self.data[:,ids]
        ids = numpy.array(ids)
        mus = [m[ids] for m in self.mu]
        sigmas = [sig[ids,:][:,ids] for sig in self.sigma]
        
        f = []
        if self.cmap == None:
            for pi, mu, sig in zip(self.pi, mus, sigmas):
                f.append(mvnormpdf(x.T , mu, sig))
        else:
            for mclust in self.cmap.keys():
                clsts = self.cmap[mclust]
                mu = [ mus[i] for i in clsts]
                sigma = [ sigmas[i] for i in clsts]
                pi = [ self.pi[i] for i in clsts]
                f.append(mixnormpdf(x.T, pi, mu, sigma))

        F = {}
        for i in range(self.c):
            for j in range(i,self.c):
                F[(i,j)] = F[(j,i)] =  sum(f[i]*f[j])/self.n

        d = []
        for i in range(self.c):
            tmpA = 0
            tmpa = 0

            for j in range(self.c):
                tmpA+= self.pi[j]*F[(j,i)]
                if i != j:
                    tmpa+= self.pi[j]*F[(j,i)]
                
            A = self.pi[i]*tmpA
            a = self.pi[i]/(1-self.pi[i])*tmpa
            
            d.append( -1*log(a/A))    

        return d
    
    def dc(self, drop):
        tmp = self.run(drop)
        return [ 100*tmp[i]/self.dc0[i] for i in range(len(self.dc0))]
                                       