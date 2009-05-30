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
    
    def __init__(self, x, pi, mu, sigma):
        """
        DiME(pi, mu, sigma):
        x: data points
        pi: mixing proportions
        mu: means of distributions
        sigma: covariances
        
        """
        
        self.pi = pi
        self.mu = mu
        self.sigma = sigma
        self.data = x
        self.k = x.shape[1] # dimension
        self.n = x.shape[0] # number of points
        self.c = len(pi) # number of clusters
        
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
        for pi, mu, sig in zip(self.pi, mus, sigmas):
            f.append(mvnormpdf(x.T , mu, sig))

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
    
if __name__ == '__main__':
    import numpy
    from numpy import random
    n = 1000

    pi1 = 0.2
    pi2 = 0.6
    pi3 = 0.2

    mu1_ = numpy.array([0,0,0])
    mu2_ = numpy.array([0,5,0])
    mu3_ = numpy.array([5,0,0])

    sigma1_ = numpy.identity(3)
    sigma2_ = numpy.identity(3)
    sigma3_ = numpy.identity(3)
    n1 = int(pi1*n)
    n2 = int(pi2*n)
    n3 = int(pi3*n)

    x1 = random.multivariate_normal(mu1_, sigma1_, n1)
    x2 = random.multivariate_normal(mu2_, sigma2_, n2)
    x3 = random.multivariate_normal(mu3_, sigma3_, n3)
    y = numpy.concatenate([x1, x2, x3])
    
    pi = numpy.array([pi1, pi2, pi3])
    mu = numpy.array([mu1_, mu2_, mu3_])
    sigma = numpy.array([sigma1_, sigma2_, sigma3_])
    
    foo = DiME(y, pi, mu, sigma)
    print foo.run([])
    print foo.run(0)
    print foo.run(1)
    print foo.run(2)
        