from __future__ import division
from distributions import mvnormpdf, mixnormpdf
from numpy import array, dot, log2, zeros, sum, diag, ones, identity

class Dime(object):
    """
    DiME analysis object
    
    """
    
    def __init__(self,cluster = None, pi = None, mu = None, sigma = None, cmap = None):
        """
        DiME(pi, mu, sigma, cmap=None):
        pi: mixing proportions
        mu: means of distributions
        sigma: covariances
        cmap: a dictionary of modal cluster to component clusters, defaults to
            None. If none perform analysis on component cluster level.
        """
        self.pi = None
        self.mu = None
        self.sigma = None
        
        #pull mixture parameters from cluster if passed
        if cluster is  not None:
            self.pi = cluster.get_pis()
            self.mu = cluster.get_mus()
            self.sigma = cluster.get_sigmas()
            # TODO add fetching cmap here
            try:
                if cmap == None:
                    cmap = cluster.cmap
            except AttributeError:
                pass
        
        #if mixture parameters are passed explicitly use them.
        if pi is not None:
            self.pi = pi
        if mu is not None:
            self.mu = mu
        if sigma is not None:
            self.sigma = sigma
            
        #check we got all pramters
        if (self.pi is None or self.mu is None or self.sigma is None):
            raise TypeError('dime requires a cluster object or pi, mu, and sigma to be explicitly given')
        
        if type(self.pi) == type([]): #not sure if htis is needed but the code expects pi to be a list
            self.pi = array(self.pi)
            
        self.k, self.p= self.mu.shape
        if cmap == None:
            self.c = len(pi) # number of clusters
            self.cpi = pi
            cmap = {}
            for i in range(self.c):
                cmap[i] = [i]
        else:
            self.c = len(cmap.keys())
            self.cpi = []
            for clst in cmap.keys():
                self.cpi.append( sum([pi[j] for j in cmap[clst]]))
        self.cmap = cmap
        
        
    def drop(self, target, drop = None):
        """
        calculate discriminitory information
        """
        if drop is None:
            drop = []
        dim = {}
        if type(drop) is type(1): # are we calculating single col?
            dim[0] = [ drop ]
        else: # drop is a list...
            if set(map(type,drop)) == set([type(1)]): # we've got a list of numbers
                dim[0] = drop
            else: #mostlikly a mixed list of list
                for c,i in enumerate(drop):
                    tmp = []
                    for j in range(self.p):
                        if j in i:
                            tmp.append(j)
                    dim[c] = tmp[:]

        nn = max(dim.keys())+1
        Dj = zeros((nn,))
        
        gpj = self.cmap[target]
        cpj = 1-sum(self.pi[gpj]);
        
        indexj = []
        for i in range(self.k):
            if i not in gpj:
                indexj.append(i)
        
        D = zeros((nn,self.k,self.k))
        
        for tt in range(nn):
            deno = 0
            nume = 0
            dimm = dim[tt]
            for i in range(self.k):
                mi = self.mu[i,:][:,dimm]
                si = self.sigma[i,:,:][dimm,:][:,dimm]
                for jj in range(i,self.k):
                    #print self.sigma[jj,:,:][dimm,:][:,dimm]
                    D[tt,jj,i] = mvnormpdf(self.mu[jj,:][:,dimm],mi,si+self.sigma[jj,:,:][dimm,:][:,dimm])
                    D[tt,i,jj] = D[tt,jj,i]
            #print 'd',D[tt,k-1,k-1]       
            for ii in gpj:
                for ppp in indexj:
                    nume = nume + self.pi[ii]*(self.pi[ppp]/cpj)*D[tt,ppp,ii]
          
                for ppp in range(self.k):
                    deno = deno + self.pi[ii]*self.pi[ppp]*D[tt,ppp,ii]
                
            Dj[tt] = nume/deno
        
        return Dj
    

