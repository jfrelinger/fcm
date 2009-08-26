from __future__ import division
from distributions import mvnormpdf, mixnormpdf
from numpy import array, dot, log2, zeros, sum, diag, ones, identity

class DiME(object):
    """
    DiME analysis object
    
    """
    
    def __init__(self, pi, mu, sigma, cmap=None):
        """
        DiME(pi, mu, sigma, cmap=None):
        pi: mixing proportions
        mu: means of distributions
        sigma: covariances
        cmap: a dictionary of modal cluster to component clusters, defaults to
            None. If none perform analysis on component cluster level.
        """
        
        self.pi = pi
        self.mu = mu
        self.sigma = sigma
        self.k = self.mu.shape[1]
        if cmap == None:
            self.c = len(pi) # number of clusters
            self.cpi = pi
        else:
            self.c = len(cmap.keys())
            self.cpi = []
            for clst in cmap.keys():
                self.cpi.append( sum([pi[j] for j in cmap[clst]]))
        self.cmap = cmap
        
        
    def d(self, drop = None):
        """
        calculate discriminitory information
        """
        if drop is None:
            drop = []
        ids = []
        if type(drop) is type(1): # are we dropping single col?
            for i in range(self.k):
                if i != drop:
                    ids.append(i)
        else: # we're dropping a list
            for i in range(self.k):
                if i not in drop:
                    ids.append(i)
        
        ids = array(ids)
        mus = [m[ids] for m in self.mu]
        sigmas = [sig[ids,:][:,ids] for sig in self.sigma]
        
        # calculate -1*\log_2(\frac{\delta_c}{\Delta_c})
        # where \detla_c =  \frac{\gamma_c}\{1-\gamma_c}\Sum_{e != c}\gamma_e F_{c,e}
        # \Delta_c = \gamma_c \Sum_{e=1:C} \gamma_e F_{c,e}
        # where F_{c,e} = \int f_c(x)*f_e(x) dx
        # and  where f_c(x) = \Sum_{J in c} \frac{\pi_j}{\gamma_c}N(x|\mu_j,\Sigma_j)
        
        # we calculate F_{c,e} as P(mu_c-mu_e ~ N(0, sigma_c+sigma_e))
        # since we're going to be calculating with P(x in j) a lot precalculate it all
        # once in advance, since we'll need it all at least once and in general multiple times
        # TODO: parallelize here
        
        size = len(self.pi)
        f = zeros((size, size))
        for i in range(size):
            for j in range(i,size):
                f[j, i] = mvnormpdf(mus[i], mus[j], sigmas[i]+sigmas[j])
                f[i,j] = f[j,i]
                
        F = zeros((self.c, self.c))
        for i in range(self.c):
            for j in range(i, self.c):
                tmp = 0
                for fclust in self.cmap[i]:
                    for tclust in self.cmap[j]:
                        tmp += (self.pi[fclust]/self.cpi[i])*(self.pi[tclust]/self.cpi[j])*f[fclust,tclust]
                F[j,i] = tmp
                F[i,j] = F[j,i]
        

        #calculate \delta_c and \Delta_c
        dc = zeros(self.c)
        Dc = zeros(self.c)
        sum_ex = 0 # use this later to caculate complete sum in \Gamma_c
        for mclust in self.cmap.keys():
            normalizing = 1.0/(1-self.cpi[mclust])
            tmp = []
            for i in self.cmap.keys():
                if i == mclust:
                    pass
                else:
                    tmp.append(i)
            sum_ex = sum([self.cpi[i]*F[mclust,i] for i in tmp])
            dc[mclust] = normalizing*sum_ex
            Dc[mclust] = 1*(sum_ex+(self.cpi[mclust]*F[mclust,mclust]))

        return -1*log2(dc/Dc)
    
    def rdrop(self, drop):
        try:
            return 100*self.d(drop)/self.d_none
        except AttributeError:
            self.d_none = self.d()
            return 100*self.d(drop)/self.d_none
