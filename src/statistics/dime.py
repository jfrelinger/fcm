from statistics.distributions import mvnormpdf, mixnormpdf
from numpy import array, dot, log2, zeros, sum
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
            self.cpi = pi
        else:
            self.c = len(cmap.keys())
            self.cpi = []
            for clst in cmap.keys():
                self.cpi.append( sum([pi[j] for j in cmap[clst]]))
        self.cmap = cmap
        
    def deltac(self, drop):
        gc = zeros(self.c)
        for mclust in self.cmap.keys():
            normalizing = self.cpi[mclust]/(1-self.cpi[mclust])
            tmp = []
            for i in self.cmap.keys():
                if i == mclust:
                    pass
                else:
                    tmp.append(i)
            fc = self.p(mclust)
            fe = [self.p(j) for j in tmp]
            gc[mclust] = normalizing*sum([self.cpi[j]*sum(fc*i) for j,i in enumerate(fe)])/self.n
        return gc
    def Deltac(self, drop):
        Gc = zeros(self.c)
        for mclust in self.cmap.keys():

            fc = self.p(mclust)
            fe = [self.p(j) for j in self.cmap.keys()]
            
            Gc[mclust] = self.cpi[mclust]*sum([self.cpi[j]*sum(fc*i) for j,i in enumerate(fe)])/self.n
        return Gc
    
    def p(self, mclust, x = None, mus= None, sigmas= None):
        """
        calculate probabilites of x in modal cluster mclust
        """
        if x is None:
            x = self.data
        if mus is None:
            mus = self.mu
        if sigmas is None:
            sigmas = self.sigma
        
        clsts = self.cmap[mclust]
        mu = [ mus[i] for i in clsts]
        sigma = [ sigmas[i] for i in clsts]
        pi = [ self.pi[i]/self.cpi[mclust] for i in clsts]
        return mixnormpdf(x.T,pi, mu, sigma)
     
    def d(self, drop = []):
        """
        calculate discriminitory information
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
        ids = array(ids)
        mus = [m[ids] for m in self.mu]
        sigmas = [sig[ids,:][:,ids] for sig in self.sigma]
        
        # calculate -1*\log_2(\frac{\delta_c}{\Delta_c})
        # where \detla_c =  \frac{\gamma_c}\{1-\gamma_c}\Sum_{e != c}\gamma_e F_{c,e}
        # \Delta_c = \gamma_c \Sum_{e=1:C} \gamma_e F_{c,e}
        # where F_{c,e} = \int f_c(x)*f_e(x) dx
        # and  where f_c(x) = \Sum_{J in c} \frac{\pi_j}{\gamma_c}N(x|\mu_j,\Sigma_j)
        # we're aproximiating F_{c,e} by \frac{ \Sum_{x} P(x in e)*P(x in c)}{n}
        
        # since we're going to be calculating with P(x in j) a lot precalculate it all
        # once in advance, since we'll need it all at least once and in general multiple times
        # TODO: parallelize here
        f = zeros((self.c, self.n))
        for mclust in self.cmap.keys():
            f[mclust, :] = self.p(mclust, x, mus, sigmas)
        
        # precaculate F_{c,e}
        # note F_{c,e} = F_{e,c}
        # TODO: parallelize here
        F = zeros((self.c, self.c), dtype='float64')
        for i in range(self.c):
            for j in range(i, self.c):
                F[i,j] = F[j,i] = sum(f[i]*f[j])/self.n
        

        #calculate \delta_c and \Delta_c
        dc = zeros(self.c)
        Dc = zeros(self.c)
        sum_ex = 0 # use this later to caculate complete sum in \Gamma_c
        for mclust in self.cmap.keys():
            normalizing = self.cpi[mclust]/(1-self.cpi[mclust])
            tmp = []
            for i in self.cmap.keys():
                if i == mclust:
                    pass
                else:
                    tmp.append(i)
            sum_ex = sum([self.cpi[i]*F[mclust,i] for i in tmp])
            dc[mclust] = normalizing*sum_ex
            Dc[mclust] = self.cpi[mclust]*(sum_ex+(self.cpi[mclust]*F[mclust,mclust]))
        return -1*log2(dc/Dc)
    
    def rdrop(self, drop):
        try:
            return 100*self.d(drop)/self.d_none
        except AttributeError:
            self.d_none = self.d()
            return 100*self.d(drop)/self.d_none