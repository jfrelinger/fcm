'''
Created on May 16, 2013
Author: Jacob Frelinger <jacob.frelinger@duke.edu>
'''

import numpy as np
from scipy.misc import logsumexp
from scipy.optimize import fmin
from kldiv import eSKLdiv
from fcm.statistics import DPMixtureModel

class DiagonalAlignData(object):
    def __init__(self, x, m=None, size=25, k=100, verbose=100,
                          maxiter=10000):

        if len(x.shape) == 1:
            x = x.reshape(x.shape[0], 1)
        self.x = x.copy()
        self.d = self.x.shape[1]

        self.k = k
        self.verbose = verbose
        self.maxiter = maxiter
        self.size = size

        if m is None:
            self.m = DPMixtureModel(self.k, 1, 1000)
        else:
            self.m = m
        self.m.ident = True
        
        self.mx = None
        self.my = None
    def align(self, y, x0=None):
        # if we don't know mx, fit x
        if self.mx is None:
            self.__fit_xy(self.x, 'mx')

        # if we don't know my fit y
        if self.my is None:
            self.__fit_xy(y, 'my')

        #set up grid for aprox skldiv
        mins = np.array([min(self.x[:, i].min(), y[:, i].min()) for i in range(self.d)])
        maxs = np.array([max(self.x[:, i].max(), y[:, i].max()) for i in range(self.d)])

        idxs = tuple([slice(mins[i], maxs[i], self.size * 1j) for i in range(self.d)])
        self.pnts = np.mgrid[idxs].T.flatten().reshape((self.size ** self.d, self.d))

        # precalculate lp for kldiv, it doesn't change between iterations
        self.lp = logsumexp(self.mx.prob(self.pnts, logged=True, use_gpu=True), 1)

        # estimate x0 if we don't know it
        if x0 is None:
            shift = -1 * y.mean(0) * self.x.std(0) / y.std(0) + self.x.mean(0)
            scale = self.x.std(0) / y.std(0)
            x0 = np.hstack((scale.flatten(), shift))

        #call minimizer on 
        z = fmin(self.__opteSKLdiv, x0, maxfun=10 ** 6)
        b = z[-self.d:]
        a = np.eye(self.d)
        for i in range(self.d):
            a[i,i] = z[i]
        
        return a, b
    def __opteSKLdiv(self, n):
            a = np.eye(self.d)
            z = 0
            for i in range(self.d):
                a[i, i] = n[i]
            b = n[-self.d:]
            return eSKLdiv(self.mx, (self.my * a), self.d, self.pnts, lp=self.lp, a=a, b=b, orig_y=self.my)



    def __fit_xy(self, x, key):

        r = self.m.fit(x, self.verbose)
        r = r.average()

        self.__setattr__(key, r)






def comp_align_data():
    pass

def full_align_data():
    pass
