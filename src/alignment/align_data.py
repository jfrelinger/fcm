'''
Created on May 16, 2013
Author: Jacob Frelinger <jacob.frelinger@duke.edu>
'''

import numpy as np
from scipy.misc import logsumexp
from scipy.optimize import fmin
from kldiv import eSKLdiv, eKLdiv
from fcm.statistics import DPMixtureModel

class BaseAlignData(object):
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
        '''
        Generate A and B that minimizes the distance between x and y
        '''
        # if we don't know mx, fit x
        if self.mx is None:
            self._fit_xy(self.x, 'mx')

        # if we don't know my fit y
        if self.my is None:
            self._fit_xy(y, 'my')

        #set up grid for aprox skldiv
        mins = np.array([min(self.x[:, i].min(), y[:, i].min()) for i in range(self.d)])
        maxs = np.array([max(self.x[:, i].max(), y[:, i].max()) for i in range(self.d)])

        idxs = tuple([slice(mins[i], maxs[i], self.size * 1j) for i in range(self.d)])
        self.pnts = np.mgrid[idxs].T.flatten().reshape((self.size ** self.d, self.d))

        # precalculate lp for kldiv, it doesn't change between iterations
        self.lp = logsumexp(self.mx.prob(self.pnts, logged=True, use_gpu=True), 1)

        # estimate x0 if we don't know it
        if x0 is None:
            x0 = self._get_x0(y)

        #call minimizer on 
        z = fmin(self._optimize, x0, maxiter=self.maxiter)
        a, b = self._format_z(z)
        
        #no need to keep my now
        self.__setattr__('my', None)
        return a, b

    def _fit_xy(self, x, key):

        r = self.m.fit(x, self.verbose)
        r = r.average()

        self.__setattr__(key, r)

    def _get_x0(self, y):
        raise NotImplementedError

    def _format_z(self, z):
        raise NotImplementedError

    def _optimize(self, n):
        raise NotImplementedError

class DiagonalAlignData(BaseAlignData):
    '''
    Generate Diagonal only alignment
    '''
    def _get_x0(self, y):
        shift = -1 * y.mean(0) * self.x.std(0) / y.std(0) + self.x.mean(0)
        scale = self.x.std(0) / y.std(0)
        return np.hstack((scale.flatten(), shift))

    def _optimize(self, n):
            a = np.eye(self.d)
            z = 0
            for i in range(self.d):
                a[i, i] = n[i]
            b = n[-self.d:]
            return eKLdiv(self.mx, (self.my * a), self.d, self.pnts, lp=self.lp, a=a, b=b, orig_y=self.my)

    def _format_z(self, z):
        b = z[-self.d:]
        a = np.eye(self.d)
        for i in range(self.d):
            a[i, i] = z[i]

        return a, b

class DiagonalAlignDataS(DiagonalAlignData):
    def _optimize(self, n):
            a = np.eye(self.d)
            z = 0
            for i in range(self.d):
                a[i, i] = n[i]
            b = n[-self.d:]
            return esKLdiv(self.mx, (self.my * a), self.d, self.pnts, lp=self.lp, a=a, b=b, orig_y=self.my)


def comp_align_data():
    def _get_x0(self, y):
        a = np.eye(self.d)
        b = np.zeros(self.d)
        return np.hstack(a.flatten(), b)
    

    def _format_z(self, z):
        raise NotImplementedError

    def _optimize(self, n):
        raise NotImplementedError


def full_align_data():
    pass #will need more work
