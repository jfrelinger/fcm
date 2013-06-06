'''
Created on May 16, 2013
Author: Jacob Frelinger <jacob.frelinger@duke.edu>
'''

import numpy as np
from scipy.misc import logsumexp
from scipy.optimize import fmin
from fcm.alignment.kldiv import eSKLdiv, eKLdiv, new_eKLdiv, new_eSKLdiv
from fcm.statistics import DPMixtureModel


class BaseAlignData(object):
    '''
    base class to align a data set to a reference data set
    '''
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
            self.m = DPMixtureModel(self.k, 100, 1000)
        else:
            self.m = m
        self.m.ident = True

        self.mx = None
        self.my = None

    def align(self, y, x0=None):
        '''
        Generate A and B that minimizes the distance between x and y
        '''
        #fit models
        self._fit_models(y)

        #build a grid of points to evaluate over
        self.pnts = self._buid_grid(y)

        # precalculate lp for kldiv, it doesn't change between iterations
        #self.lp = logsumexp(self.mx.prob(self.pnts, logged=True, use_gpu=True), 1)

        # estimate x0 if we don't know it
        if x0 is None:
            x0 = self._get_x0(y)
        
        #call minimizer on
        z = self._min(self._optimize, x0, maxiter=self.maxiter)
        a, b = self._format_z(z)

        #no need to keep my now
        self.__setattr__('my', None)
        return a, b

    def _min(self, func, x0, **kwargs):
        return fmin(func, x0, **kwargs)

    def _buid_grid(self, y):
        #set up grid for aprox kldiv
        mins = np.array([min(self.x[:, i].min(), y[:, i].min()) for i in range(self.d)])
        maxs = np.array([max(self.x[:, i].max(), y[:, i].max()) for i in range(self.d)])

        idxs = tuple([slice(mins[i], maxs[i], self.size * 1j) for i in range(self.d)])
        return np.mgrid[idxs].T.flatten().reshape((self.size ** self.d, self.d))

    def _fit_models(self, y):
        # if we don't know mx fit x
        if self.mx is None:
            self._fit_xy(self.x, 'mx')

        # if we don't know my fit y
        if self.my is None:
            self._fit_xy(y, 'my')

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
    def _buid_grid(self, y):
        #set up grid for aprox skldiv
        mins = np.array([min(self.x[:, i].min(), y[:, i].min()) for i in range(self.d)])
        maxs = np.array([max(self.x[:, i].max(), y[:, i].max()) for i in range(self.d)])
        pnts = np.empty((self.size, self.d))
        for i in range(self.d):
            pnts[:, i] = np.linspace(mins[i], maxs[i], self.size)

        return pnts

    def _get_x0(self, y):
        shift = -1 * y.mean(0) * self.x.std(0) / y.std(0) + self.x.mean(0)
        scale = self.x.std(0) / y.std(0)
        return np.hstack((scale.flatten(), shift))

    def _optimize(self, n, mx, my, pnts):
        a, b = n
        rslt = new_eKLdiv(mx, (my * a)+b, pnts)
        return rslt
    
    def _min(self, func, x0, **kwargs):
        z = np.zeros(self.d + self.d)
        for i in range(self.d):
            mx = self.mx.get_marginal(i)
            my = self.my.get_marginal(i)
            a, b = fmin(func, x0[[i, self.d + i]], (mx, my, self.pnts[:, i]))
            z[i] = a
            z[i + self.d] = b

        return z

    def _format_z(self, z):
        b = z[-self.d:]
        a = np.eye(self.d)
        for i in range(self.d):
            a[i, i] = z[i]

        return a, b


class DiagonalAlignDataS(DiagonalAlignData):
    '''
    Gnerate Diagonal alignment using symetric KLDivergence
    '''
    def _optimize(self, n, mx, my, pnts):
        a, b = n
        return new_eSKLdiv(mx, (my * a)+b, pnts)


def CompAlignData(BaseAlignData):
    '''
    Generate 'compensation' alignment: (only estimate off diagonals)
    '''
    def _get_x0(self, y):
        a = np.zeros(self.d ** 2 - self.d)
        b = np.zeros(self.d)
        return np.hstack(a, b)

    def _format_z(self, z):
        a = np.eye(self.d)
        b = np.zeros(self.d)
        counter = 0
        for i in range(self.d):
            for j in range(self.d):
                if i == j:
                    pass
                else:
                    a[i, j] = z[counter]
                    counter += 1
        return a, b

    def _optimize(self, n):
        a, b = self._format_z(n)
        return eKLdiv(self.mx, (self.my * a), self.pnts)


def CompAlignDataS(CompAlignData):
    '''
    Generate 'compensation' alignment using symetric KLDivergence
    '''
    def _optimize(self, n):
        a, b = self._format_z(n)
        return eSKLdiv(self.mx, (self.my * a), self.pnts)


def FullAlignData(BaseAlignData):
    '''
    Generate full alignment matrix
    '''
    pass  #will need more work
