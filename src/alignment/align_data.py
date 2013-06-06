'''
Created on May 16, 2013
Author: Jacob Frelinger <jacob.frelinger@duke.edu>
'''

import numpy as np
from scipy.misc import logsumexp
from scipy.optimize import fmin
from fcm.alignment.kldiv import eSKLdiv, eKLdiv


class BaseAlignData(object):
    '''
    base class to align a data set to a reference data set
    '''
    def __init__(self, x, size=100000):

        self.mx = x
        self.d = self.mx.mus.shape[1]

        self.size = size


    def align(self, y, x0=None, *args, ** kwargs):
        '''
        Generate A and B that minimizes the distance between x and y
        '''
        self.my = y
        if x0 is None:
            x0 = self._get_x0()

        #call minimizer on
        z = self._min(self._optimize, x0, *args, **kwargs)
        a, b = self._format_z(z)

        return a, b

    def _min(self, func, x0, **kwargs):
        return fmin(func, x0, **kwargs)


    def _get_x0(self):
        raise NotImplementedError

    def _format_z(self, z):
        raise NotImplementedError

    def _optimize(self, n):
        raise NotImplementedError


class DiagonalAlignData(BaseAlignData):
    '''
    Generate Diagonal only alignment
    '''

    def _get_x0(self):
        shift = self.mx.mus.mean(0) - self.my.mus.mean(0)

        scale = np.diag(self.mx.sigmas.mean(0)) / np.diag(self.my.sigmas.mean(0))
        return np.hstack((scale, shift))

    def _optimize(self, n, mx, my, size):
        a, b = n
        rslt = eKLdiv(mx, (my * a) + b, size)
        print 'rslt', rslt, a,b
        return rslt

    def _min(self, func, x0, *args, **kwargs):
        z = np.zeros(self.d + self.d)
        for i in range(self.d):
            mx = self.mx.get_marginal(i)
            my = self.my.get_marginal(i)
            a, b = fmin(func, x0[[i, self.d + i]], (mx, my, self.size), *args, **kwargs)
            z[i] = a
            z[i + self.d] = b

        return z

    def _format_z(self, z):
        b = z[-self.d:]
        a = np.eye(self.d)
        for i in range(self.d):
            a[i, i] = z[i]

        return a, b


class CompAlignData(BaseAlignData):
    '''
    Generate 'compensation' alignment: (only estimate off diagonals)
    '''
    def _get_x0(self):
        a = np.zeros(self.d ** 2 - self.d)
        return a
    
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

    def _optimize(self, n, mx, my, size):
        a = np.eye(self.d)
        counter = 0
        for i in range(self.d):
            for j in range(self.d):
                if i == j:
                    pass
                else:
                    a[i, j] = n[counter]
                    counter += 1
        rslt = eKLdiv(self.mx, (self.my * a), self.size)
        print 'comp', rslt, a
        return rslt

    def _min(self, func, x0, *args, **kwargs):
        a= fmin(func, x0, (self.mx, self.my, self.size), *args, **kwargs)
        return a

class FullAlignData(BaseAlignData):
    '''
    Generate full alignment matrix
    '''
    pass  #will need more work
