'''
Created on May 16, 2013
Author: Jacob Frelinger <jacob.frelinger@duke.edu>
'''

import numpy as np
from scipy.misc import logsumexp
from scipy.optimize import minimize
from fcm.alignment.kldiv import eKLdivVar


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

    def _min(self, func, x0, *args, **kwargs):
        z = minimize(func, x0, args=(self.mx, self.my, self.size), *args, **kwargs)
        return z.x


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
#        shift = self.mx.mus.mean(0) - self.my.mus.mean(0)
#
#        scale = np.diag(self.mx.sigmas.mean(0)) / np.diag(self.my.sigmas.mean(0))
        x = self.mx.draw(self.size)
        y = self.my.draw(self.size)
        shift = -1 * y.mean(0) * x.std(0) / y.std(0) + x.mean(0)
        scale = x.std(0) / y.std(0)

        return np.hstack((scale.flatten(), shift))

    def _optimize(self, n, mx, my, size):
        a, b = self._format_z(n)
        rslt = eKLdivVar(mx, (my * a) + b, size)
        return rslt

    def _min(self, func, x0, *args, **kwargs):
        r = minimize(func, x0, args=(self.mx, self.my, self.size), *args, **kwargs)

        return r.x

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
        rslt = eKLdivVar(self.mx, (self.my * a), self.size)
        return rslt

    def _min(self, func, x0, *args, **kwargs):
        a = minimize(func, x0, args=(self.mx, self.my, self.size), *args, **kwargs)
        return a.x

class FullAlignData(BaseAlignData):
    '''
    Generate full alignment matrix
    '''
    def __init__(self, x, size=100000, exclude=None):
        if exclude is None:
            exclude = []
            
        super(FullAlignData, self).__init__(x, size)
          
        self.exclude = exclude
        self.include = np.array([i for i in range(self.d) if i not in exclude])
        self.d_sub = len(self.include)
        self.mxm = self.mx.get_marginal(self.include)

    def align(self, y, x0=None, *args, ** kwargs):
        '''
        Generate A and B that minimizes the distance between x and y
        '''
        self.my = y
        self.mym = self.my.get_marginal(self.include)
        
        if x0 is None:
            x0 = self._get_x0()


        #call minimizer on
        z = self._min(self._optimize, x0, *args, **kwargs)
        a_sub, b_sub = self._format_z(z)
        a = x0[0:self.d**2].reshape((self.d,self.d))
        a[np.ix_(self.include,self.include)] = a_sub
        b = x0[-self.d:]
        b[self.include] = b_sub
        
        return a,b
    
    def _format_z(self, z):
        a = z[0:self.d_sub ** 2].reshape(self.d_sub, self.d_sub)
        b = z[-self.d_sub:]
        return a, b

    def _optimize(self, n, mx, my, size):
        a, b = self._format_z(n)
        rslt = eKLdivVar(self.mxm, (self.mym * a) + b, size)
        return rslt

    def _min(self, func, x0, *args, **kwargs):
        x0 = self._format_x0(x0)
        z = minimize(func, x0, args=(self.mxm, self.mym, self.size), *args, **kwargs)
        return z.x

    def _get_x0(self):
        m = DiagonalAlignData(self.mx, self.size)
        scale, shift = m.align(self.my, options={'disp':False})
#        shift = self.mx.mus.mean(0) - self.my.mus.mean(0)
#
#        scale = np.diag(self.mx.sigmas.mean(0)) / np.diag(self.my.sigmas.mean(0))
        return np.hstack((scale.flatten(), shift))

    def _format_x0(self, x0):
        scale = x0[0:self.d**2].reshape(self.d,self.d)
        shift = x0[-self.d:]
        scale = scale[np.ix_(self.include,self.include)]
        shift = shift[self.include]
        return np.hstack((scale.flatten(), shift))