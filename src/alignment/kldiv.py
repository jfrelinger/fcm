'''
Created on May 16, 2013
Author: Jacob Frelinger <jacob.frelinger@duke.edu>
'''

from __future__ import division
import numpy as np
from numpy.linalg import det, solve, LinAlgError
from scipy.misc import logsumexp

def inv(x):
    return solve(x, np.eye(x.shape[0]))


def true_kldiv(m0, m1, s0, s1):
    k = m0.shape[0]
    if s1[0,0] == 0:
        print 'FIXING BAD S1!!!!!!!!!!!!!!!!!'
        s1[0,0] = 1e-5
    try:
        first = np.trace(np.dot(inv(s1), s0))
    except LinAlgError, e:
        print 'error:',s1, s0
        raise e
    mdiff = m1 - m0
    try:
        second = np.dot(mdiff.T, np.dot(inv(s1), mdiff))
    except LinAlgError, e:
        print 'error:',s1, s0
        raise e
    third = np.log(det(s0)) - np.log(det(s1))

    return .5 * (first + second - third - k)

def true_skldiv(m0, m1, s0, s1):
    return true_kldiv(m0, m1, s0, s1) + true_kldiv(m1, m0, s1, s0)



def eKLdiv(x, y, n=100000, **kwargs):
    pnts = x.draw(n).reshape(n, -1)
    #return (1.0 / n) * np.sum(np.log(np.sum(x.prob(px, use_gpu=True), 1) / np.sum(y.prob(px, use_gpu=True), 1)))
    px = x.prob(pnts, logged=True, **kwargs)
    if len(px.shape) == 1:
       px = px.reshape(n,1)
    py = y.prob(pnts, logged=True, **kwargs)
    if len(py.shape) == 1:
       py = py.reshape(n,1)
    return np.max(1.0/n * (np.sum(logsumexp(px, 1)) - np.sum(logsumexp(py, 1)) ),0)

def eSKLdiv(x, y, n):
    return new_eKLdiv(x, y, n) + new_eKLdiv(y, x, n)


def eKLdivVar(x,y, n=100000, **kwargs):
    if len(x) == 1 and len(y) == 1:
        return true_kldiv(x[0].mu, y[0].mu, x[0].sigma, y[0].sigma)
    z = []
    for i in x.clusters:
        #numerator = np.log(np.sum([ j.pi * np.exp(-1*true_kldiv(i.mu,j.mu,i.sigma,j.sigma)) for j in x.clusters]))
        numerator = logsumexp(np.array([-1*true_kldiv(i.mu, j.mu, i.sigma, j.sigma) for j in x.clusters]), b=x.pis)
        #denominator = np.log(np.sum([ j.pi * np.exp(-1*true_kldiv(i.mu, j.mu, i.sigma, j.sigma)) for j in y.clusters]))
        denominator = logsumexp(np.array([-1*true_kldiv(i.mu, j.mu, i.sigma, j.sigma) for j in y.clusters]), b=y.pis)
        z.append(i.pi*(numerator-denominator))
    return max(np.sum(z),0)
        
        
def eKLdivVarU(x,y,n):
    z = []
    for i in x.clusters:
        for j in y.clusters:
            z.append(i.pi*j.pi*true_kldiv(i.mu,j.mu,i.sigma,j.sigma))
    return np.sum(z)
