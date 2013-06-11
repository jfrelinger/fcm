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



def eKLdiv(x, y, n=100000):
    px = x.draw(n).reshape(n, -1)
    #return (1.0 / n) * np.sum(np.log(np.sum(x.prob(px, use_gpu=True), 1) / np.sum(y.prob(px, use_gpu=True), 1)))
    return np.max(1.0/n * (np.sum(logsumexp(x.prob(px, use_gpu=True, logged=True), 1)) 
                    - np.sum(logsumexp(y.prob(px, use_gpu=True, logged=True), 1)) ),0)

def eSKLdiv(x, y, n):
    return new_eKLdiv(x, y, n) + new_eKLdiv(y, x, n)


def eKLdivVar(x,y, n):
    z = []
    for i in x.clusters:
        numerator = np.log(np.sum([ j.pi * np.exp(-1*true_kldiv(i.mu,j.mu,i.sigma,j.sigma)) for j in x.clusters]))
        denominator = np.log(np.sum([ j.pi * np.exp(-1*true_kldiv(i.mu, j.mu, i.sigma, j.sigma)) for j in y.clusters]))
        z.append(i.pi*(numerator-denominator))
    return max(np.sum(z),0)
        
        
def eKLdivVarU(x,y,n):
    z = []
    for i in x.clusters:
        for j in y.clusters:
            z.append(i.pi*j.pi*true_kldiv(i.mu,j.mu,i.sigma,j.sigma))
    return np.sum(z)