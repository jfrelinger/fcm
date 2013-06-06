'''
Created on May 16, 2013
Author: Jacob Frelinger <jacob.frelinger@duke.edu>
'''

from __future__ import division
import numpy as np
from numpy.linalg import det, inv
from scipy.misc import logsumexp

def true_kldiv(m0, m1, s0, s1):
    k = m0.shape[0]
    first = np.trace(np.dot(inv(s1), s0))
    mdiff = m1 - m0
    second = np.dot(mdiff.T, np.dot(inv(s1), mdiff))
    third = np.log(det(s0) / det(s1))

    return .5 * (first + second - third - k)

def true_skldiv(m0, m1, s0, s1):
    return true_kldiv(m0, m1, s0, s1) + true_kldiv(m1, m0, s1, s0)


def eSKLdiv(x, y, pnts, lp=None):
    if len(pnts.shape) > 1:
        axis = 1
    else:
        axis = 0
    if lp is None:
        #p = np.sum(x.prob(pnts, use_gpu=False), 1)
        lp = logsumexp(x.prob(pnts, logged=True, use_gpu=True), axis)
    p = np.exp(lp)
    try:
        lq = logsumexp(y.prob(pnts, logged=True, use_gpu=True), axis)
        q = np.exp(lq)
    except ValueError, e:
        raise e


    sp = np.sum(p)
    p = p / sp
    lp = lp - np.log(sp)

    sq = np.sum(q)
    q = q / sq
    lq = lq - np.log(sq)

    r = p * (lp - lq) + q * (lq - lp)
    if axis:
        return np.sum(r[np.isfinite(r)])
    else:
        return np.sum(r)

def eKLdiv(x, y, pnts, lp=None):
    if len(pnts.shape) > 1:
        axis = 1
    else:
        axis = 0
    if lp is None:
        #p = np.sum(x.prob(pnts, use_gpu=False), 1)
        lp = logsumexp(x.prob(pnts, logged=True, use_gpu=True), axis)
    p = np.exp(lp)
    try:
        lq = logsumexp(y.prob(pnts, logged=True, use_gpu=True), axis)
        q = np.exp(lq)
    except ValueError, e:
        raise e


    sp = np.sum(p)
    p = p / sp
    lp = lp - np.log(sp)

    sq = np.sum(q)
    q = q / sq
    lq = lq - np.log(sq)

    r = p * (lp - lq)
    
    if axis:
        return np.sum(r[np.isfinite(r)])
    else:
        return np.sum(r)
    
    
def new_eKLdiv(x,y,pnts):
    n=100000
    px = x.draw(n).reshape(n,1)
    return (1.0/n)*np.sum(np.log(np.sum(x.prob(px),1)/np.sum(y.prob(px),1)))


def new_eSKLdiv(x,y,pnts):
    return new_eKLdiv(x,y,pnts)+new_eKLdiv(y,x,pnts)