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


def eSKLdiv(x, y, dim, pnts, lp=None, a=None, b=None, orig_y=None, use_grad=False):
    if lp is None:
        #p = np.sum(x.prob(pnts, use_gpu=False), 1)
        lp = logsumexp(x.prob(pnts, logged=True, use_gpu=True), 1)
    p = np.exp(lp)
    try:
        lq = logsumexp(y.prob(pnts, logged=True, use_gpu=True), 1)
        q = np.exp(lq)
    except ValueError, e:
        print y.sigmas, y.mus
        print 'a', a, 'b', b
        raise e


    sp = np.sum(p)
    p = p / sp
    lp = lp - np.log(sp)

    sq = np.sum(q)
    q = q / sq
    lq = lq - np.log(sq)

    r = p * (lp - lq) + q * (lq - lp)

    return np.sum(r[np.isfinite(r)])
