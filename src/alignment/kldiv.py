"""
Created on May 16, 2013
Author: Jacob Frelinger <jacob.frelinger@duke.edu>
"""

from __future__ import division
import numpy as np
from numpy.linalg import det, solve, LinAlgError
from scipy.misc import logsumexp


def inv(x):
    return solve(x, np.eye(x.shape[0]))


def true_kldiv(m0, m1, s0, s1):
    k = m0.shape[0]
    if s1[0, 0] == 0:
        print 'FIXING BAD S1!!!!!!!!!!!!!!!!!'
        s1[0, 0] = 1e-5
    try:
        first = np.trace(np.dot(inv(s1), s0))
    except LinAlgError as e:
        print 'error:', s1, s0
        raise e
    mdiff = m1 - m0
    try:
        second = np.dot(mdiff.T, np.dot(inv(s1), mdiff))
    except LinAlgError as e:
        print 'error:', s1, s0
        raise e
    third = np.log(det(s0)) - np.log(det(s1))

    return .5 * (first + second - third - k)


def true_skldiv(m0, m1, s0, s1):
    return true_kldiv(m0, m1, s0, s1) + true_kldiv(m1, m0, s1, s0)


def KLdivDiff(p_mean, p_sig, q_mean, q_sig, a, b):
    a = a.T
    asai = inv(np.dot(a, np.dot(q_sig, a.T)))
    asig = np.dot(a, q_sig)
    mdiff = p_mean - np.dot(a, q_mean) - b

    rf = (
        -1 * asig) + \
        np.outer(mdiff, q_mean) + \
        np.dot(np.dot(p_sig + (np.outer(mdiff, mdiff)), asai), asig)
    return np.hstack(
        [np.dot(-1 * asai, rf).T.flatten(), (np.dot(-1 * asai, mdiff))])


def eKLdiv(x, y, n=100000, **kwargs):
    pnts = x.draw(n).reshape(n, -1)
    px = x.prob(pnts, logged=True, **kwargs)
    if len(px.shape) == 1:
        px = px.reshape(n, 1)
    py = y.prob(pnts, logged=True, **kwargs)
    if len(py.shape) == 1:
        py = py.reshape(n, 1)
    return np.max(
        1.0 / n * (np.sum(logsumexp(px, 1)) - np.sum(logsumexp(py, 1))), 0)


def eSKLdiv(x, y, n):
    return new_eKLdiv(x, y, n) + new_eKLdiv(y, x, n)


def eKLdivVar(x, y, n=100000, **kwargs):
    if len(x) == 1 and len(y) == 1:
        return true_kldiv(x[0].mu, y[0].mu, x[0].sigma, y[0].sigma)
    z = []
    for i in x.clusters:
        numerator = logsumexp(np.array(
            [-1 * true_kldiv(i.mu, j.mu, i.sigma, j.sigma) for j in x.clusters]), b=x.pis)
        denominator = logsumexp(np.array(
            [-1 * true_kldiv(i.mu, j.mu, i.sigma, j.sigma) for j in y.clusters]), b=y.pis)
        z.append(i.pi * (numerator - denominator))
    return max(np.sum(z), 0)


def eSKLdivVar(x, y, n=100000, **kwargs):
    return eKLdivVar(x, y, n, **kwargs) + eKLdivVar(y, x, n, **kwargs)


def eKLdivVarU(x, y):
    z = []
    for i in x.clusters:
        for j in y.clusters:
            z.append(i.pi * j.pi * true_kldiv(i.mu, j.mu, i.sigma, j.sigma))
    return np.sum(z)


def eKLdivVarDiff(x, y, my, a, b):
    z = []
    for i in x.clusters:
        divs = np.array(
            [KLdivDiff(i.mu, i.sigma, j.mu, j.sigma, a, b) for j in y.clusters])
        numerator = np.sum([j.pi *
                            np.exp(-
                                   1 *
                                   true_kldiv(i.mu, j.mu, i.sigma, j.sigma)) *
                            divs[k] for k, j in enumerate(my.clusters)], axis=0)
        denominator = np.sum(np.array(
            [j.pi * np.exp(-1 * true_kldiv(i.mu, j.mu, i.sigma, j.sigma)) for j in my.clusters]), axis=0)
        z.append(i.pi * numerator / denominator)
    return np.sum(z, axis=0)
