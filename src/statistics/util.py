'''
Created on Nov 5, 2009

@author: Jacob Frelinger
'''
#from scipy.optimize import fmin
#from numpy import around, array, log10   
from __future__ import division
import numpy
from numpy.linalg import solve, inv
from fcm.statistics.distributions import compmixnormpdf, mixnormpdf, mvnormpdf, mixnormrnd

def modesearch(pis, mus, sigmas, tol=1e-6, maxiter=20):
    """find the modes of a mixture of guassians"""

    mdict, sm, unused_spm = _mode_search(pis, mus, sigmas, nk=0, tol=tol, maxiter=maxiter)

    m = numpy.array([i[0] for i in mdict.values()])
    sm = numpy.array(sm)
    tmp = {}

    # use stored index as dict items are not ordered
    for j, key in enumerate(mdict):
        i = key[0]
        cur_mode = tuple(m[j, :].tolist())
        if tmp.has_key(cur_mode):
            tmp[cur_mode].append(i)
        else:
            tmp[cur_mode] = [i]

    rslt = {}
    modes = {}
    for i, key in enumerate(tmp.keys()):
        v = tmp[key]
        rslt[i] = v
        modes[i] = numpy.array(key)
    return modes, rslt




def _mode_search(pi, mu, sigma, nk=0, tol=0.000001, maxiter=20):
    """Search for modes in mixture of Gaussians"""
    k, unused_p = mu.shape
    omega = numpy.copy(sigma)
    a = numpy.copy(mu)

    for j in range(k):
        omega[j] = inv(sigma[j])
        a[j] = solve(sigma[j], mu[j])

    if nk > 0:
        allx = numpy.concatenate([mu, mixnormrnd(pi, mu, sigma, nk)])
    else:
        allx = numpy.copy(mu)
    allpx = mixnormpdf(allx, pi, mu, sigma, use_gpu=False)
    nk += k

    mdict = {} # modes
    sm = [] # starting point of mode search
    spm = [] # density at starting points

    etol = numpy.exp(tol)
    # rnd = int(-1*numpy.floor(numpy.log10(tol)))
    rnd = 1

    for js in range(nk):
        x = allx[js]
        px = allpx[js]
        sm.append(x)
        spm.append(px)
        # w = compmixnormpdf(allx,pi,mu,sigma)
        h = 0
        eps = 1 + etol

        while ((h <= maxiter) and (eps > etol)):
            w = compmixnormpdf(x, pi, mu, sigma, use_gpu=False)
            Y = numpy.sum([w[j] * omega[j] for j in range(k)], 0)
            yy = numpy.dot(w, a)
            y = solve(Y, yy)
            py = mixnormpdf(y, pi, mu, sigma, use_gpu=False)
            eps = py / px
            x = y
            px = py
            h += 1

        mdict[(js, tuple(x))] = [numpy.round(x, rnd), px] # eliminate duplicates
    return mdict, sm, spm

def check_mode(m, pm, pi, mu, sigma):
    """Check that modes are local maxima"""
    k, p = mu.shape
    z = numpy.zeros(p)
    tm = [] # true modes
    tpm = [] # true mode densities
    for _m, _pm in zip(m, pm):
        G = numpy.zeros((p, p))
        omega = sigma.copy()
        for j in range(k):
            omega[j] = inv(sigma[j])
            eij = _m - mu[j]
            S = numpy.dot(numpy.identity(p) - numpy.outer(eij, eij),
                          omega[j])
            G += pi[j] * mvnormpdf(eij, z, sigma[j], use_gpu=False) * S
        if numpy.linalg.det(G) > 0:
            tm.append(_m)
            tpm.append(_pm)
    return numpy.array(tm), numpy.array(tpm)


if __name__ == '__main__':
    pi = numpy.array([0.3, 0.4, 0.29, 0.01])
    mu = numpy.array([[0, 0],
                      [1, 3],
                      [-3, -1],
                      [1.2, 2.8]], 'd')
    sigma = numpy.array([1 * numpy.identity(2),
                         1 * numpy.identity(2),
                         1 * numpy.identity(2),
                         1 * numpy.identity(2)])
    modes, rslt = modesearch(pi, mu, sigma)

    for key in modes:
        print key, modes[key]
    print
    for key in rslt:
        print key, rslt[key]
