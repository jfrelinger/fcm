'''
Created on Nov 5, 2009

@author: Jacob Frelinger
'''
#from scipy.optimize import fmin
#from numpy import around, array, log10   
from __future__ import division
import numpy
from numpy.linalg import solve, inv
from fcm.statistics.distributions import mixnormpdf, mvnormpdf

def modesearch(pis, mus, sigmas, tol=1e-5, maxiter=20):
    n = mus.shape[0]
    mdict, sm, spm = mode_search(pis, mus, sigmas, nk=0, tol=tol, maxiter=maxiter)

    m = numpy.array([i[0] for i in mdict.values()])
    pm = numpy.array([i[1] for i in mdict.values()])
    # m = numpy.array(mdict.keys())
    # pm = numpy.array(mdict.values())
    sm = numpy.array(sm)
    tm, tpm = check_mode(m, pm, pis, mus, sigmas)
    
    tmp = {}
    # for i in range(n):
    for i in range(len(tm)):
        cur_mode = tuple(tm[i,:].tolist())
        # cur_mode = tuple(_tm.tolist())
        if tmp.has_key(cur_mode):
            tmp[cur_mode].append(i)
        else:
            tmp[cur_mode] = [i]
             
    rslt = {}
    for i,v in enumerate(tmp.itervalues()):
        rslt[i] = v
    return rslt


def mixnormrnd(pi, mu, sigma, k):
    """Generate random variables from mixture of Guassians"""
    xs = []
    for i in range(k):
        j = numpy.sum(numpy.random.random() > numpy.cumsum(pi))
        xs.append(numpy.random.multivariate_normal(mu[j],sigma[j]))
    return numpy.array(xs)                  

def mode_search(pi, mu, sigma, nk=0, tol=0.000001, maxiter=20):
    """Search for modes in mixture of Gaussians"""

    k,p = mu.shape
    omega = sigma[:]
    a = numpy.copy(mu)

    for j in range(k):
        omega[j] = inv(sigma[j])
        a[j] = solve(sigma[j], mu[j])

    if nk > 0:
        allx = numpy.concatenate([mu,mixnormrnd(pi,mu,sigma,nk)])
    else:
        allx = numpy.copy(mu)
    allpx = mixnormpdf(allx, pi, mu, sigma)
    nk += k

    mdict = {} # modes
    sm = [] # starting point of mode search
    spm = [] # density at starting points

    etol = numpy.exp(tol)
    rnd = -1*numpy.floor(numpy.log10(tol))
            

    for js in range(nk):
        x = allx[js]
        px = allpx[js]

        sm.append(x)
        spm.append(px)

        h = 0
        eps = 1+etol
        while ((h<=maxiter) and (eps>etol)):
            y = numpy.zeros(p)
            Y = numpy.zeros((p,p))
            
            for j in range(k):
                w = pi[j]*mvnormpdf(x, mu[j], sigma[j])
                Y += w*omega[j]
                y += w*a[j]
            y = solve(Y, y)
            py = mixnormpdf(y, pi, mu, sigma)
            eps = py/px
            
            x = y
            px = py
            h += 1

        # mdict[tuple(allx[js])] = [numpy.round(x,rnd),px] # eliminate duplicates
        mdict[tuple(numpy.round(x,2))] = [numpy.round(x,rnd),px] # eliminate duplicates

    return mdict, sm, spm

def check_mode(m, pm, pi, mu, sigma):
    """Check that modes are local maxima"""
    k,p = mu.shape
    z = numpy.zeros(p)
    tm = [] # true modes
    tpm = [] # true mode densities
    for _m, _pm in zip(m, pm):
        G = numpy.zeros((p,p))
        omega = sigma.copy()
        for j in range(k):
            omega[j] = inv(sigma[j])
            eij = _m-mu[j]
            S = numpy.dot(numpy.identity(p)-numpy.outer(eij, eij),
                          omega[j])
            G += pi[j]*mvnormpdf(eij, z, sigma[j])*S
        if numpy.linalg.det(G) > 0:
            tm.append(_m)
            tpm.append(_pm)
    return numpy.array(tm), numpy.array(tpm)    
    
    
