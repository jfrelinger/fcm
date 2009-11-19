'''
Created on Nov 5, 2009

@author: Jacob Frelinger
'''
from scipy.optimize import fmin
from numpy import around, array, log10
from distributions import mixnormpdf

def efunc(x, pi, mu, sig):
    return -mixnormpdf(x,pi,mu,sig)

def modesearch(pis, mus, sigmas, tol=1e-5):
    n = mus.shape[0]
    precision = int(-1*log10(tol))
    modes = []
    for i in range(n):
        modes.append(fmin(efunc, mus[i,:], args=(pis,mus,sigmas), disp=0))
    modes = around(array(modes), precision)
    tmp = {}
    for i in range(n):
        cur_mode = tuple(modes[i,:].tolist())
        if tmp.has_key(cur_mode):
            tmp[cur_mode].append(i)
        else:
            tmp[cur_mode] = [i]
            
    rslt = {}
    for i,v in enumerate(tmp.itervalues()):
        rslt[i] = v
    return rslt
            
    
            
    
    