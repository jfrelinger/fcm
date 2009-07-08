from scipy.optimize import fsolve, brentq
from scipy import interpolate, weave
from numpy import arange, exp, log, min, max, sign, concatenate, zeros, vectorize
from numpy.random import normal, lognormal, shuffle
import pylab
import time

def quantile(x, n):
    try:
        return sorted(x)[int(n*len(x))]
    except IndexError:
        return 0

def productlog(x, prec=1e-12):
    """Productlog or LambertW function computes principal solution for w in f(w)
 = w*exp(w).""" 
    #  fast estimate with closed-form approximation
    if (x <= 500):
        lxl = log(x + 1.0)
        return 0.665 * (1+0.0195*lxl) * lxl + 0.04
    else:
        return log(x - 4.0) - (1.0 - 1.0/log(x)) * log(log(x))

def S(x, y, T, m, w):
    p = w/(2*productlog(0.5*exp(-w/2)*w))
    sgn = sign(x-w)
    xw = sgn*(x-w)
    return sgn*T*exp(-(m-w))*(exp(xw)-p**2*exp(-xw/p)+p**2-1) - y

def logicle0(y, T, m, r):
    if r>=0:
        return m+log((exp(-m)*T+y)/T)
    else:
        w = (m-log(T/abs(r)))/2
        return brentq(S, -100, 100, (y, T, m, w))
logicle0 = vectorize(logicle0)

def logicle(y, T, m, r, order=2, intervals=1000.0):
    ub = log(max(y)+1-min(y))
    xx = exp(arange(0, ub, ub/intervals))-1+min(y)
    yy = logicle0(xx, T, m, r)
    t = interpolate.splrep(xx, yy, k=order)
    return interpolate.splev(y, t)

if __name__ == '__main__':
    d1 = normal(0, 50, (50000))
    d2 = lognormal(8, 1, (50000))
    d3 = concatenate([d1, d2])

    T = 262144
    d = 4
    m = d*log(10)
    r = quantile(d3[d3<0], 0.05)
    w = (m-log(T/abs(r)))/2

    pylab.clf()
    pylab.figtext(0.5, 0.94, 'Logicle transform with r=%.2f, d=%d and T=%d\nData is normal(0, 50, 50000) + lognormal(8, 1, 50000)' % (r, d, T),
                  va='center', ha='center', fontsize=12)

    pylab.subplot(3,1,1)
    x = arange(0, m, 0.1)
    pylab.plot(x, S(x, 0, T, m, w))
    locs, labs = pylab.xticks()
    pylab.xticks([])
    pylab.yticks([])
    pylab.ylabel('Inverse logicle')

    pylab.subplot(3,1,2)
    pylab.hist(d3, 1250)
    locs, labs = pylab.xticks()
    pylab.xticks([])
    pylab.yticks([])
    pylab.ylabel('Raw data')

    pylab.subplot(3,1,3)
    pylab.hist(logicle(d3, T, m, r), 1250)
    locs, labs = pylab.xticks()
    pylab.xticks([])
    pylab.yticks([])
    pylab.ylabel('Data after transform')

    # pylab.savefig('logicle.png')
    pylab.show()
