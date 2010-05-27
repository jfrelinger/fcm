"""
Distributions used in FCS analysis
"""

from numpy import pi, exp, dot, ones, array, sqrt, fabs, tile, sum, prod, diag, zeros
from numpy.linalg import inv, det, cholesky
from cdp import mvnpdf
#def mvnormpdf(x, mu, va):
#    """
#    multi variate normal pdf, derived from David Cournapeau's em package
#    for mixture models
#    http://www.ar.media.kyoto-u.ac.jp/members/david/softwares/em/index.html
#    """
#    d       = mu.size
#    inva    = inv(va)
#    fac     = 1 /sqrt( (2*pi) ** d * fabs(det(va)))
#
#    y   = -0.5 * dot(dot((x-mu), inva) * (x-mu), 
#                       ones((mu.size, 1), x.dtype))
#
#    y   = fac * exp(y)
#    return y

def mvnormpdf(x, mu, va):
    try:
        n,p = x.shape
    except ValueError:
        n = x.shape
        p = 0
    if p > 0:
        results = zeros((n,))
        for i in range(n):
            results[i] = mvnpdf(x[i,:],mu,va)
    else:
        results = array([mvnpdf(x,mu,va)])
    
    return results

def mixnormpdf(x, prop, mu, Sigma):
    """Mixture of multivariate normal pdfs for maximization"""
    # print "in mixnormpdf ..."
    tmp = 0.0
    for i in range(len(prop)):
        tmp += prop[i]*mvnormpdf(x, mu[i], Sigma[i])
    return tmp


if __name__ == '__main__':
    x = array([1,0])
    mu = array([5,5])
    sig = array([[1,0],[0, 1]])

    print 'new:', mvnormpdf(x,mu,sig)
    x = array([1,0])
    mu = array([0,0])
    sig = array([[1,.75],[.75, 1]])

    print 'new:', mvnormpdf(x,mu,sig)
    print 'array:', mvnormpdf(array([x,x-1]), mu, sig)
    
    x = array([[1,0],[5,5],[0,0]])
    mu = array([[0,0],[5,5]])
    sig = array([[[1,.75],[.75, 1]],[[1,0],[0,1]]])
    p = array([.5,.5])
    print 'mix:', mixnormpdf(x,p,mu,sig)