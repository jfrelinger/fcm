"""
Distributions used in FCS analysis
"""

from numpy import pi, exp, dot, ones, array, sqrt, fabs
from numpy.linalg import inv, det


def mvnormpdf(x, mu, va):
    """
    multi variate normal pdf, derived from David Cournapeau's em package
    for mixture models
    http://www.ar.media.kyoto-u.ac.jp/members/david/softwares/em/index.html
    """
    d       = mu.size
    inva    = inv(va)
    fac     = 1 /sqrt( (2*pi) ** d * fabs(det(va)))

    y   = -0.5 * dot(dot((x-mu), inva) * (x-mu), 
                       ones((mu.size, 1), x.dtype))

    y   = fac * exp(y)
    return y


def mixnormpdf(x, prop, mu, Sigma):
    """Mixture of multivariate normal pdfs for maximization"""
    # print "in mixnormpdf ..."
    tmp = 0.0
    for i in range(len(prop)):
        # print i,
        tmp += prop[i]*mvnormpdf(x, mu[i], Sigma[i])
    return tmp


if __name__ == '__main__':
    x = array([1,0])
    mu = array([0,0])
    sig = array([[1,0],[0, 1]])

    print 'new:', mvnormpdf(x,mu,sig)
    x = array([1,0])
    mu = array([0,0])
    sig = array([[1,.75],[.75, 1]])

    print 'new:', mvnormpdf(x,mu,sig)
    print 'array:', mvnormpdf(array([x,x]), mu, sig)