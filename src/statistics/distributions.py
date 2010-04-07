"""
Distributions used in FCS analysis
"""

from numpy import pi, exp, dot, ones, array, sqrt, fabs, tile, sum, prod, diag
from numpy.linalg import inv, det, cholesky


def mvnormpdf(x, mu, sigma):
    """
    multi variate normal pdf, takes point/array of points, mu, and sigma and calculates pdf.
    calculates over all points simutaneiously.
    """
    n = x.shape[0]
    try:
        p = x.shape[1]
    except IndexError:
        p = n
        n = 1
    C = cholesky(sigma)
    fmu = tile(mu,(n,1))
    e= dot(inv(C).T,(x-fmu).T)
    num = exp(-sum(e*e, axis=0)/2)
    denom = ( prod(diag(C))*((2*pi)**(p/2)) )
    return num/denom

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