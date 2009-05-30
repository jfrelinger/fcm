"""
Distributions used in FCS analysis
"""

from numpy import transpose, shape, pi, exp, dot, kron, ones, diag, reshape, prod
from numpy.linalg import cholesky, inv

def mvnormpdf(x, mu, Sigma):
    """Multivariate normal density."""
    if len(x.shape) == 1:
        x = reshape(x, (len(x), 1))
    p, n = shape(x)
    c = cholesky(Sigma)
    e = dot(inv(c).T, x - transpose(kron(ones((n, 1)), mu)))
    return exp(-sum(e*e, 0)/2.0)/(prod(diag(c))*(2.0*pi)**(p/2.0))

def mixnormpdf(x, prop, mu, Sigma):
    """Mixture of multivariate normal pdfs for maximization"""
    # print "in mixnormpdf ..."
    tmp = 0.0
    for i in range(len(prop)):
        # print i,
        tmp += prop[i]*mvnormpdf(x, mu[i], Sigma[i])
    return tmp