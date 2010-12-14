from fcm.statistics.distributions import mvnormpdf, compmixnormpdf, mixnormpdf
import unittest
from numpy import array, eye, pi, fabs, sqrt, dot, ones, exp, sum
from numpy.linalg import inv, det, cholesky
from random import uniform

def pmvnormpdf(x, mu, va):
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

class mvnpdfTestCase(unittest.TestCase):
    def testSinglePointSingleComponent(self):
        x = array([0,0])
        mu = array([1,1])
        sigma = eye(2)
        
        self.assertAlmostEqual( pmvnormpdf(x,mu,sigma), mvnormpdf(x,mu,sigma), 6,'pmvnormpdf and mvnormpdf differ in result, %f != %f' % (pmvnormpdf(x,mu,sigma), mvnormpdf(x,mu,sigma))) 
        for i in range(100):
            x = array([uniform(-4,4),uniform(-4,4)])
            mu = array([uniform(-4,4),uniform(-4,4)])
            a = uniform(0,4)
            sigma = eye(2)+a
            self.assertAlmostEqual( pmvnormpdf(x,mu,sigma), mvnormpdf(x,mu,sigma), 6,'pmvnormpdf and mvnormpdf differ in result, %f != %f, (%d): %s, %s, %s ' % (pmvnormpdf(x,mu,sigma), mvnormpdf(x,mu,sigma),i, str(x), str(mu), str(sigma).replace('\n',','))) 
        
    def testMultiplePointSingleComponent(self): 
        for i in range(100):
            x = array([[uniform(-4,4),uniform(-4,4)],[uniform(-4,4),uniform(-4,4)]])
            mu = array([uniform(-4,4),uniform(-4,4)])
            a = uniform(0,4)
            sigma = eye(2)+a
            self.assertAlmostEqual( float(pmvnormpdf(x,mu,sigma)[0][0]),
                    float(mvnormpdf(x,mu,sigma)[0]), 6,
                    'pmvnormpdf and mvnormpdf differ in result, %f != %f, (%d): %s, %s, %s ' % (float(pmvnormpdf(x,mu,sigma)[0][0]), float(mvnormpdf(x,mu,sigma)[0]),i, str(x), str(mu), str(sigma).replace('\n',',')))
            self.assertAlmostEqual( float(pmvnormpdf(x,mu,sigma)[1][0]),
                    float(mvnormpdf(x,mu,sigma)[1]), 6,
                    'pmvnormpdf and mvnormpdf differ in result, %f != %f, (%d): %s, %s, %s ' % (float(pmvnormpdf(x,mu,sigma)[1][0]), float(mvnormpdf(x,mu,sigma)[1]),i, str(x), str(mu), str(sigma).replace('\n',',')))
            
    
    def testSinglePointMultipleComponent(self):
        for i in range(100):
            x = array([uniform(-4,4),uniform(-4,4)])
            mu = array([[uniform(-4,4),uniform(-4,4)],[uniform(-4,4),uniform(-4,4)]])
            a = uniform(0,4)
            b = uniform(0,4)
            pi = array([1,1])
            sigma = array([eye(2)+a, eye(2)+b])
            result = array([float(pmvnormpdf(x,mu[0,:],sigma[0,:,:])),float(pmvnormpdf(x,mu[1,:],sigma[1,:,:]))])
            #print result, compmixnormpdf(x,pi,mu,sigma)
            self.assertAlmostEqual( result[0], compmixnormpdf(x,pi,mu,sigma)[0], 6,'pmvnormpdf and mvnormpdf differ in result, %f != %f, (%d): %s, %s, %s ' % (result[0], compmixnormpdf(x,pi,mu,sigma)[0],i, str(x), str(mu[0]), str(sigma[0]).replace('\n',','))) 
            self.assertAlmostEqual( result[1], compmixnormpdf(x,pi,mu,sigma)[1], 6,'pmvnormpdf and mvnormpdf differ in result, %f != %f, (%d): %s, %s, %s ' % (result[1], compmixnormpdf(x,pi,mu,sigma)[1],i, str(x), str(mu[1]), str(sigma[1]).replace('\n',',')))
        
    
    def testMultiplePointMultipleComponent(self):
        for i in range(100):
            x = array([[uniform(-4,4),uniform(-4,4)],[uniform(-4,4),uniform(-4,4)]])
            mu = array([[uniform(-4,4),uniform(-4,4)],[uniform(-4,4),uniform(-4,4)]])
            a = uniform(0,4)
            b = uniform(0,4)
            pi = array([1,1])
            sigma = array([eye(2)+a, eye(2)+b])
            result = array([[float(pmvnormpdf(x[0,:],mu[0,:],sigma[0,:,:])),float(pmvnormpdf(x[0,:],mu[1,:],sigma[1,:,:]))],[float(pmvnormpdf(x[1,:],mu[0,:],sigma[0,:,:])),float(pmvnormpdf(x[1,:],mu[1,:],sigma[1,:,:]))]])
            #print result, compmixnormpdf(x,pi,mu,sigma)
            #print mixnormpdf(x,pi,mu,sigma), sum(result,1)
            self.assertAlmostEqual( result[0,0], compmixnormpdf(x,pi,mu,sigma)[0,0], 6,'pmvnormpdf and mvnormpdf differ in result, %f != %f, (%d): %s, %s, %s ' % (result[0,0], compmixnormpdf(x,pi,mu,sigma)[0,0],i, str(x[0,:]), str(mu[0]), str(sigma[0]).replace('\n',','))) 
            self.assertAlmostEqual( result[1,0], compmixnormpdf(x,pi,mu,sigma)[1,0], 6,'pmvnormpdf and mvnormpdf differ in result, %f != %f, (%d): %s, %s, %s ' % (result[1,0], compmixnormpdf(x,pi,mu,sigma)[1,0],i, str(x[0,:]), str(mu[1]), str(sigma[1]).replace('\n',',')))
            self.assertAlmostEqual( result[0,0], compmixnormpdf(x,pi,mu,sigma)[0,0], 6,'pmvnormpdf and mvnormpdf differ in result, %f != %f, (%d): %s, %s, %s ' % (result[0,1], compmixnormpdf(x,pi,mu,sigma)[0,1],i, str(x[0,:]), str(mu[0]), str(sigma[0]).replace('\n',','))) 
            self.assertAlmostEqual( result[1,0], compmixnormpdf(x,pi,mu,sigma)[1,0], 6,'pmvnormpdf and mvnormpdf differ in result, %f != %f, (%d): %s, %s, %s ' % (result[1,1], compmixnormpdf(x,pi,mu,sigma)[1,0],i, str(x[0,:]), str(mu[1]), str(sigma[1]).replace('\n',',')))
            self.assertAlmostEqual( sum(result,1)[0], mixnormpdf(x,pi,mu,sigma)[0],6,'')
            self.assertAlmostEqual( sum(result,1)[1], mixnormpdf(x,pi,mu,sigma)[1],6,'')
if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testSubSample']
    unittest.main()
