from pylab import *
import time
import sys
import densities2 as dens
import numpy

from TestData import case1, case2, case3, case4
import fcm.statistics as stats

print "foo"
#time.sleep(30)
x1, x2, x3, x4 = case1, case2, case3, case4

k = 16
n = 100 # set n = 0 to see prior distribution

phi0 = 0.25
ee = 10
ff = 1
aa = 2
last = 1
nu = 0.1
gamma = 5.0/(phi0*nu)
mus = numpy.concatenate([[[10,3]],
                         numpy.random.multivariate_normal(
                            [0,0],gamma*numpy.eye(2),k-1)])

xs = [x1, x2, x3, x4]
for j, x in enumerate(xs):
    print x.shape
    model = stats.DPMixtureModel(x, nclusts=k, burnin=0, iter=n, last=last)
    model.gamma = gamma
    model.phi0 = phi0
    model.aa = aa
    model.nu = nu
    model.ee = ee
    model.ff = ff
    model.load_mu(mus)
    model.fit(True)
    r = model.get_results()
    c = r.make_modal()
    z = c.classify(x)
    mu = r.mus()
    sg = r.sigmas()

    figure(1)
    subplot(2,2,j+1)
    scatter(x[:,0], x[:,1])
    for m, s in zip(mu, sg):
        Xe, Ye = dens.gauss_ell(m.T, s, dim=[0,1], npoints=100, level=0.5)
        plot(Xe, Ye, '-', linewidth=1)

    figure(2)
    subplot(2,2,j+1)
    scatter(x[:,0], x[:,1], c=z)
    for i, m in enumerate(c.modes()):
    	pass
        #if sum(z==i) > 1:
            #text(m[0], m[1], str(i), va='center', ha='center',
            #     bbox=dict(facecolor='yellow', alpha=0.5))
	#    pass
        #axis([0,12,0,12])

    #del model

#show()
