import sys
sys.path.append("/home/jolly/MyPython") 

from fcm import loadFCS
from fcm.statistics import DPMixtureModel
from pylab import scatter, show, subplot, figure, title
from fcm.graphics.plot import heatmap

import time
import numpy




if __name__ == '__main__':
    #load data
    data = loadFCS('../sample_data/3FITC_4PE_004.fcs')
    #data = loadFCS('../sample_data/coulter.lmd')

    # heatmap(data, [(0,1),(0,2),(0,3),(2,3)], s=1, edgecolors='none')

    k = 16
    niter = 20
    nburnin = 0
    print "Data dimensions: ", data.shape
    print "Fitting %d component model" % k

    # initalize model
    model = DPMixtureModel(data, k, iter=niter, burnin=nburnin, last=5)
    m2 = DPMixtureModel(data,k, iter=0, burnin=nburnin, last=0)
    xm = model.m
    xs = model.s
    model.load_mu(numpy.zeros((16,4)))
    model.sample_mu = False
    # fit model
    start = time.clock()
    print "about to fit"
    model.fit(verbose=True)
    print "Time for %d iterations: %.2f seconds" % (niter + nburnin, 
                                                    time.clock() - start)
    m2.fit()
    for i in range(niter-5):
        m2.step()
    r = []
    for i in range(5):
        r.append(m2.step())
    # get results
    baz = model.get_results()
    co = model.get_class()
    print model.get_class()
    print numpy.max(model.get_class())
    print numpy.min(model.get_class())
    #print model._getp(1)
    # pull out all means
    mus = baz.mus()[-k:]
    pis = baz.pis()[-k:]
    #idx = numpy.where(pis > 1e-3, True, False)
    #mod = baz.make_modal(tol=1e-5)
    #modes = mod.modes()
    print mus
    print r[-1].mus()
    # print baz.pis()
    # mus = mus*xs + xm
    
    #plot results.
    for i,j,k in [(1,0,1),(2,0,2),(3,0,3), (4,2,3)]:
        #title('tol=1e-5')
        subplot(2,2,i)
        scatter(data[:,j], data[:,k], s=.1, c=co, edgecolors='none')
        scatter(mus[:,j], mus[:,k], c='r')
        scatter(r[-1].mus()[:,j], r[-1].mus()[:,k], c='b')
        #scatter(modes[:,j], modes[:,k], c='b')
        
    show()

#     ss = 1500*pis
#     pis[pis<20] = 20
#     for i,j,k in [(1,0,1),(2,0,2),(3,0,3), (4,2,3)]:
#         subplot(2,2,i)
#         scatter(data[::10,j], data[::10,k], s=.1)
#         scatter(mus[:,j], mus[:,k], s=ss, c=cs)

#     show()
