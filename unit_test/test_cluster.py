import sys
sys.path.append("/home/jolly/MyPython") 

from fcm import loadFCS
from fcm.statistics import DPMixtureModel
from pylab import scatter, show, subplot, figure
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
    xm = model.m
    xs = model.s
    
    # fit model
    start = time.clock()
    model.fit(verbose=True)
    print "Time for %d iterations: %.2f seconds" % (niter + nburnin, 
                                                    time.clock() - start)

    # get results
    baz = model.get_results()
    print model.get_class()
    print numpy.max(model.get_class())
    print numpy.min(model.get_class())
    #print model._getp(1)
    # pull out all means
    mus = baz.mus()[-k:]
    pis = baz.pis()[-k:]

    # print baz.pis()
    # mus = mus*xs + xm
    
    #plot results.
    for i,j,k in [(1,0,1),(2,0,2),(3,0,3), (4,2,3)]:
        subplot(2,2,i)
        scatter(data[:,j], data[:,k], s=.1)
        scatter(mus[:,j], mus[:,k], c='r')
        
    show()

#     ss = 1500*pis
#     pis[pis<20] = 20
#     for i,j,k in [(1,0,1),(2,0,2),(3,0,3), (4,2,3)]:
#         subplot(2,2,i)
#         scatter(data[::10,j], data[::10,k], s=.1)
#         scatter(mus[:,j], mus[:,k], s=ss, c=cs)

#     show()
