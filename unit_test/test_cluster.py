import sys
sys.path.append("/home/jolly/MyPython") 

from fcm import loadFCS
from fcm.statistics import DPMixtureModel
from pylab import scatter, show, subplot
import numpy




if __name__ == '__main__':
    #load data
    data = loadFCS('../sample_data/3FITC_4PE_004.fcs')
    #data = loadFCS('../sample_data/coulter.lmd')
    
    # initalize model
    model = DPMixtureModel(data, 16, iter=10, burnin=10)
    
    # fit model
    model.fit(verbose=True)
    
    # get results
    baz = model.get_results()
    print model.get_class()
    print numpy.max(model.get_class())
    print numpy.min(model.get_class())
    #print model._getp(1)
    # pull out all means
    mus = baz.mus()
    print baz.pis()
    
    #plot results.
    for i,j,k in [(1,0,1),(2,0,2),(3,0,3), (4,2,3)]:
        subplot(2,2,i)
        scatter(data[:,j], data[:,k], s=.1)
        scatter(mus[:,j], mus[:,k], c='r')
        
    show()
