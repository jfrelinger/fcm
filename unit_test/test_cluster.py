from fcm import loadFCS
from fcm.statistics import DPMixtureModel
from pylab import scatter, show
import numpy
import pickle

if __name__ == '__main__':
    data = loadFCS('/home/jolly/Projects/fcm/sample_data/3FITC_4PE_004.fcs')
    model = DPMixtureModel(data, 20, itter=100, burnin=10)
    model.fit(verbose=True)
    baz = model.get_results()
    
    scatter(data[:,0], data[:,1], s=.1)
    scatter(baz.mus()[:,0], baz.mus()[:,1], c='r')
        
    show()