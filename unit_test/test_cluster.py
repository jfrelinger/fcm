from fcm import loadFCS
from fcm.statistics import DPMixtureModel
from pylab import scatter, show, subplot


if __name__ == '__main__':
    #load data
    data = loadFCS('/home/jolly/Projects/fcm/sample_data/3FITC_4PE_004.fcs')
    
    # initalize model
    model = DPMixtureModel(data, 20, itter=100, burnin=10)
    
    # fit model
    model.fit(verbose=True)
    
    # get results
    baz = model.get_results()
    
    # pull out all means
    mus = baz.mus()
    
    #plot results.
    for i,j,k in [(1,0,1),(2,0,2),(3,0,3), (4,2,3)]:
        subplot(2,2,i)
        scatter(data[:,j], data[:,k], s=.1)
        scatter(mus[:,j], mus[:,k], c='r')
        
    show()