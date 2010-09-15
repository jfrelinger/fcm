"""
A python object representing flow cytomoetry data
"""
from __future__ import division
from numpy import array, median, mean, std, log
from enthought.traits.api import HasTraits, String, Instance, List
from annotation import Annotation
from fcmexceptions import BadFCMPointDataTypeError
from transforms import logicle as _logicle
from transforms import hyperlog as _hyperlog
from transforms import log_transform as _log
from util import Tree, RootNode, fcmlog

class FCMdata(HasTraits):
    """
    Object representing flow cytometry data
    FCMdata.pnts : a numpy array of data points
    FCMdata.channels : a list of which markers/scatters are on which column of
                         the array.
    FCMdata.scatters : a list of which indexes in fcmdata.channels are scatters

    """
    
    name = String()
    tree = Instance(Tree)
    scatters = List()
    markers = List()
    notes = Instance(Annotation)

    @fcmlog
    def __init__(self, name, pnts, channels, scatters=None, notes=None):
        """
        fcmdata(name, pnts, channels, scatters=None)
        name: name of corresponding FCS file minus extension
        pnts: array of data points
        channels: a list of which markers/scatters are on which column of
                    the array.
        scatters: a list of which indexes in channels are scatters
        
        """
        self.name = name
#        if type(pnts) != type(array([])):
#            raise BadFCMPointDataTypeError(pnts, "pnts isn't a numpy.array")
        self.tree = Tree(pnts, channels)
        #self.pnts = pnts
        #self.channels = channels
        #TODO add some default intelegence for determining scatters if None
        self.scatters = scatters
        self.markers = []
        if self.scatters is not None:
            for chan in range(len(channels)):
                if chan in self.scatters:
                    pass
                elif self.tree.root.channels[chan] in self.scatters:
                    pass
                else:
                    self.markers.append(chan)
        if notes == None:
            notes = Annotation()
        self.notes = notes

    def __unicode__(self):
        return self.name

    def __repr__(self):
        return self.name
        
    def __getitem__(self, item):
        """return FCMdata.pnts[i] by name or by index"""
        
        if type(item) == type(''):
            try:
                return self.get_channel_by_name(item)
            except:
                raise ValueError('field named a not found')
        elif type(item) == tuple:
            if type(item[0]) == type(''):
                return self.get_channel_by_name(list(item))
            else:
                return self.tree.view()[item]
        else:
            return self.tree.view()[item]

    def __getattr__(self, name):
        if name == 'channels':
            return self.current_node().channels
        else:
            return self.tree.view().__getattribute__(name)
                
    def name_to_index(self, channels):
        """Return the channel indexes for the named channels"""
        
        if type(channels) == type(''):
            channels = [channels]
        to_return = [ self.channels.index(i) for i in channels]
        return to_return
    
    def get_channel_by_name(self, channels):
        """Return the data associated with specific channel names"""
        
        return self.tree.view()[:, self.name_to_index(channels)]
    
    def get_markers(self):
        """return the data associated with all the markers"""
        
        return self.view()[:, self.markers]
    
    def get_spill(self):
        """return the spillover matrix from the original fcs used in compisating""" 
        try:
            return self.notes.text['spill']
        except KeyError:
            return None
    
    @fcmlog
    def view(self):
        """return the current view of the data"""
        return self.tree.view()
    
    @fcmlog
    def visit(self, name):
        """Switch current view of the data"""
        self.tree.visit(name)
        
    def current_node(self):
        """return the current node"""
        return self.tree.current
    
    def copy(self, npnts=None):
        #TODO rewrite so the tree is copied...
        """return a copy of fcm data object"""
        if npnts is None:
            tpnts = self.view().copy()
        else:
            tpnts = npnts
        tnotes = self.notes.copy()
        tchannels = self.channels[:]
        tmarkers = self.markers[:]
        return FCMdata(tpnts, tchannels, tmarkers, tnotes)
    
    @fcmlog
    def logicle(self, channels=None, T=262144, m=4.5*log(10), r=None, order=2, intervals=1000.0, scale_max=1e5, scale_min=0):
        """return logicle transformed channels"""
        if channels is None:
            channels = self.markers
        return _logicle(self, channels, T, m, r, order, intervals, scale_max, scale_min)
        
    @fcmlog
    def hyperlog(self, channels, b, d, r, order=2, intervals=1000.0):
        """return hyperlog transformed channels"""
        return _hyperlog(self, channels, b, d, r, order, intervals)
    
    @fcmlog
    def log(self, channels):
        """return log base 10 transformed channels"""
        return _log(self, channels)
    
    @fcmlog
    def gate(self, g, chan=None):
        """return gated region of fcm data"""
        return g.gate(self, chan)
    
    @fcmlog
    def subsample(self, s):
        """return subsampled/sliced fcm data"""
        return s.subsample(self)
    
    def get_cur_node(self):
        return self.tree.get()
    
    def add_view(self, node):
        """add a new node to the view tree"""        
        self.tree.add_child(node.name, node)
        return self

    def summary(self):
        """returns summary of current view"""
        pnts = self.view()
        means = pnts.mean(0)
        stds = pnts.std(0)
        mins = pnts.min(0)
        maxs = pnts.max(0)
        medians = median(pnts, 0)
        n, dim = pnts.shape
        summary = ''
        for i in range(dim):
            summary = summary + self.channels[i] + ":\n"
            summary = summary + "    max: " + str(maxs[i]) + "\n"
            summary = summary + "   mean: " + str(means[i]) + "\n"
            summary = summary + " median: " + str(medians[i]) + "\n"
            summary = summary + "    min: " + str(mins[i]) + "\n"
            summary = summary + "    std: " + str(stds[i]) + "\n"
        return summary
            
    def boundary_events(self):
        """returns dictionary of fraction of events in first and last
        channel for each channel"""
        boundary_dict = {}
        for k, chan in enumerate(self.channels):
            col = self.view()[:,k]
            boundary_dict[chan] = \
                sum((col==min(col)) | (col==max(col)))/len(col) 
        return boundary_dict
