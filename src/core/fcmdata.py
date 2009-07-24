"""
A python object representing flow cytomoetry data
"""
from numpy import array
from annotation import Annotation
from fcmexceptions import BadFCMPointDataTypeError
from transforms import logicle as _logicle
from transforms import hyperlog as _hyperlog
from util import Tree, RootNode

class FCMdata(object):
    """
    Object representing flow cytometry data
    FCMdata.pnts : a numpy array of data points
    FCMdata.channels : a list of which markers/scatters are on which column of
                         the array.
    FCMdata.scatters : a list of which indexes in fcmdata.channels are scatters

    """
    
    
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
        if type(pnts) != type(array([])):
            raise BadFCMPointDataTypeError(pnts, "pnts isn't a numpy.array")
        self.tree = Tree(pnts)
        #self.pnts = pnts
        self.channels = channels
        #TODO add some default intelegence for determining scatters if None
        self.scatters = scatters
        self.markers = []
        if scatters is not None:
            for chan in range(len(channels)):
                if chan not in self.scatters:
                    self.markers.append(chan)
        if notes == None:
            notes = Annotation()
        self.notes = notes
        
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
        
        return self.pnts[:, self.markers]
    
    def __getitem__(self, item):
        """return FCMdata.pnts[i] by name or by index"""
        
        if type(item) == type(''):
            return self.get_channel_by_name(item)
        elif type(item) == tuple:
            if type(item[0]) == type(''):
                return self.get_channel_by_name(list(item))
            else:
                return self.tree.view()[item]
        else:
            return self.tree.view()[item]
        
    def view(self):
        return self.tree.view()
    
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
    
    def logicle(self, channels, T, m, r, order=2, intervals=1000.0):
        """return logicle transformed channels"""
        return _logicle(self, channels, T, m, r, order, intervals)
        
    def hyperlog(self, channels, b, d, r, order=2, intervals=1000.0):
        """return hyperlog transformed channels"""
        return _hyperlog(self, channels, b, d, r, order, intervals)
    
    def gate(self, g, chan=None):
        """return gated region of fcm data"""
        return g.gate(self, chan)
    
    def get_cur_node(self):
        return self.tree.get()
    
    def add_view(self, node):
        """add a new node to the view tree"""
        self.tree.add_child(node.name, node)
        return self
    
    def __getattr__(self, name):
        return self.tree.view().__getattribute__(name)

        
