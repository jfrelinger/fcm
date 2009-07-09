"""
A python object representing flow cytomoetry data
"""
from numpy import array
from fcmannotation import Annotation
from fcmexceptions import BadFCMPointDataTypeError
from fcmtransforms import logicle as _logicle
from fcmtransforms import hyperlog as _hyperlog
from fcmgate import Gate

class FCMdata(object):
    """
    Object representing flow cytometry data
    FCMdata.pnts : a numpy array of data points
    FCMdata.channels : a list of which markers/scatters are on which column of
                         the array.
    FCMdata.scatters : a list of which indexes in fcmdata.channels are scatters

    """
    
    
    def __init__(self, pnts, channels, scatters=None, annotations=None):
        """
        fcmdata(pnts, channels, scatters=None)
        pnts: array of data points
        channels: a list of which markers/scatters are on which column of
                    the array.
        scatters: a list of which indexes in channels are scatters
        
        """
        if type(pnts) != type(array([])):
            raise BadFCMPointDataTypeError(pnts, "pnts isn't a numpy.array")
        self.pnts = pnts
        self.channels = channels
        #TODO add some default intelegence for determining scatters if None
        self.scatters = scatters
        self.markers = []
        if scatters is not None:
            for chan in range(len(channels)):
                if chan not in self.scatters:
                    self.markers.append(chan)
        if annotations == None:
            annotations = Annotation()
        self.annotation = annotations
        
    def name_to_index(self, channels):
        """Return the channel indexes for the named channels"""
        
        if type(channels) == type(''):
            channels = [channels]
        to_return = [ self.channels.index(i) for i in channels]
        return to_return
    
    def get_channel_by_name(self, channels):
        """Return the data associated with specific channel names"""
        
        return self.pnts[:, self.name_to_index(channels)]
    
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
                return self.pnts[item]
        else:
            return self.pnts[item]
        
    def copy(self, npnts=None):
        """return a copy of fcm data object"""
        if npnts is None:
            tpnts = self.pnts.copy()
        else:
            tpnts = npnts
        tanno = self.annotation.copy()
        tchannels = self.channels[:]
        tmarkers = self.markers[:]
        return FCMdata(tpnts, tchannels, tmarkers, tanno)
    
    def logicle(self, channels, T, m, r, order=2, intervals=1000.0):
        """return logicle transformed channels"""
        return _logicle(self, channels, T, m, r, order=2, intervals=1000.0)
        
    def hyperlog(self, channels, b, d, r, order=2, intervals=1000.0):
        """return hyperlog transformed channels"""
        return _hyperlog(self, channels, b, d, r, order=2, intervals=1000.0)
    
    def gate(self, g, chan=None):
        """return gated region of fcm data"""
        return g.gate(self, chan)
    
    def __getattr__(self, name):
        return self.pnts.__getattribute__(name)

        