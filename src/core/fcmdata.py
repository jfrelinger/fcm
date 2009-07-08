"""
A python object representing flow cytomoetry data
"""
from numpy import array
from fcmannotation import Annotation
from fcmexceptions import BadFCMPointDataTypeError

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
        for chan in range(len(channels)):
            if chan not in self.scatters:
                self.markers.append(chan)
        if annotations == None:
            annotations = Annotation()
        self.annotation = annotations
        
    def get_channel_by_name(self, channels):
        """Return the data associated with specific channel names"""
        if type(channels) == type(''):
            channels = [channels]
        to_return = [ self.channels.index(i) for i in channels]
        return self.pnts[:, to_return]
    
    def get_markers(self):
        """return the data associated with all the markers"""
        
        return self.pnts[:, self.markers]
    
    def __getitem__(self, item):
        if type(item) == type(''):
            return self.get_channel_by_name(item)
        elif type(item) == tuple:
            if type(item[0]) == type(''):
                return self.get_channel_by_name(list(item))
            else:
                return self.pnts[item]
        else:
            return self.pnts[item]
        
        