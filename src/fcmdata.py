"""
A python object representing flow cytomoetry data
"""

class FCMdata(object):
    """
    Object representing flow cytometry data
    FCMdata.pnts : a numpy array of data points
    FCMdata.channels : a list of which markers/scatters are on which column of
                         the array.
    FCMdata.scatters : a list of which indexes in fcmdata.channels are scatters

    """
    
    
    def __init__(self, pnts, channels, scatters=None):
        """
        fcmdata(pnts, channels, scatters=None)
        pnts: array of data points
        channels: a list of which markers/scatters are on which column of
                    the array.
        scatters: a list of which indexes in channels are scatters
        
        """
        self.pnts = pnts
        self.channels = channels
        #TODO add some default intelegence for determining scatters if None
        self.scatters = scatters
        self.markers = []
        for chan in range(len(channels)):
            if chan not in self.scatters:
                self.markers.append(chan)

        
    def get_channel_by_name(self, channels):
        """Return the data associated with specific channel names"""
        if type(channels) == type(''):
            channels = [channels]
        to_return = [ self.channels.index(i) for i in channels]
        return self.pnts[:, to_return]
    
    def get_markers(self):
        """return the data associated with all the markers"""
        
        return self.pnts[:, self.markers]
        
        