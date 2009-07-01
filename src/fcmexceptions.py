"""
classes for exceptions in FCM module
"""
#from exceptions import Exception

class BadFCMPointDataTypeError(Exception):
    """Exception raised on bad FCM data type
    
    data: the wrong data
    message: explanation of the error
    """
    
    def __init__(self, data, message):
        self.type = type(data)
        self.data = data
        self.message = message
        
class UnimplementedFcsDataMode(Exception):
    """Exception raised on unimplemented data modes in fcs files
    
    mode: mode
    """
    
    def __init__(self, mode):
        self.mode = mode
        self.message = "Currently fcs data stored as type \'%s\' is unsupported" % mode
        