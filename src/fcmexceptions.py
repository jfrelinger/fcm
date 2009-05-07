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
        