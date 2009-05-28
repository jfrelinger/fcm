"""FCM annotation and annotation sets for FCM data and files
"""

class Annotation(object):
    """
    Annotation object for storing metadata about FCM data
    """
    def __init__(self, annotations=None):
        """
        Annotation([annotations = {}])
        """
        if annotations == None:
            annotations = {}
        
        self.__dict__['_mydict'] = annotations
        
    def __getattr__(self, name):
        """
        allow usage of annotation.foo or annotation[foo] to return the
        intendede value
        """
        if name in self._mydict.keys():
            self.__dict__[name] = self._mydict[name]
            return self._mydict[name]
        else:
            return getattr(self._mydict, name)
        
        
    def __setattr__(self, name, value):
        """
        allow usage of annotation.foo  = x or annotation[foo] =x to set the
        intendede value
        """
        #return setattr(self._mydict, name, value)
        self._mydict[name] = value
        self.__dict__[name] = self._mydict[name]
        
    def __setitem__(self, name, value):
        """
        allow usage of annotation.foo  = x or annotation[foo] =x to set the
        intendede value
        """
        self._mydict[name] = value
        self.__dict__[name] = self._mydict[name]

    def __getitem__(self, name):
        """
        allow usage of annotation.foo or annotation[foo] to return the
        intendede value
        """
        return self._mydict[name]
    
    