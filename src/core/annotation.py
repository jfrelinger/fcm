"""FCM annotation and annotation sets for FCM data and files
"""

from enthought.traits.api import HasTraits, DictStrAny

class Annotation(HasTraits):
    """
    Annotation object for storing metadata about FCM data
    """
    annotations = DictStrAny()
    
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
        intended value
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
        intended value
        """
        return self._mydict[name]
    
    def __repr__(self):
        return 'Annotation('+self._mydict.__repr__()+')'
    
    def __getstate__(self):
        return self._mydict
    def __setstate(self, state):
        self._mydict = state
        
    def __getinitargs__(self):
        return (self._mydict,)
    
    def copy(self):
        return Annotation(self._mydict.copy())
    