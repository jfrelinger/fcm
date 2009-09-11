"""
Data structure for a collection of FCMData objects.
All operations will be performed on each FCMData in the collection.
"""

from UserDict import DictMixin
from annotation import Annotation
from enthought.traits.api import HasTraits, DictStrAny, Instance
import numpy
from util import fcmlog

class FCMcollection(DictMixin, HasTraits):
    """
    Represent collection of FCMdata objects.
    Attributes: 
    note = collection level annotations
    tree = tree of operations
    """

    fcmdict = DictStrAny()
    notes = Instance(Annotation)

    @fcmlog
    def __init__(self, name, fcms=None, notes=None):
        """
        Initialize with fcm collection and notes.
        Tree of operations not implemented yet - how is this done in fcmdata?
        """
        self.fcmdict = {}
        self.name = name
        if fcms is not None:
            for fcm in fcms:
                self.fcmdict[fcm.name] = fcm
        if notes is not None:
            self.notes =Annotation()
        else:
            self.notes = notes

    def keys(self):
        return self.fcmdict.keys()

    def __getitem__(self, item):
        """return fcmcollection.fcmdict[item]"""
        return self.fcmdict[item]

    @fcmlog
    def __setitem__(self, key, value):
        """set fcmcollection.fcmdict[key] = value."""
        self.fcmdict[key] = value

    def __delitem__(self, key):
        """delete fcmcollection.fcmdict[key]"""
        del self.fcmdict[key]

    def __getattr__(self, name):
        """Convenience function to access fcm object by name."""
        return self.fcmdict[name]

    def check_names(self):
        """Checks for channel name consistency. 

        Returns dictionary of (fcmcollecion.name, [bool] | dictionary) where 
        True = all fcm have same name
        for the channel and False = at least one different name.
        """
        result_dict = {}
        results = []
        channels_list = []
        for item in self.values():
            if isinstance(item, self.__class__):
                results.append(item.check_names())
            else:
                channels_list.append(item.channels)
        name_tuples = zip(*channels_list)
        for name_tuple in name_tuples[:]:
            bits = []
            for name in name_tuple[1:]:
                if name == name_tuple[0]:
                    bits.append(True)
                else:
                    bits.append(False)
            results.append(
                reduce(numpy.logical_and, bits))
        result_dict[self.name] = results
        return result_dict

if __name__ == '__main__':
    from io import loadFCS
    f1 = loadFCS('../../sample_data/3FITC_4PE_004.fcs')
    fs = FCMcollection([f1])

    print fs.keys()
    print fs.values()

