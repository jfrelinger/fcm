"""
Data structure for a collection of FCMData objects.
All operations will be performed on each FCMData in the collection.
"""

class FCMdict(dict):
    """
    Subclasses dict to represent collection of FCMdata objects.
    Attributes: 
    note = collection level annotations
    tree = tree of operations
    """

    def __init__(self, fcms=None, notes=None):
        """
        Initialize with fcm collection and notes.
        Tree of operations not implemented yet - how is this done in fcmdata?
        """
        if fcms is not None:
            for fcm in fcms:
                self[fcm.name] = fcm
        if notes is not None:
            self.notes =Annotation()
        else:
            self.notes = notes

if __name__ == '__main__':
    from io import loadFCS
    f1 = loadFCS('../../sample_data/3FITC_4PE_004.fcs')
    fs = FCMdict([f1])

    print fs.keys()
    print fs.values()

