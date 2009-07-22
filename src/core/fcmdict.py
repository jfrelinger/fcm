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

    def __init__(self, fcms=None):
        if fcms is not None:
            for fcm in fcms:
                self[fcm.name] = fcm

class FCM(object):
    def __init__(self, name):
        self.name = name

if __name__ == '__main__':
    f1 = FCM('f1')
    f2 = FCM('f2')
    fs = FCMdict([f1, f2])

    print fs['f1']
    print fs.keys()
    print fs.values()

