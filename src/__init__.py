"""setup all things exported from FCM
"""

from core import FCMdata, FCMcollection
from core import Annotation
from core import PolyGate, points_in_poly, QuadGate, IntervalGate, ThresholdGate
from core import BadFCMPointDataTypeError, UnimplementedFcsDataMode
from core import CompensationError
from core import load_compensate_matrix, compensate, gen_spill_matrix
from io import FCSreader, loadFCS, loadMultipleFCS
from core import Subsample, SubsampleFactory, DropChannel
from core  import logicle, hyperlog

__all__ = [
            #Objects
            'FCMdata',
            'Gate',
            'QuadGate',
            'FCSreader',
            'Annotation',
            #Exceptions
            'BadFCMPointDataTypeError',
            'UnimplementedFcsDataMode',
            'CompensationError',
            #functions
            'logicle',
            'hyperlog',
            'loadFCS'
            ]
