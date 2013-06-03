"""setup all things exported from FCM
"""

from fcm.core import FCMdata, FCMcollection
from fcm.core import Annotation
from fcm.core import PolyGate, points_in_poly, QuadGate, IntervalGate, ThresholdGate
from fcm.core import generate_f_score_gate
from fcm.core import BadFCMPointDataTypeError, UnimplementedFcsDataMode
from fcm.core import CompensationError
from fcm.core import load_compensate_matrix, compensate, gen_spill_matrix
from fcm.io import FCSreader, loadFCS, loadMultipleFCS, FlowjoWorkspace, load_flowjo_xml, export_fcs
from fcm.core import Subsample, SubsampleFactory, DropChannel
from fcm.core  import logicle, hyperlog

__all__ = [
            #Objects
            'FCMdata',
            'FCMcollection',
            'PolyGate',
            'QuadGate',
            'IntervalGate',
            'ThresholdGate',
            'FCSreader',
            'Annotation',
            'FlowjoWorkspace',
            #Exceptions
            'BadFCMPointDataTypeError',
            'UnimplementedFcsDataMode',
            'CompensationError',
            #functions
            'generate_f_score_gate',
            'logicle',
            'hyperlog',
            'loadFCS',
            'loadMultipleFCS',
            'load_compensate_matrix',
            'load_flowjo_xml',
            ]
