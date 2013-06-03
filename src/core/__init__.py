'''
Core objects and methods for working with flow cytometry data
'''

from fcm.core.fcmdata import FCMdata
from fcm.core.fcmcollection import FCMcollection
from fcm.core.annotation import Annotation
from fcm.core.fcmexceptions import BadFCMPointDataTypeError, UnimplementedFcsDataMode
from fcm.core.fcmexceptions import CompensationError
from fcm.core.transforms import logicle, hyperlog, productlog
from fcm.core.gate import PolyGate, points_in_poly, QuadGate, IntervalGate, ThresholdGate
from fcm.core.gate import generate_f_score_gate
from fcm.core.subsample import Subsample, SubsampleFactory, DropChannel
from fcm.core.compensate import load_compensate_matrix, compensate, gen_spill_matrix
