"""setup all things exported from FCM
"""

from fcmdata import FCMdata
from fcmexceptions import BadFCMPointDataTypeError

__all__ == [
            'FCMdata',
            'BadFCMPointDataTypeError',
            ]