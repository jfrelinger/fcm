from numpy import array, reshape
from numpy.linalg import solve
from core.fcmexceptions import CompensationError

def get_spill(text):
    """Extracts spillover matrix from FCS text entry.

    Returns (spillover matrix S, column headers)
    """
    spill = text.split(',')
    n = int(spill[0])
    markers = spill[1:(n+1)]
    markers = [item.strip().replace('\n', '') for item in markers]
    items = [item.strip().replace('\n','') for item in spill[n+1:]]
    S = reshape(map(float, items), (n, n))        
    return S, markers

def compensate(fcm, S=None, markers=None):
    """Compensate data given spillover matrix S and markers to compensate
    If S, markers is not given, will look for fcm.annotate.text['SPILL']
    """
    if S is None and markers is not None:
        msg = 'Attempted compnesation on markers without spillover matrix'
        raise CompensationError(msg)
    if S is None:
        S, m = get_spill(fcm.annotate.text['SPILL'])
        if markers is None:
            markers = m
    idx = fcm.name_to_index(markers)
    return solve(S.T, fcm.pnts[idx])

