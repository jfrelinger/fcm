from numpy import array, reshape, max, dot
from numpy.linalg import solve, inv
from fcmexceptions import CompensationError
from util import TransformNode

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

def compensate(fcm, S=None, markers=None, comp=False, scale=False):
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
            
    if scale and not comp:
        S = S/max(S)
    if not comp:
        S = inv(S)
    idx = fcm.name_to_index(markers)
    
    c = dot(fcm.view()[idx], S)
    new = fcm.view()[:]
    new[idx] = c
    node = TransformNode('Compensated', fcm.get_cur_node, new)
    fcm.add_view(node)
    return new

