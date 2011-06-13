import numpy
from matplotlib.nxutils import points_inside_poly
from util import GatingNode


class Filter(object):
    """An object representing a gatable region"""

    def __init__(self, vert, channels):
        """
        vert = vertices of gating region
        channels = indices of channels to gate on.
        """
        self.vert = vert
        self.chan = channels

    def gate(self, fcm, chan=None):
        """do the actual gating here."""
        pass

class PolyGate(Filter):
    """An object representing a polygonal gatable region"""

    def gate(self, fcm, chan=None, invert=False):
        """
        return gated region of FCM data
        """
        if chan is None:
            chan = self.chan
        #idxs = points_in_poly(self.vert, fcm.view()[:, chan])

        # matplotlib has points in poly routine in C
        # no faster than our numpy version
        idxs = points_inside_poly(fcm.view()[:, chan], self.vert)

        if invert:
            idxs = numpy.invert(idxs)

        node = GatingNode("", fcm.get_cur_node(), idxs)
        fcm.add_view(node)
        return fcm

class QuadGate(Filter):
    """
    An object to divide a region to four quadrants
    """
    def gate(self, fcm, chan=None):
        """
        return gated region
        """
        if chan is None:
            chan = self.chan
        # I (+,+), II (-,+), III (-,-), and IV (+,-)
        x = fcm.view()[:, chan[0]]
        y = fcm.view()[:, chan[1]]
        quad = {}
        quad[1] = (x > self.vert[0]) & (y > self.vert[1]) # (+,+)
        quad[2] = (x < self.vert[0]) & (y > self.vert[1]) # (-,+)
        quad[3] = (x < self.vert[0]) & (y < self.vert[1]) # (-,-)
        quad[4] = (x > self.vert[0]) & (y < self.vert[1]) # (+,-)
        root = fcm.get_cur_node()
        name = root.name
        for i in quad.keys():
            if True in quad[i]:
                fcm.tree.visit(name)
                node = GatingNode("q%d" % i, root, quad[i])
                fcm.add_view(node)
        return fcm

class IntervalGate(Filter):
    """
    An objeect to return events within an interval in any one channel.
    """
    def gate(self, fcm, chan=None):
        """
        return interval region.
        """
        if chan is None:
            chan = self.chan

        assert(len(self.chan) == 1)
        assert(len(self.vert) == 2)
        assert(self.vert[1] >= self.vert[0])

        x = fcm.view()[:, chan[0]]
        idxs = numpy.logical_and(x > self.vert[0], x < self.vert[1])

        node = GatingNode("", fcm.get_cur_node(), idxs)
        fcm.add_view(node)
        return fcm

class ThresholdGate(Filter):
    """
    an object to return events above or below a threshold in any one channel
    """
    def __init__(self, vert, channels, op = 'g'):
        """
        vert = boundry region
        channels = indices of channel to gate on.
        op = 'g' (greater) or 'l' (less) 
        """
        self.vert = vert
        self.chan = channels
        self.op = op
        
        
    def gate(self, fcm, chan=None, op=None):
        """
        return all events greater (or less) than a threshold
        allowed op are 'g' (greater) or 'l' (less)
        """
        if chan is None:
            chan = self.chan
            
        x = fcm.view()[:, chan]
        if op is None:
            op = self.op
            
        if op == 'g':
            idxs = numpy.greater(x,self.vert)
        elif op == 'l':
            idxs = numpy.less(x,self.vert)
        else:
            raise ValueError('op should be "g" or "l", received "%s"' % str(op))
            
        node = GatingNode("", fcm.get_cur_node(), idxs)
        fcm.add_view(node)
        return fcm

def points_in_poly(vs, ps):
    """Return boolean index of events from ps that are inside polygon with vertices vs.

    vs = numpy.array((k, 2))
    ps = numpy.array((n, 2))
    """

    # optimization to check only events within bounding box
    # for polygonal gate - useful if gating region is small
    # area_ratio_threshold = 0.5
    # area_gate_bb = numpy.prod(numpy.max(vs, 0) - numpy.min(vs, 0))
    # area_gate_ps = numpy.prod(numpy.max(ps, 0) - numpy.min(ps, 0))
    # if area_gate_bb/area_gate_ps < area_ratio_threshold:
    #     idx = numpy.prod((ps > numpy.min(vs, 0)) & (ps < numpy.max(vs, 0)),1)
    #     ps = ps[idx.astype('bool'), :]

    j = len(vs) - 1
    inPoly = numpy.zeros((len(vs), len(ps)), 'bool')

    for i, v in enumerate(vs):
        inPoly[i, :] |= ((v[0] < ps[:, 0]) & (vs[j, 0] >= ps[:, 0])) | ((vs[j, 0] < ps[:, 0]) & (v[0] >= ps[:, 0]))
        inPoly[i, :] &= (v[1] + (ps[:, 0] - v[0]) / (vs[j, 0] - v[0]) * (vs[j, 1] - v[1]) < ps[:, 1])
        j = i

    return numpy.bitwise_or.reduce(inPoly, 0)

if __name__ == '__main__':
    vertices = numpy.array([[2, 2], [10, 2], [10, 10], [2, 10]], 'd')
    points = numpy.random.uniform(0, 10, (10000000, 2))

    import time
    start = time.clock()
    inside = points_in_poly(vertices, points)
    print "Time elapsed: ", time.clock() - start
    print numpy.sum(inside)

