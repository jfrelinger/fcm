# draggable rectangle with the animation blit techniques; see
# http://www.scipy.org/Cookbook/Matplotlib/Animations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

class DraggableVertex(object):
    lock = None  # only one can be animated at a time
    def __init__(self, circle, parent):
        self.parent = parent
        self.circle = circle
        self.press = None
        self.background = None

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.circle.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.circle.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.circle.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)

    def on_press(self, event):
        'on button press we will see if the mouse is over us and store some data'
        if event.inaxes != self.circle.axes: return
        if DraggableVertex.lock is not None: return
        contains, attrd = self.circle.contains(event)
        if not contains: return
        print 'event contains', self.circle.center
        x0, y0 = self.circle.center
        self.press = x0, y0, event.xdata, event.ydata
        DraggableVertex.lock = self

        # draw everything but the selected circle and store the pixel buffer
        canvas = self.circle.figure.canvas
        axes = self.circle.axes
        self.circle.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.circle.axes.bbox)

        # now redraw just the circle
        axes.draw_artist(self.circle)

        # and blit just the redrawn area
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        'on motion we will move the circle if the mouse is over us'
        if DraggableVertex.lock is not self:
            return
        if event.inaxes != self.circle.axes: return
        x0, y0, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.circle.center = (x0+dx, y0+dy)

        canvas = self.circle.figure.canvas
        axes = self.circle.axes
        # restore the background region
        canvas.restore_region(self.background)

        # redraw just the current circle
        axes.draw_artist(self.circle)

        # blit just the redrawn area
        canvas.blit(axes.bbox)

        # update parent
        self.parent.update()

    def on_release(self, event):
        'on release we reset the press data'
        if DraggableVertex.lock is not self:
            return

        self.press = None
        DraggableVertex.lock = None

        # turn off the circle animation property and reset the background
        self.circle.set_animated(False)
        self.background = None

        # redraw the full figure
        self.circle.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.circle.figure.canvas.mpl_disconnect(self.cidpress)
        self.circle.figure.canvas.mpl_disconnect(self.cidrelease)
        self.circle.figure.canvas.mpl_disconnect(self.cidmotion)

class Gate(object):
    def __init__(self, fig, ax):
        self.fig = fig
        self.ax = ax
        self.vertices = []
        self.poly = None
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def add_vertex(self, vertex):
        print vertex.center
        self.ax.add_patch(vertex)
        dv = DraggableVertex(vertex, self)
        dv.connect()
        self.vertices.append(dv)  
        self.update()
        self.fig.canvas.draw()
        
    def update(self):
        if len(self.vertices) >= 3:
            xy = np.array([v.circle.center for v in self.vertices])
            if self.poly is None:
                self.poly = Polygon(xy, closed=True, alpha=0.1, facecolor='pink')
                self.ax.add_patch(self.poly)
            else:
                self.poly.set_xy(xy)
            self.ax.draw_artist(self.poly)        

    def onclick(self, event):
        if event.button == 3:
            vertex = Circle((event.xdata, event.ydata), radius=0.01)
            self.add_vertex(vertex)

fig = plt.figure()
ax = fig.add_subplot(111)
gate = Gate(fig, ax)

plt.show()
