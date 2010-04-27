import numpy
import pylab

# nice color scheme but discrete only for 8 values
# cs = numpy.array([(178,24,43), (214,96,77), (244,165,130), 
#                   (253,219,199), (209,229,240), (146,197,222), 
#                   (67,147,195), (33,102,172)], 'd')/256.0

def dimechart(indicators, values, labels, cmap):
    indicators = numpy.array(indicators)
    values = numpy.array(values)
    k, p = values.shape
    w, h = 1, 10
    cs = cmap(numpy.linspace(0, 1, p))
    
    pylab.figure(figsize=(p, 3*k))

    # keys
    pylab.subplot(k+1, 1, 1)
    for _p in range(p):
        xpts = 10*numpy.array([_p+1,_p+2,_p+2,_p+1])
        ypts = 10*numpy.array([0,0,h,h])
        pylab.fill(xpts, ypts, color=cs[_p], alpha=0.5, closed=True,
                   ec='k')
        pylab.text(10*(_p+1.5), 75, labels[_p], va='center', ha='center',
                   fontsize=16, rotation=90)

    # data
    for _k in range(k):
        pylab.subplot(k+1, 1, _k+2)
        for _p in range(p):
            make_spark(indicators[_k, _p],
                       (2+p)*_p,
                       (2+k)*10*values[_k,_p],
                       w, h, cmap)

    pylab.show()

def make_spark(xs, xoffset=0, yoffset=0, w=1, h=10,
               cmap=pylab.cm.gist_rainbow):
    # make central line
    n = len(xs)
    dw = (w-1)/2.0
    pylab.plot([xoffset+1-dw,xoffset+1+n+dw],[yoffset,yoffset],'k-')
    cs = cmap(numpy.linspace(0, 1, n))
    for k, x in enumerate(xs):
        if x: # up
            xpts = numpy.array([k+1-dw,k+2+dw,k+2+dw,k+1-dw]) + xoffset
            ypts = numpy.array([0,0,h,h]) + yoffset
            pylab.fill(xpts, ypts, color=cs[k], alpha=0.5, closed=True,
                       ec='k')
        else:
            xpts = numpy.array([k+1-dw,k+2+dw,k+2+dw,k+1-dw]) + xoffset
            ypts = numpy.array([0,0,-h,-h]) + yoffset
            pylab.fill(xpts, ypts, color=cs[k], alpha=0.5, closed=True,
                       ec='k')

if __name__ == '__main__':
    i = numpy.random.randint(0,2,(2,10,10))
    d = numpy.random.uniform(0,1,(2,10))
    l = map(str, range(10))
    print d
    dimechart(i, d, l, pylab.cm.gist_rainbow)
    # xoffset = numpy.zeros(10)
    # make_spark(i, xoffset, d)
    pylab.show()
    
