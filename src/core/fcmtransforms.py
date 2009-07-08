import numpy

def productlog(x, prec=1e-12):
    """Productlog or LambertW function computes principal solution for
 w in f(w) = w*exp(w).
     
    x is a numpy array
""" 
    idx = x <= 500
    x1 = x[idx]
    x2 = x[~idx]
    lxl = numpy.log(x1 + 1.0)
    y1 = 0.665 * (1+0.0195*lxl) * lxl + 0.04
    y2 =  numpy.log(x2 - 4.0) - (1.0 - 1.0/numpy.log(x2)) * numpy.log(numpy.log(x2))
    y = numpy.zeros(len(x))
    y[idx] = y1
    y[~idx] = y2
    return y

if __name__ == '__main__':
    print productlog(numpy.array([0.0, 1.0, 10.0, 100.0, 1000.0, 10000.0]))

