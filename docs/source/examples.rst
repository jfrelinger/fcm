Examples using fcm
==================

Loading fcs data
----------------

fcm provides the loadFCS function to load fcs files:

.. code-block:: python

    >>> from fcm import loadFCS
    >>> data = loadFCS('../sample_data/3FITC_4PE_004.fcs')
    >>> data
    3FITC_4PE_004
    >>> data.channels
    ['FSC-H', 'SSC-H', 'FL1-H', 'FL2-H']
    >>> data.shape
    (94569, 4)


Since the FCMData object returned by loadFCS delegates to underlying numpy
array, you can pass the FCMData object to most numpy functions

.. code-block:: python

    >>> import numpy as np
    >>> np.mean(data)
    410.38791252947584
    >>> np.mean(data,0)
    array([ 538.76464803,  421.57733507,  340.03599488,  341.17367213])
    >>> import pylab
    >>> pylab.scatter(data[:,0],data[:,1], s=1, edgecolors='none')
    >>> pylab.show()
    

.. plot::

    import matplotlib.pyplot as plt
    from fcm import loadFCS
    data = loadFCS('../sample_data/3FITC_4PE_004.fcs')
    plt.scatter(data[:,0],data[:,1], s=1, edgecolors='none')
    plt.show()

foo
bar

