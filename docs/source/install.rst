.. py:currentmodule:: fcm

Installing *fcm*
################

Without GPU support
*******************
*fcm* depends on the following libraries:

.. table:: Dependant Packages

    ==========  ==================================================================
    package     homepage
    ==========  ==================================================================
    numpy       http://numpy.scipy.org/
    scipy       http://scipy.scipy.org/
    matplotlib  http://matplotlib.sourceforge.net/
    dpmix       http://github.com/andrewcron/pycdp
    cython      http://cython.org/ (DPMix dependency)
    mip4py      http://mpi4py.scipy.org/ (optional DPMix dependency for threading)
    ==========  ==================================================================

With GPU support
****************
to enable gpu support using cuda for fitting mixture models from dpmix the following packages
need to be installed in addition to those above.  Once installed, no changes need 
be made, it will automatically begin using the gpu for fitting mixture models

.. table:: Additional Packages

    ============  ====================================================================================================
    package       homepage
    ============  ====================================================================================================
    pycuda        http://mathema.tician.de/software/pycuda
    gpustats      http://github.com/dukestats/gpustats
    ============  ====================================================================================================
    