.. py:currentmodule:: fcm

Installing *fcm*
################

Without GPU support
*******************
*fcm* depends on the following libraries:

.. table:: Dependant Packages

    ==========  ===================================================
    package     homepage
    ==========  ===================================================
    numpy       http://numpy.scipy.org/
    scipy       http://scipy.scipy.org/
    matplotlib  http://matplotlib.sourceforge.net/
    dpmix       http://github.com/andrewcron/pycdp
    munkres     http://github.com/jfrelinger/cython-munkres-wrapper
    mip4py      http://http://mpi4py.scipy.org/
    pymc        http://code.google.com/p/pymc/
    ==========  ===================================================

With GPU support
****************
to enable gpu support using cuda for fitting mixture models the following packages
need to be installed in addition to those above.  Once installed, no changes need 
be made to fcm, it will automatically begin using the gpu for fitting mixture models

.. table:: Additional Packages

    ============  ====================================================================================================
    package       homepage
    ============  ====================================================================================================
    pycuda        http://mathema.tician.de/software/pycuda
    gpustats      http://github.com/dukestats/gpustats
    scikits.cuda  http://lebedov.github.com/scikits.cuda/ (or without cula https://github.com/jfrelinger/scikits.cuda)
    ============  ====================================================================================================
    