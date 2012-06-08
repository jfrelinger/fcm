Advanced *fcm* tutorial
#######################

Graphics
********

Automated positivity thresholds
*******************************

Clustering
**********
The :py:mod:`fcm.statistics` module provides several models to automate 
cell subset identification.  The basic models are fit using k-means by :py:class:`fcm.statistics.KMeansModel`
and or a mixture of Gaussians by :py:class:`fcm.statistics.DPMixtureModel`.  Models are thought
of as a collection of model parameters that can be used to fit multiple datasets using their fit method.
fit methods then return a result object describing the estimated model fitting (means locations
for :py:class:`fcm.statistics.KMeansModel`, weights, means and covariances for :py:class:`fcm.statistics.DPMixtureModel`)


Clustering using K-Means
========================

.. code-block:: ipython
   
   In [1]: import fcm, fcm.statistics as stats
   
   In [2]: import pylab
   
   In [3]: data = fcm.loadFCS('/home/jolly/Projects/fcm/sample_data/3FITC_4PE_004.fcs')
   
   In [4]: kmmodel = stats.KMeansModel(10, niter=20, tol=1e-5)
   
   In [5]: results = kmmodel.fit(data)
   
   In [6]: c = results.classify(data)
   
   In [7]: pylab.subplot(1,2,1)
   Out[7]: <matplotlib.axes.AxesSubplot at 0x81b3ca0d0>
   
   In [8]: pylab.scatter(data[:,0], data[:,1], c=c, s=1, edgecolor='none')
   Out[8]: <matplotlib.collections.CircleCollection at 0x81b3eab90>
   
   In [9]: pylab.subplot(1,2,2)
   Out[9]: <matplotlib.axes.AxesSubplot at 0x81b3b3690>
   
   In [10]: pylab.scatter(data[:,2], data[:,3], c=c, s=1, edgecolor='none')
   Out[10]: <matplotlib.collections.CircleCollection at 0x827d0ee10>
   
   In [11]: pylab.savefig('kmeans.png')

produces

.. figure:: kmeans.png
   :align: center
   :height: 600px
   :alt: kmeans model fitting
   :figclass: align-center



Clustering with Mixture Models
==============================

Fitting the model using MCMC
============================

Fitting the model using BEM
===========================

Supervised learning
*******************

Subset identification with k-nearest neighbors
==============================================

Subset identification with SVM
==============================

Report generation
*****************

Building GUI frontends
**********************

Building web frontends
**********************
