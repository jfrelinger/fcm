.. toctree::
   :maxdepth: 2

.. _demo:

Demo Usage
==========

::

    from fcm import LoadFCS
    from fcm.statistics import DPMixtureModel
    from fcm.gui import pair_plot
    from pylab import scatter, show, xlabel, ylabel

    data = LoadFCS('/home/jolly/Projects/fcm/sample_data/3FITC_4PE_004.fcs')
    pair_plot(data, savefile='pair.png', display=False)

    model = DPMixtureModel(data, 16, 100, last=1)
    model.fit()
    m = model.get_results()
    d = m.classify(data)

    scatter(data['FSC-H'], data['SSC-H'], c=d)
    xlabel('FSC-H')
    ylabel('SSC-H')
    show()

