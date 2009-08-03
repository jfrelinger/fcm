from distutils.core import setup
setup(name='fcm',
      version='0.01',
      packages=['fcm', 'fcm.core', 'fcm.graphics', 'fcm.gui', 'fcm.io', 'fcm.statistics' ],
      package_dir = {'fcm': 'src'},
      package_data= {'': ['data/*']},
      description='Python Flow Cytometry (FCM) Tools',
      author='Jacob Frelinger',
      author_email='jacob.frelinger@duke.edu',
      requires=['numpy (>=1.3.0)'], # figure out the rest of what's a required package.
      )
