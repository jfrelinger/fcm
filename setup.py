from distutils.core import setup
setup(name='fcm',
      version='0.01',
      py_modules=['fcm'],
      package_dir = {'': 'src'},
      package_data= {'': 'data/*'},
      description='Python Flow Cytometry (FCM) Tools',
      author='Jacob Frelinger',
      author_email='jacob.frelinger@duke.edu',
      requires=['numpy (>=1.3.0)']
      )
