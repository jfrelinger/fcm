from distutils.core import setup, Extension
from numpy import get_include

mvnpdf_extension = Extension('fcm.statistics._mvnpdf',
                             sources = ['src/statistics/%s' % i for i in 
                                ['mvnpdf_ext/mvnpdf.cpp',
                                'mvnpdf_ext/mvnpdf_wrap.cpp',
                                'mvnpdf_ext/random/specialfunctions2.cpp',
                                'mvnpdf_ext/random/SpecialFunctions.cpp',
                                'mvnpdf_ext/matrix/newmat1.cpp',
                                'mvnpdf_ext/matrix/newmat2.cpp',
                                'mvnpdf_ext/matrix/newmat3.cpp',
                                'mvnpdf_ext/matrix/newmat4.cpp',
                                'mvnpdf_ext/matrix/newmat5.cpp',
                                'mvnpdf_ext/matrix/newmat6.cpp',
                                'mvnpdf_ext/matrix/newmat7.cpp',
                                'mvnpdf_ext/matrix/newmat8.cpp',
                                'mvnpdf_ext/matrix/newmat9.cpp',
                                'mvnpdf_ext/matrix/newmatex.cpp',
                                'mvnpdf_ext/matrix/myexcept.cpp',
                                'mvnpdf_ext/matrix/bandmat.cpp',
                                'mvnpdf_ext/matrix/submat.cpp',
                                'mvnpdf_ext/matrix/newmatrm.cpp',
                                'mvnpdf_ext/matrix/svd.cpp',
                                'mvnpdf_ext/matrix/sort.cpp',
                                'mvnpdf_ext/matrix/cholesky.cpp']],
                            include_dirs = [get_include(), 
                                    'src/statistics/mvnpdf_ext/matrix',
                                    'src/statistics/mvnpdf_ext/random',
                                    'src/statistics/mvnpdf_ext'],
                            #libraries = ['m', 'stdc++']
                            )
                                        

logicle_extension = Extension('fcm.core._logicle',
                              sources = [ 'src/core/logicle_ext/%s' %i for i in [
                                'Logicle.cpp',
                                'my_logicle.cpp',
                                'my_logicle_wrapper.cpp']],
                              include_dirs = [get_include()]
                              )
setup(name='fcm',
      version='0.8',
      url='http://code.google.com/p/py-fcm/',
      packages=['fcm', 'fcm.core', 'fcm.graphics', 'fcm.gui', 'fcm.io', 'fcm.statistics' ],
      package_dir = {'fcm': 'src'},
      package_data= {'': ['data/*']},
      description='Python Flow Cytometry (FCM) Tools',
      author='Jacob Frelinger',
      author_email='jacob.frelinger@duke.edu',
      ext_modules = [logicle_extension, mvnpdf_extension],
      requires=['numpy (>=1.3.0)',
                'scipy (>=0.6)',
		'pymc (>=2.1)',
		'dpmix (>=0.1)',
                'matplotlib (>=1.0)'], # figure out the rest of what's a required package.
      )
