from setuptools import setup, find_packages

setup(name='callat_ga_lib',
      version='1.0',
      description=u"Chiral-continuum extrapolation for gA",
      author=u"CalLat",
      author_email='chiachang@lbl.gov',
      url='https://github.com/callat-qcd/project_gA',
      license='',
      packages=['callat_ga_lib',],
      install_requires=[
          'lsqfit',
          'gvar'
          ]
      )
