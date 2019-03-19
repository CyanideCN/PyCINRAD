from setuptools import setup, find_packages
import os
import glob

try:
    from Cython.Build import cythonize
    import numpy as np
    ext_modules = cythonize(os.path.join('cinrad', '_utils.pyx'))
    include_dirs = [np.get_include()]
except ImportError:
    ext_modules = None
    include_dirs = None

setup(
    name = 'cinrad',
    version = '1.3',
    description = 'cinrad reading and plotting',
    long_description = 'cinrad reading and plotting',
    license = 'GPL Licence',
    author = 'Puyuan Du',
    author_email = 'dpy274555447@gmail.com',
    packages = find_packages(),
    include_package_data = True,
    platforms = 'Windows',
    python_requires='>=3.5',
    install_requires = ['metpy>=0.8', 'cartopy>=0.15', 'pyshp!=2.0.0, !=2.0.1', 'pyresample', 'matplotlib>=2.2'],
    data_files = [('cinrad', ['RadarStation.npy', 'chinaCity.json']),
                  ('cinrad' + os.path.sep + 'colormap', glob.glob(r'colormap/*.cmap')),
                  ('cinrad' + os.path.sep + 'shapefile', glob.glob(r'shapefile/*'))],
    scripts = [],
    ext_modules = ext_modules,
    include_dirs = include_dirs,
    entry_points = {
        'console_scripts': [
            'test = test.help:main'
        ]
    }
)
