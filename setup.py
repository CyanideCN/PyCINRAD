from setuptools import setup, find_packages
import os
from os.path import join
import glob

try:
    from Cython.Build import cythonize
    import numpy as np
    ext_modules = cythonize([join('cinrad', '_utils.pyx'), join('cinrad', 'correct', '_unwrap_2d.pyx')])
    include_dirs = [np.get_include()]
except ImportError:
    ext_modules = None
    include_dirs = None

data_pth = join('cinrad', 'data')

setup(
    name = 'cinrad',
    version = '1.3.2',
    description = 'Decode CINRAD radar data and visualize',
    long_description = 'Decode CINRAD radar data and visualize',
    license = 'GPL Licence',
    author = 'Puyuan Du',
    author_email = 'dpy274555447@gmail.com',
    packages = find_packages(),
    include_package_data = True,
    platforms = 'Windows',
    python_requires='>=3.5',
    install_requires = ['metpy>=0.8', 'cartopy>=0.15', 'pyshp!=2.0.0, !=2.0.1', 'pyresample', 'matplotlib>=2.2'],
    data_files = [(data_pth, [join(data_pth, 'RadarStation.npy'), join(data_pth, 'chinaCity.json')]),
                  (data_pth + os.path.sep + 'colormap', glob.glob(join(data_pth,'colormap', '*.cmap'))),
                  (data_pth + os.path.sep + 'shapefile', glob.glob(join(data_pth,'shapefile', '*')))],
    scripts = [],
    ext_modules = ext_modules,
    include_dirs = include_dirs,
    entry_points = {
        'console_scripts': [
            'test = test.help:main'
        ]
    }
)
