from setuptools import setup, find_packages
import os
import glob

setup(
    name = 'cinrad',
    version = '1.2',
    description = 'cinrad reading and plotting',
    long_description = 'cinrad reading and plotting',
    license = 'GPL Licence',
    author = 'Puyuan Du',
    author_email = 'dpy274555447@gmail.com',
    packages = find_packages(),
    include_package_data = True,
    platforms = 'Windows',
    python_requires='>=3.5',
    install_requires = ['metpy>=0.8', 'cartopy>=0.15', 'pyshp>=1.2'],
    data_files=[('cinrad', ['RadarStation.npy']),
                    ('cinrad' + os.path.sep + 'colormap', glob.glob(r'colormap/*.txt')),
                    ('cinrad' + os.path.sep + 'shapefile', glob.glob(r'shapefile/*'))],
    scripts = [],
    entry_points = {
        'console_scripts': [
            'test = test.help:main'
        ]
    }
)
