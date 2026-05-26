from os.path import join, sep
from setuptools import find_packages, setup
from setuptools.extension import Extension

import numpy as np
from Cython.Build import cythonize

macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]

pyx_paths = [
    join("cinrad", "_utils"),
    join("cinrad", "correct", "_unwrap_2d"),
]

ext_suffix = ".pyx"

ext_modules = [
    Extension(
        path.replace(sep, "."),
        [path + ext_suffix],
        define_macros=macros,
    ) for path in pyx_paths
]

ext_modules = cythonize(ext_modules)

data_pth = join("cinrad", "data")

setup(
    name="cinrad",
    version="1.9.3",
    description="Decode CINRAD radar data and visualize",
    long_description="Decode CINRAD radar data and visualize",
    license="GPL Licence",
    author="PyCINRAD Developers",
    author_email="dpy274555447@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    platforms=["Windows", "Linux", "MacOS"],
    python_requires=">=3.9",
    install_requires=[
        "metpy>=0.8",
        "cartopy>=0.15",
        "pyshp!=2.0.0, !=2.0.1",
        "matplotlib>=2.2",
        "vanadis",
        "cinrad_data>=0.1"
    ],
    package_dir={"cinrad": "cinrad"},
    package_data={"cinrad": [
        "data/*.*",
        "data/*/*.*"
    ]},
    scripts=[],
    ext_modules=ext_modules,
    include_dirs=[np.get_include()],
)
