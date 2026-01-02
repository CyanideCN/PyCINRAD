from setuptools import setup, find_packages
from setuptools.extension import Extension
from os.path import join, exists, sep

try:
    from Cython.Build import cythonize
    import numpy as np

    pyx_paths = [
        join("cinrad", "_utils.pyx"),
        join("cinrad", "correct", "_unwrap_2d.pyx"),
    ]
    cythonize_flag = True
    for _pyx in pyx_paths:
        if not exists(_pyx):
            cythonize_flag = False
            break
    if cythonize_flag:
        ext_modules = cythonize(pyx_paths)
    else:
        ext_modules = list()
        for _pyx in pyx_paths:
            name = _pyx.rstrip(".pyx").replace(sep, ".")
            source = _pyx.replace(".pyx", ".c")
            ext_modules.append(Extension(name, [source]))
    include_dirs = [np.get_include()]
except ImportError:
    ext_modules = None
    include_dirs = None

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
    include_dirs=include_dirs,
)
