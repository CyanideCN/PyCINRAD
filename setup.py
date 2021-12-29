from setuptools import setup, find_packages
from setuptools.extension import Extension
from os.path import join, exists, sep
import glob

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
    version="1.7.0",
    description="Decode CINRAD radar data and visualize",
    long_description="Decode CINRAD radar data and visualize",
    license="GPL Licence",
    author="Puyuan Du",
    author_email="dpy274555447@gmail.com",
    packages=find_packages(),
    include_package_data=True,
    platforms="Windows",
    python_requires=">=3.5",
    install_requires=[
        "metpy>=0.8",
        "cartopy>=0.15",
        "pyshp!=2.0.0, !=2.0.1",
        "matplotlib>=2.2",
        "vanadis",
    ],
    data_files=[
        (
            data_pth,
            glob.glob(join(data_pth, "*.*")),
        ),
        (join(data_pth, "colormap"), glob.glob(join(data_pth, "colormap", "*.cmap"))),
        (join(data_pth, "shapefile"), glob.glob(join(data_pth, "shapefile", "*"))),
        (join(data_pth, "font"), glob.glob(join(data_pth, "font", "*"))),
        (join("cinrad", "correct"), [join("cinrad", "correct", "unwrap_2d_ljmu.c")]),
    ],
    scripts=[],
    ext_modules=ext_modules,
    include_dirs=include_dirs,
)
