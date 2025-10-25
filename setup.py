from setuptools import setup, find_packages
from setuptools.extension import Extension
from os.path import join, exists, sep
import re

def fix_cython_c_file(filepath):
    """
    Fix Cython-generated C files for MinGW compatibility.
    
    Cython generates a compile-time check using enum that is not compatible
    with strict C compilers like MinGW GCC. This function removes the
    problematic check to allow compilation.
    """
    if not exists(filepath):
        return
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Fix SIZEOF_VOID_P check that causes compilation error with MinGW
        # The generated code: enum { __pyx_check_sizeof_voidp = 1 / (int)(SIZEOF_VOID_P == sizeof(void*)) };
        # causes "error: enumerator value is not an integer constant" in MinGW GCC
        pattern = r'enum\s*\{\s*__pyx_check_sizeof_voidp\s*=\s*1\s*/\s*\(int\)\(SIZEOF_VOID_P\s*==\s*sizeof\(void\*\)\)\s*\};'
        if re.search(pattern, content):
            content = re.sub(
                pattern,
                '/* __pyx_check_sizeof_voidp check disabled for MinGW compatibility */',
                content
            )
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"Fixed MinGW compatibility issue in {filepath}")
    except Exception as e:
        print(f"Warning: Failed to fix {filepath}: {e}")

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
        extensions = []
        for _pyx in pyx_paths:
            name = _pyx.rstrip(".pyx").replace(sep, ".")
            ext = Extension(
                name,
                [_pyx],
                include_dirs=[np.get_include()],
                define_macros=[
                    ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION'),
                ]
            )
            extensions.append(ext)
        ext_modules = cythonize(
            extensions,
            compiler_directives={
                'language_level': '3',
            }
        )
        
        # 修复生成的 C 文件
        for _pyx in pyx_paths:
            c_file = _pyx.replace(".pyx", ".c")
            fix_cython_c_file(c_file)
    else:
        ext_modules = list()
        for _pyx in pyx_paths:
            name = _pyx.rstrip(".pyx").replace(sep, ".")
            source = _pyx.replace(".pyx", ".c")
            ext = Extension(
                name,
                [source],
                include_dirs=[np.get_include()],
                define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
            )
            ext_modules.append(ext)
    include_dirs = [np.get_include()]
except ImportError:
    ext_modules = None
    include_dirs = None

data_pth = join("cinrad", "data")

setup(
    name="cinrad",
    version="1.9.2",
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
        "numpy>=1.20", 
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
