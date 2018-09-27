from . import io
from . import grid
from . import utils
from . import easycalc

__version__ = '1.1'

def set_savepath(path):
    import os
    os.environ['CINRAD_PATH'] = path
