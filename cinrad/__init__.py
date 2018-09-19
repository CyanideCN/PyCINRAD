from . import io
from . import grid
from . import visualize
from . import utils
from . import easycalc

__version__ = '1.0'

def set_savepath(path):
    import os
    os.environ['CINRAD_PATH'] = path