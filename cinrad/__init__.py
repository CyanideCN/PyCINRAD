from . import io
from . import grid
from . import utils
from . import calc
from . import visualize
from . import correct

from .deprecation import Deprecated

__version__ = "1.6.0"

# deprecate `easycalc` namespace
easycalc = Deprecated(
    calc,
    "Please use `calc` instead of `easycalc`.\
The use of namespace `easycalc` is deprecated and will be removed in the future.",
)
