"""Source-checkout compatibility for retained research scripts; not distributed."""
import sys
from mfrmpy._core import multivariate_gd as _implementation
sys.modules[__name__] = _implementation
