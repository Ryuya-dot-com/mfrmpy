"""Source-checkout compatibility for retained research scripts; not distributed."""
import sys
from mfrmpy._core import numerical_qualification as _implementation
sys.modules[__name__] = _implementation
