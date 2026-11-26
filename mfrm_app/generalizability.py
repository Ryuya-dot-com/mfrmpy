"""Source-checkout compatibility for retained research scripts; not distributed."""
import sys
from mfrmpy._core import generalizability as _implementation
sys.modules[__name__] = _implementation
