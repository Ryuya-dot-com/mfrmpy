"""Source-checkout compatibility for retained research scripts; not distributed."""
import sys
from mfrmpy._core import visual_maps as _implementation
sys.modules[__name__] = _implementation
