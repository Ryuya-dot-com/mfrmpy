"""Source-checkout compatibility for retained research scripts; not distributed."""
import sys
from mfrmpy._core import mml_fixed_sd_review as _implementation
sys.modules[__name__] = _implementation
