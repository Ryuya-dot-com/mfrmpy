"""Source-checkout compatibility for retained research scripts; not distributed."""
import sys
from mfrmpy._core import desktop_i18n as _implementation
sys.modules[__name__] = _implementation
