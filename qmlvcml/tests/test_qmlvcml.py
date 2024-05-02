"""
Unit and regression test for the qmlvcml package.
"""

# Import package, test suite, and other packages as needed
import sys
import pytest

from qmlvcml import *


def test_qmlvcml_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "qmlvcml" in sys.modules

