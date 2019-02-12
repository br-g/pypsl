# pylint: disable=missing-docstring,redefined-outer-name,unused-variable
"""Tests for model.ground_atom"""

import pytest
from pypsl.model.ground_atom import GroundAtom


def test_ground_atom():
    with pytest.raises(ValueError):
        GroundAtom(['a', 'b'], -0.2, None, True)
    with pytest.raises(ValueError):
        GroundAtom(['a', 'b'], 1.2, None, True)
    ga = GroundAtom(['a', 'b'], 0.6, None, True)

    copy = ga.make_copy(True)
    assert isinstance(copy, GroundAtom)
    assert ga == copy.copy_of
