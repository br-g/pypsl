# pylint: disable=missing-docstring,redefined-outer-name,unused-variable
"""Tests for model.ground_rule"""

import pytest
from pypsl.model.ground_atom import GroundAtom
from pypsl.model.ground_rule import GroundRule


@pytest.mark.parametrize('ground_atoms, distance', [
	(
	    [
	    	GroundAtom(['a', 'b'], 0.3, None, True, True),
	    ], 0.3
    ),
    (
	    [
	    	GroundAtom(['a'], 0.0, None, True, False),
	    	GroundAtom(['b'], 0.8, None, True, True)
	    ], 0.8
    ),
    (
	    [
	    	GroundAtom(['a', 'b', 'c'], 0.3, None, True, False),
	    	GroundAtom(['b', 'c'], 0.4, None, False, False),
	    	GroundAtom(['c'], 0.9, None, True, True)
	    ], 0.2
    ),
])
def test_ground_rule(ground_atoms, distance):
    with pytest.raises(ValueError):
        GroundRule([])

    gr = GroundRule(ground_atoms)
    assert abs(gr.get_observed_dist_to_satisfaction() - distance) < 1e-5
