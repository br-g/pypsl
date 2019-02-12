# pylint: disable=invalid-name
"""Base class for ground rules"""

from typing import FrozenSet
from pypsl.utils.representation import obj_to_repr
from .ground_atom import GroundAtom


class GroundRule:
    """A rule with ground atoms.

    Attributes
    ----------
    open_ground_atoms : tuple[GroundAtom]
    dist_const : float
        The constant term of the distance to satisfaction.
    """

    def __init__(self, ground_atoms: FrozenSet['GroundAtom']) -> None:
        if not ground_atoms:
            raise ValueError('Empty ground rule')
        self.open_ground_atoms = tuple(ga for ga in ground_atoms if ga.is_open)
        self.dist_const = GroundRule._get_dist_const(ground_atoms)

    @staticmethod
    def _get_dist_const(ground_atoms: FrozenSet['GroundAtom']) -> float:
        """Returns the constant term of the distance to satisfaction."""
        dist_const = 0  # type: float
        for ga in ground_atoms:
            if ga.is_negated:
                dist_const += 1
            if not ga.is_open:
                dist_const += -ga.value if ga.is_negated else ga.value
        return 1 - dist_const

    def get_observed_dist_to_satisfaction(self) -> float:
        """Returns the distance to satisfaction, using the observed values."""
        dist = self.dist_const
        for ga in self.open_ground_atoms:
            dist += ga.value if ga.is_negated else -ga.value
        return dist

    def get_expected_dist_to_satisfaction(self) -> float:
        """Returns the distance to satisfaction, using the gold values."""
        dist = self.dist_const
        for ga in self.open_ground_atoms:
            if ga.gold_value is None:
                raise ValueError('Missing gold value')
            dist += ga.gold_value if ga.is_negated else -ga.gold_value
        return dist

    def __repr__(self):
        regular = ['dist_const']
        minimized = ['ground_atoms']
        return obj_to_repr(self, regular, minimized)
