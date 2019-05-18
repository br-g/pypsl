"""Base class for rules"""

from typing import TYPE_CHECKING, Dict, FrozenSet, List, Tuple
from pypsl.utils.representation import obj_to_repr
from .ground_rule import GroundRule
from .atom import Atom
from .ground_atom import GroundAtom

if TYPE_CHECKING:
    # pylint: disable=cyclic-import
    from .predicate import Predicate


class Rule:
    """A disjunction of atoms, and a weight.

    Attributes
    ----------
    atoms : list[Atom]
    negated : list[bool]
    weight : float
    grounded : Dict[FrozenSet[GroundAtom], GroundRule]
    """

    def __init__(self,
                 positive_atoms: List[Tuple['Predicate', Tuple[str, ...]]],
                 negative_atoms: List[Tuple['Predicate', Tuple[str, ...]]]) \
                -> None:
        """Creates a new rule.

        Parameters
        ----------
        positive_atoms : List[Tuple[Predicate, Tuple[str, ...]]]
            The positive atoms composing the rule.
        negative_atoms : List[Tuple[Predicate, Tuple[str, ...]]]
            The negative atoms composing the rule.
        """
        self.atoms = [Atom(*at) for at in positive_atoms + negative_atoms]
        self.negated = [False] * len(positive_atoms) \
                       + [True] * len(negative_atoms)
        self.weight = 0.
        self.grounded = {}  # type: Dict[FrozenSet[GroundAtom], GroundRule]

    def set_weight(self, value: float) -> None:
        """Sets rule's weight."""
        if value < 0:
            raise ValueError("Negative rule's weight")
        self.weight = value

    def get_grounded(self, ground_atoms: FrozenSet['GroundAtom']) \
                    -> 'GroundRule':
        """Creates a grounding if it doesn't exist and returns it."""
        if ground_atoms in self.grounded:
            return self.grounded[ground_atoms]
        new_grounded = GroundRule(ground_atoms)
        self.grounded[ground_atoms] = new_grounded
        return new_grounded

    def get_observed_objective(self) -> float:
        # pylint: disable=invalid-name
        """Returns the observed objective."""
        obj = 0.
        for gr in self.grounded.values():
            dist = gr.get_observed_dist_to_satisfaction()
            obj += max(0, 1 - self.weight * max(0, dist) ** 2)
        return obj

    def get_expected_objective(self) -> float:
        # pylint: disable=invalid-name
        """Returns the expected objective."""
        obj = 0.
        for gr in self.grounded.values():
            dist = gr.get_expected_dist_to_satisfaction()
            obj += max(0, 1 - self.weight * max(0, dist) ** 2)
        return obj

    def __repr__(self):
        regular = ['negated', 'weight']
        minimized = ['atoms', 'grounded']
        return obj_to_repr(self, regular, minimized)
