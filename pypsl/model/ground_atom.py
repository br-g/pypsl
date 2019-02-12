# pylint: disable=too-many-arguments, too-many-instance-attributes
"""Base class for ground atoms"""

from typing import Tuple, List, Optional, Any
from pypsl.utils.representation import obj_to_repr


class GroundAtom:
    """An atom with ground terms.

    Attributes
    ----------
    terms : Tuple[str]
    value : float
    gold_value : float
    is_open : bool
    is_negated : bool
    copy_of : GroundAtom
    copies : List[GroundAtom]
    lagrange_mult : float
    """

    def __init__(self, terms: Tuple[Any, ...], value: float,
                 gold_value: Optional[float], is_open: bool,
                 is_negated: bool = None, copy_of: 'GroundAtom' = None) -> None:
        if value < 0 or value > 1:
            raise ValueError('Invalid truth value: `{}`'.format(value))
        self.value = value
        if gold_value is not None and (gold_value < 0 or gold_value > 1):
            raise ValueError('Invalid gold value: `{}`'.format(gold_value))
        self.gold_value = gold_value

        self.terms = terms
        self.is_open = is_open
        self.is_negated = is_negated
        self.copy_of = copy_of
        self.copies = []  # type: List[GroundAtom]
        self.lagrange_mult = 0.

    def make_copy(self, is_negated: bool) -> 'GroundAtom':
        """Makes a copy of the current object and returns it."""
        copy = GroundAtom(self.terms,
                          self.value,
                          self.gold_value,
                          self.is_open,
                          is_negated,
                          self)
        self.copies.append(copy)
        return copy

    def __repr__(self):
        regular = ['terms', 'value', 'gold_value', 'is_open', 'is_negated',
                   'lagrange_mult']
        minimized = ['copy_of', 'copies']
        return obj_to_repr(self, regular, minimized)
