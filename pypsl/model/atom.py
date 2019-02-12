# pylint: disable=too-few-public-methods
"""Base class for atoms"""

from copy import deepcopy
from typing import TYPE_CHECKING, Tuple
from pypsl.utils.representation import obj_to_repr

if TYPE_CHECKING:
    # pylint: disable=cyclic-import
    import pandas as pd
    from .predicate import Predicate


class Atom:
    """The combination of a predicate, some terms and a value.

    Attributes
    ----------
    predicate : Predicate
    terms : Tuple[str, ...]
    values : pd.DataFrame
    """

    def __init__(self, predicate: 'Predicate', terms: Tuple[str, ...]) -> None:
        self.predicate = predicate
        Atom._check_terms(terms, predicate.arity)
        self.terms = terms
        self.values = Atom._get_values(predicate, terms)

    @staticmethod
    def _check_terms(terms: Tuple[str, ...], arity: int) -> None:
        """Checks terms are valid."""
        if len(terms) != arity:
            raise ValueError('The number of terms does not match '
                             "predicate's arity")
        if len(terms) != len(set(terms)):
            raise ValueError('Duplicated terms')

    @staticmethod
    def _get_values(predicate: 'Predicate',
                    terms: Tuple[str, ...]) -> 'pd.DataFrame':
        """Returns atom's values."""
        values = deepcopy(predicate.values)
        for i, term in enumerate(terms):
            values.rename(columns={'arg{}'.format(i): term}, inplace=True)
        return values

    def __repr__(self):
        regular = ['terms']
        minimized = ['predicate', 'values']
        return obj_to_repr(self, regular, minimized)
