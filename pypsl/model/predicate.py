# pylint: disable=invalid-name
"""Base class for predicates"""

from typing import Tuple, List, Dict, Union, Any, Optional
import pandas as pd
from pypsl.utils.representation import obj_to_repr
from .ground_atom import GroundAtom


class Predicate:
    """A relation defined by a unique identifier and an arity.
    A predicate could be closed (the value of its atoms is a constant) or open
    (the value of its atoms is a variable).

    Attributes
    ----------
    name : str
    values : pd.DataFrame
    gold_values : pd.DataFrame
    arity : int
    is_open : bool
    grounded : Dict[Tuple[Any, ...], GroundAtom]
    """

    def __init__(self,
                 name: str,
                 input_data: Tuple[Tuple[Union[str, float]]],
                 gold_data: Optional[Tuple[Tuple[Union[str, float]]]] = None,
                 predict: bool = False) -> None:
        """Creates a new predicate.

        Parameters
        ----------
        name : str
            The identifier of the predicate.
        input_data : Tuple[Tuple[Union[str, float]]]
            The argments the predicate can take, with the associated values. In
            the nested tuples, the first elements are the arguments (str), the
            last element is the value (float).
        gold_data : Optional[Tuple[Tuple[Union[str, float]]]], optional
            The argments the predicate can take, with the associated truth
            values. This is used for learning model's weights.
        predict : bool, optional
            Whether the values associated to the predicate should be inferred.
        """
        if name == '' or not all(c.isalnum() or c == '_' for c in name):
            raise ValueError('Invalid predicate name: `{}`'.format(name))
        self.name = name
        self.values, self.arity = Predicate._parse_data(input_data, gold_data)
        self.is_open = predict
        self.grounded = {}  # type: Dict[Tuple[Any, ...], GroundAtom]

    @staticmethod
    def _parse_data(input_data: Tuple[Tuple[Union[str, float]]], \
                    gold_data: Optional[Tuple[Tuple[Union[str, float]]]]) \
                   -> Tuple['pd.DataFrame', int]:
        """Parses user's raw data."""
        if not input_data:
            raise ValueError('Empty input data')
        arity = len(input_data[0]) - 1
        if arity == 0:
            raise ValueError("Predicates' arity should be strictly positive")

        # Initialize
        values = {
            '__value': [],
            '__gold_value': []
            }  # type: Dict[str, List[Union[str, Optional[float]]]]
        for i in range(arity):
            values['arg{}'.format(i)] = []

        # Populate
        for row in input_data:
            if len(row) != arity + 1:
                raise ValueError('Inconsistent tuples length in data')
            if not isinstance(row[-1], (int, float)):
                raise ValueError("Atom's value is not a number")
            if not all(isinstance(e, str) for e in row[:-1]):  # type: ignore
                raise ValueError("Atom's arguments should be strings")
            for i, e in enumerate(row[:-1]):  # type: ignore
                values['arg{}'.format(i)].append(e)
            values['__value'].append(row[-1])

        if gold_data:
            for row in gold_data:
                values['__gold_value'].append(row[-1])
        else:
            values['__gold_value'] = [None] * len(input_data)

        values_df = pd.DataFrame(data=values)
        values_df['__merge_key__'] = 1  # for merging with pandas
        return values_df, arity

    def get_grounded(self, args: Tuple[Any, ...], value: float,
                     gold_value: float, is_negated: bool) -> 'GroundAtom':
        """Creates a grounding and returns it."""
        if args not in self.grounded:
            ground_atom = GroundAtom(args, value, gold_value, self.is_open)
            self.grounded[args] = ground_atom
        return self.grounded[args].make_copy(is_negated)

    def __repr__(self):
        regular = ['name', 'arity', 'is_open']
        minimized = ['values', 'gold_values', 'grounded']
        return obj_to_repr(self, regular, minimized)
