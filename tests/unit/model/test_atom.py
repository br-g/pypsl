# pylint: disable=missing-docstring,redefined-outer-name,unused-variable
"""Tests for model.atom"""

import pytest
import pandas as pd
from pypsl.model.predicate import Predicate
from pypsl.model.atom import Atom


@pytest.mark.parametrize("predicate", [
    Predicate('test_pred', [('1', '2', 0.36)]),
])
def test_atom(predicate):
    with pytest.raises(ValueError):
        Atom(predicate, ['a'])
    with pytest.raises(ValueError):
        Atom(predicate, ['a', 'a'])
    atom = Atom(predicate, ['a', 'b'])

    assert isinstance(atom.values, pd.DataFrame)
    assert atom.values.shape[0] == atom.values.shape[0]
    assert atom.values.shape[1] == atom.values.shape[1]
