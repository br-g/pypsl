# pylint: disable=missing-docstring,redefined-outer-name,unused-variable
"""Tests for model.predicate"""

import pytest
import pandas as pd
from pypsl.model.predicate import Predicate


def test_predicate():
    with pytest.raises(ValueError):
        Predicate('', [('1', '2', 0.36)])
    with pytest.raises(ValueError):
        Predicate('abc$', [('1', '2', 0.36)])
    with pytest.raises(ValueError):
        Predicate('abc', [])
    with pytest.raises(TypeError):
        Predicate('abc', [(0.36)])

    pred = Predicate('abc', [
        ('a', 'b', 'c', 0.36),
        ('a', 'c', 'b', 0.71)
        ], predict=True)
    assert pred.arity == 3
    assert isinstance(pred.values, pd.DataFrame)
    assert pred.values.shape[0] == 2
    assert pred.values.shape[1] == 6

    copy1 = pred.get_grounded(('1', '2', '3'), 0.54, None, True)
    copy2 = pred.get_grounded(('1', '2', '3'), 0.54, None, False)
    assert copy1 != copy2
