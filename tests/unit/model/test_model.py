# pylint: disable=missing-docstring,redefined-outer-name,unused-variable
"""Tests for model.model"""

import pytest
from pypsl.model.predicate import Predicate
from pypsl.model.rule import Rule
from pypsl.model.model import Model


def test_model():
    pred = Predicate('test_pred', [('1', '2', 0.36)])
    rule = Rule(
        [(pred, ('a', 'b'))],
        [(pred, ('b', 'c'))]
        )

    with pytest.raises(ValueError):
        Model(())
    with pytest.raises(ValueError):
        Model([(-1.0, rule)])

    model = Model([(1.0, rule)])
    model.ground(check_data=False)
    model.infer()
