# pylint: disable=missing-docstring,redefined-outer-name,unused-variable
"""Tests for model.rule"""

from pypsl.model.predicate import Predicate
from pypsl.model.atom import Atom
from pypsl.model.ground_atom import GroundAtom
from pypsl.model.rule import Rule
from pypsl.model.ground_rule import GroundRule


def test_rule():
    pred = Predicate('test_pred', [('1', '2', 0.36)], predict=True)
    rule = Rule(
        [(pred, ('a', 'b'))],
        [(pred, ('b', 'c'))]
        )

    rule.set_weight(0.5)
    assert rule.weight == 0.5

    ground_atoms = frozenset([
        GroundAtom(['a', 'b', 'c'], 0.4, None, True)
        ])
    grounding1 = rule.get_grounded(ground_atoms)
    grounding2 = rule.get_grounded(ground_atoms)
    assert isinstance(grounding1, GroundRule)
    assert isinstance(grounding2, GroundRule)
    assert grounding1 == grounding2

    assert abs(rule.get_observed_objective() - 0.82) < 1e-5
