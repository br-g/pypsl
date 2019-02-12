"""Base class for PSL models"""

from typing import TYPE_CHECKING, Iterable, Dict, Tuple, Set, Any
from pypsl import grounding, admm, voted_perceptron
from pypsl.utils.representation import obj_to_repr

if TYPE_CHECKING:
    # pylint: disable=cyclic-import
    from .rule import Rule
    from .predicate import Predicate
    from .ground_atom import GroundAtom


class Model:
    """A PSL model

    Attributes
    ----------
    rules : Iterable[Rule]
    open_predicates : Iterable[Predicate]
    open_ground_atoms : Set[GroundAtom]
    """

    def __init__(self,
                 weighted_rules: Iterable[Tuple[float, 'Rule']]) -> None:
        """Creates a new model.

        Parameters
        ----------
        weighted_rules : Iterable[Tuple[float, 'Rule']]
            Some PSL rules with their associated weights.
        """
        if not weighted_rules:
            raise ValueError('Empty model')

        # Collects weights and rules
        weights = [r[0] for r in weighted_rules]
        self.rules = [r[1] for r in weighted_rules]
        for weight, rule in zip(weights, self.rules):
            if weight < 0:
                raise ValueError('Negative rule weight')
            rule.set_weight(weight)
        self.rules = [r for r in self.rules if r.weight != 0]

        self.open_predicates = Model._get_predicates(self.rules)
        self.open_ground_atoms = set()  # type: Set[GroundAtom]

    @staticmethod
    def _get_predicates(rules: Iterable['Rule']) -> Iterable['Predicate']:
        """Collects all the open predicates of the model."""
        open_pred = set()
        for rule in rules:
            for atom in rule.atoms:
                if atom.predicate.is_open:
                    open_pred.add(atom.predicate)
        return open_pred

    def collect_ground_atoms(self) -> None:
        """Collects all the open ground atoms of the model."""
        for pred in self.open_predicates:
            self.open_ground_atoms.update(pred.grounded.values())

    def ground(self, **kwargs) -> Any:
        """Runs grounding."""
        return grounding.do_ground_model(self, **kwargs)

    def learn_weights(self, **kwargs) -> Any:
        """Learns rules' weights."""
        return voted_perceptron.learn_weights(self, **kwargs)

    def infer(self, **kwargs) -> Any:
        """Runs inference."""
        return admm.infer(self, **kwargs)

    def __repr__(self):
        regular = []
        minimized = ['rules', 'open_predicates', 'open_ground_atoms']
        return obj_to_repr(self, regular, minimized)
