# pylint: disable=line-too-long
"""Functions for model grounding"""

from typing import (TYPE_CHECKING, Tuple, List, Dict, Set,
                    Iterator, Optional, Any)
from functools import reduce
from itertools import chain
from joblib import Parallel, delayed, parallel_backend
import numpy as np
import pandas as pd
from pypsl.utils.logger import Logger
from pypsl.model.ground_atom import GroundAtom

if TYPE_CHECKING:
    # pylint: disable=cyclic-import
    from pypsl.model.model import Model
    from pypsl.model.rule import Rule
    from pypsl.model.atom import Atom


def do_ground_model(model: 'Model',
                    check_data: bool = True,
                    n_jobs=-1) -> None:
    """Creates ground rules and ground atoms.

    Parameters
    ----------
    model : Model
        A non-grounded PSL model.
    check_data : bool, optional
        Whether to check for missing data or ignore it.
    n_jobs : int, optional
        The number of jobs to be used for the computation.
        If the value is negative, (n_cpus + 1 + n_jobs) jobs are used.
    """
    with parallel_backend('threading', n_jobs=n_jobs):
        for rule in model.rules:
            _do_ground_rule(rule, check_data)
    model.collect_ground_atoms()
    _log_results(model)


def _do_ground_rule(rule: 'Rule', check_data: bool) -> None:
    """Creates ground rules (in parallel)."""
    Parallel()(delayed(
        _do_ground_combination)(rule, val) \
        for val in _get_combinations(rule, check_data))


def _get_combinations(rule: 'Rule', check_data: bool) \
                     -> Iterator['pd.core.series.Series']:
    """Combines ground atoms in rules."""
    # Merge atoms' values within the rule
    values = [
        atom.values.rename(columns={
            '__value': '__value{}'.format(i),
            '__gold_value': '__gold_value{}'.format(i)
            }) \
        for i, atom in enumerate(rule.atoms)
    ]
    comb = reduce(lambda l, r: pd.merge(l, r, how='outer'), values)

    # Discard constant ground rules
    constants = comb.apply(lambda c: _is_constant(rule, c), axis=1)
    comb = comb.loc[~constants]

    # Discard ground rules with ground term duplicates
    ground_terms = [t for t in comb if t[:2] != '__']
    dup = comb.apply(lambda c: _has_duplicates(c, ground_terms), axis=1)
    comb = comb.loc[~dup]

    # Handle missing data
    if check_data:
        _check_missing_data(rule.atoms, comb)
    else:
        subset = ['__value{}'.format(i) for i in range(len(rule.atoms))]
        comb = comb.dropna(subset=subset)

    for _, val in comb.iterrows():
        yield val


def _do_ground_combination(rule: 'Rule', comb: 'pd.core.series.Series') -> None:
    """Creates ground rules from combinations."""
    ground_atoms = {}  # type: Dict[Tuple[Optional[GroundAtom], bool], GroundAtom]
    for i, (atom, is_negated) in enumerate(zip(rule.atoms, rule.negated)):
        value = comb['__value{}'.format(i)]
        gold_value = comb['__gold_value{}'.format(i)]
        grounded = _do_ground_atom(atom, comb, value, gold_value, is_negated)
        if not grounded:
            continue
        # If the rule is trivial, skip it
        if (grounded.copy_of, not is_negated) in ground_atoms:
            return
        ground_atoms[(grounded.copy_of, is_negated)] = grounded
    rule.get_grounded(frozenset(ground_atoms.values()))


def _do_ground_atom(atom: 'Atom', args: 'pd.core.series.Series', value: float,
                    gold_value: float, is_negated: bool) \
                   -> Optional['GroundAtom']:
    """Creates a ground atom."""
    if not atom.predicate.is_open:
        if (not is_negated and value == 0) \
            or (is_negated and value == 1):
            return None  # atom is always satisfied, skip it
    return atom.predicate.get_grounded(
        tuple(args[atom.terms]), value, gold_value, is_negated)


def _check_missing_data(atoms: List['Atom'],
                        comb: 'pd.core.series.Series') -> None:
    """Raises an exception if there is missing data."""
    term_count = 0
    print(comb)
    while '__value{}'.format(term_count) in comb:
        if np.isnan(comb['__value{}'.format(term_count)]).any():
            raise ValueError('Missing values for predicate `{}`'.format(
                atoms[term_count].predicate.name))
        term_count += 1


def _is_constant(rule: 'Rule', comb: 'pd.core.series.Series') -> bool:
    """Decides whether a ground rule is constant."""
    for i, (atom, negated) in enumerate(zip(rule.atoms, rule.negated)):
        if atom.predicate.is_open:
            continue
        if not negated and comb['__value{}'.format(i)] == 1:
            return True
        if negated and comb['__value{}'.format(i)] == 0:
            return True
    return False


def _has_duplicates(comb: 'pd.core.series.Series', terms: List[Any]) -> bool:
    """Decides whether a ground rule has ground term duplicates."""
    ground_terms = set()  # type: Set[str]
    for term in terms:
        if comb[term] in ground_terms:
            return True
        ground_terms.add(comb[term])
    return False


def _log_results(model: 'Model') -> None:
    """Logs the outcome of grounding."""
    n_ground_rules = sum([len(r.grounded) for r in model.rules])
    n_ground_atoms = len(list(
        chain(*[chain(*r.grounded.keys()) for r in model.rules]
    ))) # pylint: disable=bad-continuation
    Logger.info('{} ground rules and {} ground atoms have been created.'.format(
        n_ground_rules, n_ground_atoms))
