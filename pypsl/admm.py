# pylint: disable=invalid-name, too-many-arguments, line-too-long
"""Functions for inference with ADMM"""

from math import sqrt
from collections import defaultdict
from typing import (TYPE_CHECKING, Dict, DefaultDict, List, Tuple,
                    Set, Iterator, Optional, Any)
from joblib import Parallel, delayed, parallel_backend
import numpy as np
from natsort import natsorted
from pypsl.utils.logger import Logger

if TYPE_CHECKING:
    # pylint: disable=cyclic-import
    from pypsl.model.model import Model
    from pypsl.model.ground_rule import GroundRule
    from pypsl.model.ground_atom import GroundAtom


_cholesky_cache = {}  # type: Dict[Tuple[int, float], np.ndarray]


def infer(model: 'Model',
          step_size: float = 1.0,
          max_iterations: int = 25000,
          epsilon_residuals: Tuple[float, float] = (1e-5, 1e-3),
          epsilon_objective: Optional[float] = None,
          n_jobs: int = -1,
          logging_period: Optional[int] = 10) \
          -> Dict[str, Tuple[Tuple[Any, ...], ...]]:
    """Runs inference with ADMM.

    Args:
        model : Model
            A grounded PSL model.
        step_size : float, optional
            The step size for ADMM.
        max_iterations : int, optional
            A stopping criterion: maximum number of iterations to be performed.
        epsilon_residuals : Tuple[float, float], optional
            A stopping criterion using residuals.
            If the difference between residuals of two consecutive iterations
            overcomes these values, the optimization stops.
        epsilon_objective : Optional[float], optional
            A stopping criterion using the objective.
            If the difference between the objective of two consecutive
            iterations overcomes this value, the optimization stops.
        n_jobs : int, optional
            The number of jobs to be used for the computation.
            If the value is negative, (n_cpus + 1 + n_jobs) jobs are used.
        logging_period : Optional[int], optional
            The period for logging (number of iterations). If `None`, there
            will be no logging.
    Returns:
        The predictions, and the objective at each iteration.
    """
    _check_arguments(step_size, max_iterations, epsilon_residuals,
                     epsilon_objective)
    epsilon_const_term = _get_epsilon_const_term(model, epsilon_residuals[0])

    objectives = []  # type: List[float]
    for it_idx in range(max_iterations):
        # Optimization
        with parallel_backend('threading', n_jobs=n_jobs):
            _optimize_all(model, step_size)
            residual, norm = _join_all(model.open_ground_atoms, step_size)
        objectives.append(
            sum(r.get_observed_objective() for r in model.rules))

        # Logging
        if logging_period is not None and (it_idx + 1) % logging_period == 0:
            Logger.info(
                '--- iteration {} ---\n'.format(it_idx + 1) \
                + 'objective: {}\n'.format(objectives[-1]) \
                + 'primal residual: {}\n'.format(residual['primal']) \
                + 'dual residual: {}\n'.format(residual['dual'])
            )

        # Stopping criteria
        if _is_completed(residual, norm, objectives, epsilon_residuals[1],
                         epsilon_const_term, epsilon_objective):
            break

    if logging_period is not None:
        Logger.info('Completed after {} iterations'.format(len(objectives)))

    return _get_output(model)


def _check_arguments(step_size: float, max_iterations: int,
                     epsilon_res: Tuple[float, float],
                     epsilon_obj: Optional[float]) -> None:
    """Checks arguments' value and raises an exception if one of them is
       invalid."""
    if step_size <= 0:
        raise ValueError('The step size should be strictly positive')

    if max_iterations <= 0:
        raise ValueError(
            'The maximum number of iterations should be strictly positive')

    if epsilon_res is not None \
        and (len(epsilon_res) != 2 \
        or not all(isinstance(e, (int, float)) for e in epsilon_res)):
        raise ValueError('Invalid epsilon residual values')

    if epsilon_obj and not isinstance(epsilon_obj, (int, float)):
        raise ValueError('Invalid epsilon objective values')


def _get_epsilon_const_term(model: 'Model', eps_primal_res: float) -> float:
    """Computes the constant term of epsilon for residuals."""
    n_variables = len([
        copy for ga in model.open_ground_atoms for copy in ga.copies])
    return sqrt(n_variables) * eps_primal_res


def _optimize_all(model: 'Model', step_size: float) -> None:
    """Optimizes independently for each ground atom copy."""
    # Update lagrange multipliers (in parallel)
    def _lagrange_gen(model: 'Model') -> Iterator[Tuple['GroundAtom', float]]:
        for ga in model.open_ground_atoms:
            for copy in ga.copies:
                yield (copy, step_size)
    Parallel()(delayed(
        _update_lagrange)(*e) for e in _lagrange_gen(model))

    # Minimize the objective (in parallel)
    def _minimize_gen(model: 'Model') \
                     -> Iterator[Tuple['GroundRule', float, float]]:
        for r in model.rules:
            for gr in r.grounded.values():
                yield (gr, r.weight, step_size)
    Parallel()(delayed(
        _minimize)(*e) for e in _minimize_gen(model))


def _update_lagrange(copy: 'GroundAtom', step_size: float) -> None:
    """Updates local copies with Lagrange multipliers."""
    assert copy.copy_of
    copy.lagrange_mult += step_size * (copy.value - copy.copy_of.value)
    copy.value = copy.copy_of.value - copy.lagrange_mult / step_size


def _minimize(ground_rule: 'GroundRule', weight: float,
              step_size: float) -> None:
    # pylint: disable=too-many-branches
    """Optimizes the potential of ground rules using gradient."""
    if ground_rule.get_observed_dist_to_satisfaction() <= 0:
        return

    # Local copies of ground atoms
    copies = ground_rule.open_ground_atoms

    # Constant term of the gradient
    for copy in copies:
        assert copy.copy_of
        copy.value = step_size * copy.copy_of.value - copy.lagrange_mult
        if copy.is_negated:
            copy.value -= 2 * weight * ground_rule.dist_const
        else:
            copy.value += 2 * weight * ground_rule.dist_const

    # Corner cases
    if len(copies) == 1:
        copies[0].value /= 2 * weight + step_size

    elif len(copies) == 2:
        a0 = 2 * weight + step_size
        if copies[0].is_negated == copies[1].is_negated:
            a1b0 = 2 * weight
        else:
            a1b0 = -2 * weight
        copies[1].value -= a1b0 * copies[0].value / a0
        copies[1].value /= a0 - a1b0 ** 2 / a0
        copies[0].value -= a1b0 * copies[1].value
        copies[0].value /= a0

    # General case, using Cholesky decomposition
    else:
        cholesky_L = _get_cholesky_L(copies, weight, step_size)

        for i, ga_i in enumerate(copies):
            for j, ga_j in list(enumerate(copies))[:i]:
                ga_i.value -= cholesky_L[i][j] * ga_j.value
            ga_i.value /= cholesky_L[i][i]

        for i, ga_i in reversed(list(enumerate(copies))):
            for j, ga_j in reversed(list(enumerate(copies))[i+1:]):
                ga_i.value -= cholesky_L[j][i] * ga_j.value
            ga_i.value /= cholesky_L[i][i]


def _get_cholesky_L(copies: Tuple['GroundAtom', ...], weight: float,
                    step_size: float) -> 'np.ndarray':
    """Solves the system of linear equations using Cholesky decomposition."""
    # Reuse previous results when possible
    mat_hash = (len(copies), weight)
    if mat_hash in _cholesky_cache:
        return _cholesky_cache[mat_hash]

    mat = [[0. for x in range(len(copies))] for y in range(len(copies))]
    for i, ga_i in enumerate(copies):
        mat[i][i] = 2 * weight + step_size
        for j, ga_j in list(enumerate(copies))[i+1:]:
            if ga_i.is_negated == ga_j.is_negated:
                mat[i][j] = 2 * weight
            else:
                mat[i][j] = -2 * weight
            mat[j][i] = mat[i][j]

    _cholesky_cache[mat_hash] = np.linalg.cholesky(mat)
    return _cholesky_cache[mat_hash]


def _join_all(ground_atom: Set['GroundAtom'], step_size: float) \
             -> Tuple[Dict[str, float], Dict[str, float]]:
    """Computes the new consensus value of ground atoms (in parallel)."""
    sum_res = {'primal': 0., 'dual': 0.}
    sum_norm = {'ax': 0., 'ay': 0., 'bz': 0.}

    if ground_atom:
        res, norms = zip(*Parallel()(
            delayed(_join)(ga, step_size) for ga in ground_atom))

        for r, n in zip(res, norms):
            sum_res['primal'] += r['primal']
            sum_res['dual'] += r['dual']
            sum_norm['ax'] += n['ax']
            sum_norm['ay'] += n['ay']
            sum_norm['bz'] += n['bz']

    sum_res = {k: sqrt(sum_res[k]) for k in sum_res}
    return sum_res, sum_norm


def _join(ga: 'GroundAtom', step_size: float) \
         -> Tuple[Dict[str, float], Dict[str, float]]:
    """Computes a new consensus value by joining the value of local copies."""
    # Compute new consensus value
    local = [c.value + c.lagrange_mult / step_size for c in ga.copies]
    consensus = max(min(sum(local) / len(local), 1), 0)

    # Residuals and norms
    residual = {
        'primal': sum((c.value - consensus) ** 2 for c in ga.copies),
        'dual': len(ga.copies) * (step_size * (ga.value - consensus)) ** 2
    }
    norm = {
        'ax': sum(c.value ** 2 for c in ga.copies),
        'ay': sum(c.lagrange_mult ** 2 for c in ga.copies),
        'bz': consensus ** 2 * len(ga.copies)
    }

    # Update ground atom's value
    ga.value = consensus

    return residual, norm


def _is_completed(residual: Dict[str, float], norm: Dict[str, float],
                  objectives: List[float], eps_dual_res: float,
                  eps_abs_term: float, eps_obj: Optional[float]) -> bool:
    """Decides whether inference is completed."""
    eps_primal = eps_abs_term \
        + eps_dual_res * max(sqrt(norm['ax']), sqrt(norm['bz']))
    eps_dual = eps_abs_term + eps_dual_res * sqrt(norm['ay'])

    if len(objectives) >= 2 and eps_obj is not None \
       and objectives[-2] - objectives[-1] < eps_obj:
        return True
    if residual['primal'] < eps_primal and residual['dual'] < eps_dual:
        return True
    return False


def _get_output(model: 'Model') -> Dict[str, Tuple[Tuple[Any, ...], ...]]:
    """Collects and formats inference results."""
    res = defaultdict(list)  # type: DefaultDict[str, List[Tuple[Any, ...]]]
    for pred in model.open_predicates:
        for ga in pred.grounded.values():
            res[pred.name].append(
                tuple(list(ga.terms) + [ga.value]))

    output = {}  # type: Dict[str, Tuple[Tuple[Any, ...], ...]]
    for pred_name in res:
        output[pred_name] = tuple(
            natsorted(res[pred_name], key=lambda x: x[:-1]))
    return output
