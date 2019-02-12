# pylint: disable=invalid-name, too-many-locals, too-many-arguments, line-too-long
"""Functions for learning rules' weights"""

from typing import TYPE_CHECKING, Tuple, Optional
from copy import deepcopy
from joblib import Parallel, delayed, parallel_backend
from .utils.logger import Logger

if TYPE_CHECKING:
    # pylint: disable=cyclic-import
    from pypsl.model.model import Model
    from pypsl.model.rule import Rule


def learn_weights(model: 'Model',
                  step_size: float = 0.2,
                  max_iterations: int = 25,
                  scale_gradient: bool = True,
                  l1: float = 0,
                  l2: float = 0,
                  n_jobs: int = -1,
                  logging_period: Optional[int] = 1,
                  inference_step_size: float = 1.0,
                  inference_max_iterations: int = 25000,
                  inference_epsilon_residuals: Tuple[float, float] = (1e-5, 1e-3),
                  inference_epsilon_objective: Optional[float] = None) \
                 -> Tuple[float, ...]:
    """Learns rules' weights using the voted perceptron algorithm.

    Args:
        model : Model
            A grounded PSL model.
        step_size : float, optional
            The step size for ADMM.
        max_iterations : int, optional
            Maximum number of iterations to be performed for weights learning.
        scale_gradient : bool, optional
        	Whether to scale gradient by number of groundings.
        l1 : float, optional
        	Value of L1 regularization.
        l2 : float, optional
        	Value of L2 regularization.
        n_jobs : int, optional
            The number of jobs to be used for the computation.
            If the value is negative, (n_cpus + 1 + n_jobs) jobs are used.
        logging_period : Optional[int], optional
            The period for logging (number of iterations). If `None`, there
            will be no logging.
		inference_step_size : float, optional
            The step size for the inference subroutine.
        inference_max_iterations : int, optional
            A stopping criterion for the inference subroutine: maximum number
            of iterations to be performed.
        inference_epsilon_residuals : Tuple[float, float], optional
            A stopping criterion for the inference subroutine, using residuals.
            If the difference between residuals of two consecutive iterations
            overcomes these values, the optimization stops.
        inference_epsilon_objective : Optional[float], optional
            A stopping criterion for the inference subroutine, using the
            objective. If the difference between the objective of two
            consecutive iterations overcomes this value, the optimization stops.
    Returns:
        The learnt weights.
    """
    if max_iterations <= 0:
        raise ValueError(
            'Invalid max number of iterations')

    cur_model = model
    gradients = [0.] * max_iterations

    for it_idx in range(max_iterations):
        # Reset model, but keep learnt weights
        cur_model = _get_next_model(model, cur_model)

        # Run inference with the last weights
        cur_model.infer(step_size=inference_step_size,
                        max_iterations=inference_max_iterations,
                        epsilon_residuals=inference_epsilon_residuals,
                        epsilon_objective=inference_epsilon_objective,
                        n_jobs=n_jobs,
                        logging_period=None)

        # Compute new weights
        with parallel_backend('threading', n_jobs=n_jobs):
            grads = Parallel()(delayed(
                _update_weight)(rule, step_size, scale_gradient, l1, l2)
                               for rule in cur_model.rules)
            gradients[it_idx] = sum(grads)

        # Logging
        if logging_period is not None and (it_idx + 1) % logging_period == 0:
            Logger.info(
                '--- iteration {} ---\n'.format(it_idx + 1)
                + 'gradient: {}\n'.format(gradients[it_idx])
            )

    # Update model's weights
    _copy_weights(cur_model, model)

    return tuple([r.weight for r in model.rules])


def _get_next_model(initial: 'Model', prev: 'Model') -> 'Model':
    """Creates a new reset model with the last weights."""
    next_model = deepcopy(initial)
    _copy_weights(prev, next_model)
    return next_model


def _update_weight(rule: 'Rule', step_size: float, scale_gradient: bool,
                   l1: float, l2: float) -> float:
    """Computes a new weight value that optimizes likelihood."""
    gradient = rule.get_observed_objective() - rule.get_expected_objective()

    # Regularization
    gradient -= l2 * rule.weight - l1

    # Scale by number of groundings
    if scale_gradient:
        gradient /= max(1, len(rule.grounded))

    rule.weight = max(0., rule.weight - gradient * step_size)

    return gradient


def _copy_weights(source_m: 'Model', target_m: 'Model') -> None:
    """Copies rules' weights."""
    for source_r, target_r in zip(source_m.rules, target_m.rules):
        target_r.weight = source_r.weight
