"""
MAP (Maximum A Posteriori) estimation using quadratic (Laplace) approximation.

"""

import numpy as np
from typing import Callable, Any, Tuple

from .optimization import BFGSOptimizer, Config


def log_posterior(parameters: np.ndarray,
                           model: Callable,
                           data: Any,
                           prior_mean: np.ndarray,
                           prior_precision: np.ndarray) -> float:
    """
    Compute negative log posterior (for minimization).

    Args:
        parameters: Parameters
        model: Model function that computes log-likelihood
        data: Data for this subject
        prior_mean: Prior mean (d-dimensional)
        prior_precision: Prior precision matrix (d×d)

    Returns:
        Negative log posterior = -(log_likelihood + log_prior)
    """
    # Compute log-likelihood
    log_lik = model(parameters, data)

    # Compute log prior
    diff = parameters.reshape(-1, 1) - prior_mean.reshape(-1, 1)
    # log_det_precision = np.linalg.slogdet(prior_precision)[1]
    # --- OLD (Line 36) ---
    # log_det_precision = np.log(np.linalg.det(prior_precision))
    
    # --- NEW ---
    log_det_precision = np.linalg.slogdet(prior_precision)[1]
    log_prior = -len(diff) / 2 * np.log(2 * np.pi) + 0.5 * log_det_precision - 0.5 * (diff.T @ prior_precision @ diff).item()

    return (log_lik + log_prior)


def optimize_map(data: Any,
                 model: Callable[[np.ndarray, Any], float],
                 config: Config,
                 prior_mean: np.ndarray,
                 prior_precision: np.ndarray,
                 method: str = 'LAP') -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Quadratic approximation using Laplace approximation (MAP estimation).

    Args:
        data: Data for one subject (any type that model can handle)
        model: Model function that computes log-likelihood
        optimizer: BFGSOptimizer instance (already configured)
        prior_mean: Prior mean (d-dimensional array)
        prior_precision: Prior precision matrix (d×d array)
        method: Method name (only 'LAP' supported)

    Returns:
        Tuple of (loglik, parameters, hessian, grad, flag) where:
            - loglik: Log-likelihood at MAP estimate
            - parameters: MAP estimate (d-dimensional array)
            - hessian: Hessian of negative log posterior at MAP (d×d, this is the precision)
            - grad: Gradient at MAP (d-dimensional array)
            - flag: Convergence flag (1.0=success, 0.5=partial, 0.0=failed)

        NOTE: If optimization fails (flag=0), all values except flag are np.nan
    """
    if method != 'LAP':
        raise ValueError(f"Method '{method}' is not recognized! Only 'LAP' is supported.")

    d = len(prior_mean)    
    optimizer = BFGSOptimizer(d, config=config)

    # Define objective function (negative log posterior)
    def objective(theta_vec):
        return -log_posterior(theta_vec, model, data, prior_mean, prior_precision)

    # Optimize, starting from prior mean
    result = optimizer.optimize(objective, x_init=prior_mean.flatten())

    # Extract results
    if result.flag == 0:
        # Optimization failed
        parameters = np.full(d, np.nan)
        hessian = np.full((d, d), np.nan)
        grad = np.full(d, np.nan)
        loglik = np.nan
    else:
        parameters = result.x
        hessian = result.hess  # Hessian of negative log posterior = precision at MAP
        grad = result.grad

        # Compute log posterior at MAP
        loglik = log_posterior(result.x, model, data, prior_mean, prior_precision)

    flag = result.flag

    return loglik, parameters, hessian, grad, flag