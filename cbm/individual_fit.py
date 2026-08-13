"""
Individual subject fitting using Laplace approximation.

This module provides the individual_fit function for fitting computational models
to multiple subjects using Laplace approximation (MAP estimation).

"""

import numpy as np
import pickle
from typing import Callable, Optional, List, Any, Union
from dataclasses import dataclass
from datetime import datetime
import warnings
import time

from .map_estimation import optimize_map, log_posterior
from .optimization import BFGSOptimizer, Config


@dataclass
class Prior:
    """
    Gaussian prior specification.

    Attributes:
        mean: Prior mean (d-dimensional array)
        variance: Prior variance (scalar, vector, or d×d matrix)
        precision: Prior precision matrix (inverse of covariance), computed automatically
    """
    mean: np.ndarray
    variance: np.ndarray
    precision: Optional[np.ndarray] = None

    def __post_init__(self):
        """Compute precision matrix from variance."""
        d = len(self.mean)

        # Convert variance to covariance matrix
        if np.isscalar(self.variance):
            cov = self.variance * np.eye(d)
        elif self.variance.ndim == 1:
            cov = np.diag(self.variance)
        else:
            cov = self.variance

        # Compute precision (inverse of covariance)
        self.precision = np.linalg.inv(cov)

        # Ensure mean is column vector
        if self.mean.ndim == 1:
            self.mean = self.mean.reshape(-1, 1)

@dataclass
class FitInput:
    """Profile and input information."""
    model_name: str
    prior_mean: np.ndarray
    prior_precision: np.ndarray
    fname: Optional[str]

@dataclass
class FitProfile:
    """Profile and input information."""
    datetime: str
    filename: str  # function name
    telapsed: float
    config: Any  # stored as plain dict to avoid pickle class-identity issues
    prior_mean: np.ndarray
    prior_precision: np.ndarray

@dataclass
class FitMath:
    """Mathematical details from fitting."""
    loglik: np.ndarray
    parameters: List[np.ndarray]
    hessian: List[np.ndarray]
    lme: np.ndarray
    hessian_inv_diag: List[np.ndarray]
    log_det_hessian: np.ndarray
    flag: np.ndarray
    gradient: np.ndarray    

@dataclass
class FitOutput:
    """Main output from fitting."""
    parameters: np.ndarray  # N×d matrix
    log_evidence: np.ndarray  # N×1 vector


@dataclass
class FitResult:
    """
    Result from individual fitting.

    Attributes:
        method: Method name ('individual_fit')
        profile: Profile and input information
        math: Mathematical details
        output: Main output (parameters and log_evidence)
    """
    method: str
    input: FitInput
    profile: FitProfile
    math: FitMath
    output: FitOutput


def individual_fit(data: List[Any],
                   model: Callable[[np.ndarray, Any], float],
                   prior_mean: np.ndarray,
                   prior_variance: np.ndarray | float,
                   fname: Optional[str] = None,
                   config: Optional[Union[Config, dict]] = None) -> FitResult:
    """
    Individual subject fitting using Laplace approximation.

    Args:
        data: List of data for N subjects (each element can be any type)
        model: Function that computes log-likelihood given parameters and data
               Signature: model(theta, data) -> log_likelihood
        prior: Prior object with mean and variance
        fname: Filename for saving output using pickle (None for no saving)
        config: Configuration object (optional)

    Returns:
        Tuple of (cbm, success) where:
            - cbm: CBM dataclass with all results
    """
    # Setup
    N = len(data)  # Number of subjects
    d = len(prior_mean)  # Number of parameters

    prior = Prior(
        mean=prior_mean,
        variance=prior_variance  # prior variances
    )

    # Configuration handling: allow Config or dict
    if config is None:
        config = Config(d=d)
    else:
        if isinstance(config, dict):
            # Ensure dimension d is present/consistent
            cfg_kwargs = dict(config)
            cfg_kwargs["d"] = d
            config = Config(**cfg_kwargs)
        elif isinstance(config, Config):
            # If provided Config has different d, overwrite to current d
            # to keep consistency with provided prior dimensions
            if getattr(config, "d", None) != d:
                # Reconstruct config preserving fields while updating d
                cfg_kwargs = config.__dict__.copy()
                cfg_kwargs["d"] = d
                config = Config(**cfg_kwargs)
        else:
            raise TypeError("config must be a Config or dict or None")

    # Initial report
    start_time = datetime.now()
    if config.verbose:
        print("=" * 70)
        print(f"{'individual_fit':<40}{start_time.strftime('%Y-%m-%d %H:%M:%S'):>30}")
        print("=" * 70)
        print(f"Number of samples: {N}")
        print(f"Number of parameters: {d}\n")
        print(f"Number of initializations: {config.num_init}")
        print("-" * 70)

    # Test the model at prior mean
    try:
        test_loglik = model(prior.mean.flatten(), data[0])
        if not np.isfinite(test_loglik):
            warnings.warn("Model returns non-finite value at prior mean!")
    except Exception as e:
        raise ValueError(f"Model failed at prior mean: {str(e)}")

    # Initialize storage
    flags = np.full(N, np.nan)
    loglik = np.full(N, np.nan)
    parameters_list = []
    hessian_list = []
    G = np.full((d, N), np.nan)
    lme = np.full(N, np.nan)  # log-model-evidence
    hessian_inv_diag = []
    log_det_hessian = np.full(N, np.nan)


    # Main loop over subjects
    t_start = time.time()

    for n in range(N):
        if config.verbose:
            print(f"Subject: {n + 1:02d}")

        dat = data[n]

        # Create optimizer for this subject        

        # Call optimize_map for this subject
        loglik_n, parameters_n, hessian_n, grad_n, flag_n = optimize_map(
            dat, model, config, prior.mean.flatten(), prior.precision, method='LAP'
        )

        # Handle failed optimization
        if flag_n == 0:
            if config.verbose:
                print(f"No minimum found for subject {n + 1:02d}")

            if config.prior_for_failed:
                if config.verbose:
                    print("No minimum found, use prior values as individual parameters")
                parameters_n = prior.mean.flatten()
                loglik_n = log_posterior(parameters_n, model, dat, prior.mean.flatten(), prior.precision)
                hessian_n = prior.precision.copy()
                grad_n = np.full(d, np.nan)
            else:
                print(f"No minimum found for subject {n + 1:02d}")
                raise RuntimeError(f"Optimization failed: No minimum found for subject {n+1:02d}")

        # Store results
        flags[n] = flag_n
        parameters_list.append(parameters_n)
        loglik[n] = loglik_n
        hessian_list.append(hessian_n)
        hessian_inv_diag.append(np.diag(np.linalg.inv(hessian_n)))
        G[:, n] = grad_n

        log_det_hess = np.linalg.slogdet(hessian_n)[1]
        log_det_hessian[n] = log_det_hess

        # Compute log model evidence (Laplace approximation)
        lme[n] = loglik_n + 0.5 * d * np.log(2 * np.pi) - 0.5 * log_det_hess

    t_elapsed = time.time() - t_start

    # Prepare output using dataclasses (no data stored)
    fit_input = FitInput(
        model_name=model.__name__ if hasattr(model, '__name__') else str(model),
        prior_mean=prior.mean,
        prior_precision=prior.precision,        
        fname=fname
    )

    profile_info = FitProfile(
        datetime=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        filename='individual_fit',
        telapsed=t_elapsed,
        config=config.__dict__.copy(),  # dict avoids pickle class-identity issues
        prior_mean=prior.mean,
        prior_precision=prior.precision    
    )

    math_details = FitMath(
        loglik=loglik,
        parameters=parameters_list,
        hessian=hessian_list,
        lme=lme,
        hessian_inv_diag=hessian_inv_diag,
        log_det_hessian=log_det_hessian,
        flag=flags,
        gradient=G        
    )

    # Stack parameters into N×d matrix
    parameters_array = np.vstack(parameters_list)

    output = FitOutput(
        parameters=parameters_array,
        log_evidence=lme
    )

    cbm = FitResult(
        method='LAP individual',
        input=fit_input,
        profile=profile_info,
        math=math_details,
        output=output
    )

    # Save if filename provided
    if fname is not None:
        with open(fname, 'wb') as f:
            pickle.dump(cbm, f)

    if config.verbose:
        print("done :]")

    return cbm


# Example usage
if __name__ == "__main__":
    # Example: Simple linear model
    def linear_model(theta, data):
        """
        Simple linear model: y ~ N(X*theta, sigma^2)
        theta = [slope, intercept, log(sigma)]
        """
        X, y = data
        slope, intercept, log_sigma = theta

        y_pred = X * slope + intercept
        sigma = np.exp(log_sigma)

        # Log-likelihood of Gaussian
        log_lik = -0.5 * np.sum((y - y_pred) ** 2 / sigma ** 2) - len(y) * np.log(sigma * np.sqrt(2 * np.pi))

        return log_lik


    # Generate synthetic data for 5 subjects
    np.random.seed(42)
    N_subjects = 5
    data = []
    true_parameters = []

    X = np.linspace(0, 10, 50)
    noise = np.random.randn(len(X))
    print(X)

    for i in range(N_subjects):

        true_slope = 2.0 + np.random.randn() * 0.5
        true_intercept = 1.0 + np.random.randn() * 0.5
        true_log_sigma = np.log(0.5)

        y = true_slope * X + true_intercept + noise * np.exp(true_log_sigma)
        data.append((X, y))
        true_parameters.append([true_slope, true_intercept, true_log_sigma])

    true_parameters = np.array(true_parameters)

    # Define prior
    prior = Prior(
        mean=np.array([0, 0, 0]),  # slope, intercept, log(sigma)
        variance=np.array([10, 10, 10])  # prior variances
    )

    # Configure
    config = Config(
        d=3,
        num_init=20,
        tol_grad=1e-4,
        verbose=True
    )

    # Run individual fitting
    print("\n" + "=" * 70)
    print("Running Individual Fit Example")
    print("=" * 70 + "\n")

    cbm = individual_fit(data, linear_model, prior, fname=None, config=config)

    llp = log_posterior(cbm.output.parameters[0, :], linear_model, data[0], prior.mean, prior.precision)

    print("\n" + "=" * 70)
    print("Results")
    print("=" * 70)
    print(f"\nTrue parameters (N×d):")
    print(true_parameters)
    print(f"\nFitted parameters (N×d):")
    print(cbm.output.parameters)
    print(f"\nLog model evidence:\n{cbm.output.log_evidence}")
    print(f"\nLog model loglik:\n{cbm.math.loglik}")
    print(f"\nLog det hessian:\n{cbm.math.log_det_hessian}")

    print(f"\ndiff:\n{cbm.output.log_evidence - cbm.math.loglik}")

