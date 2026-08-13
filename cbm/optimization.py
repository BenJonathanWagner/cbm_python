import numpy as np
from scipy.optimize import minimize, approx_fprime
from typing import Callable, Optional, List
from dataclasses import dataclass
import warnings

@dataclass
class Config:
    """
    Configuration for individual fitting.

    These parameters match BFGSOptimizer configuration.

    Attributes:
        d: Dimension of parameters
        range_bounds: 2×d array for parameter ranges
        tol_grad: Tolerance for gradient
        tol_grad_liberal: Liberal tolerance for bad subjects
        num_init: Number of random initializations
        num_init_med: Increased number for bad subjects
        num_init_up: Maximum number for bad subjects
        inits: Optional custom initialization points (n_inits × d array)
        max_iter: Maximum iterations per optimization run
        prior_for_failed: Whether to use prior for subjects with no good fit
        verbose: Whether to print progress
        save_data: Whether to save data in output
    """
    d: Optional[int] = None
    range_bounds: Optional[int | np.ndarray] = 5
    hard_bounds: Optional[int | np.ndarray] = 100
    tol_grad: float = 0.001001
    tol_grad_liberal: float = 0.1
    num_init: Optional[int] = None
    num_init_med: Optional[int] = None
    num_init_up: Optional[int] = None
    inits: Optional[np.ndarray] = None
    max_iter: int = 1000
    prior_for_failed: bool = True
    verbose: bool = True
    save_data: bool = False

    def __post_init__(self):
        """Set defaults based on dimension."""
        if self.num_init is None:
            self.num_init = min(7 * self.d, 100)

        if self.num_init_med is None:
            self.num_init_med = self.num_init + 10
        elif self.num_init_med < self.num_init:
            raise ValueError("num_init_med must be >= num_init")

        if self.num_init_up is None:
            self.num_init_up = self.num_init_med + 10
        elif self.num_init_up < self.num_init_med:
            raise ValueError("num_init_up must be >= num_init_med")

        # if self.range_bounds is None:
        #     self.range_bounds = np.array([
        #         -5 * np.ones(self.d),
        #         5 * np.ones(self.d)
        #     ])
        # elif np.isscalar(self.range_bounds):
        #     self.range_bounds = np.array([
        #         -self.range_bounds * np.ones(self.d),
        #         self.range_bounds * np.ones(self.d)
        #     ])
        # else:
        #     if self.range_bounds.shape != (2, self.d):
        #         raise ValueError(f"range_bounds must be 2×{self.d} array, got shape {self.range_bounds.shape}")
        #     self.range_bounds = self.range_bounds
        
        # if self.hard_bounds is None:
        #     self.hard_bounds = np.array([
        #         -100 * np.ones(self.d),
        #         100 * np.ones(self.d)
        #     ])
        # elif np.isscalar(self.hard_bounds):
        #     self.hard_bounds = np.array([
        #         -self.hard_bounds * np.ones(self.d),
        #         self.hard_bounds * np.ones(self.d)
        #     ])    
        # else:
        #     if self.hard_bounds.shape != (2, self.d):
        #         raise ValueError(f"hard_bounds must be 2×{self.d} array, got shape {self.hard_bounds.shape}")
        #     self.hard_bounds = self.hard_bounds               

@dataclass
class OptimizationResult:
    """
    Result from BFGS optimization.

    Attributes:
        x: Optimized parameters (d-dimensional array)
        f: Optimal function value (scalar)
        hess: Hessian matrix at optimum (d × d array), computed via finite differences
              Can be None for intermediate results
        grad: Gradient at optimum (d-dimensional array)
        flag: Success flag (1.0=full success, 0.5=partial success, 0.0=failed)
        success: Boolean indicating if scipy optimization succeeded
        nit: Number of iterations in best run
        n_runs: Total number of optimization runs attempted
        is_hess_pos: Whether Hessian is positive definite
        abs_g: Mean absolute gradient at optimum
        x_init: Initial point used for the best run
    """
    x: np.ndarray
    f: float
    hess: Optional[np.ndarray]
    grad: np.ndarray
    flag: float
    success: bool
    nit: int
    n_runs: int
    is_hess_pos: bool
    abs_g: float
    x_init: np.ndarray


class BFGSOptimizer:
    """
    BFGS optimizer with multiple initializations and convergence criteria.
    The optimizer is configured at initialization and can be run multiple times
    with different functions.
    """

    def __init__(self,
                 d: int,
                 config: Config,
                 gtol: float = 1e-5,
                 ftol: float = 1e-9):
        """
        Initialize BFGS optimizer with configuration parameters.

        Args:
            config: Configuration object with optimization parameters
            gtol: Gradient tolerance for scipy optimizer
            ftol: Function tolerance for scipy optimizer
        """
        self.d = d
        self.tol_grad = config.tol_grad
        self.tol_grad_liberal = config.tol_grad_liberal
        self.num_init = config.num_init
        self.num_init_med = config.num_init_med
        self.num_init_up = config.num_init_up
        self.max_iter = config.max_iter
        self.range_bounds = config.range_bounds
        self.hard_bounds = config.hard_bounds
        self.inits = config.inits
        self.gtol = gtol
        self.ftol = ftol

        # # Store or generate initial points
        # if inits is None:
        #     self.inits = None  # Will generate random inits when optimize is called
        # else:
        #     if inits.shape[1] != d:
        #         raise ValueError(f"inits must have {d} columns, got shape {inits.shape}")
        #     self.inits = inits

        # History tracking
        self.history_x = []
        self.history_f = []
        self.all_results = []
    
        """Set defaults based on dimension."""
        if self.range_bounds is None:
            self.range_bounds = np.array([
                -5 * np.ones(self.d),
                5 * np.ones(self.d)
            ])
        elif np.isscalar(self.range_bounds):
            self.range_bounds = np.array([
                -self.range_bounds * np.ones(self.d),
                self.range_bounds * np.ones(self.d)
            ])
        
        if self.hard_bounds is None:
            self.hard_bounds = np.array([
                -100 * np.ones(self.d),
                100 * np.ones(self.d)
            ])
        elif np.isscalar(self.hard_bounds):
            self.hard_bounds = np.array([
                -self.hard_bounds * np.ones(self.d),
                self.hard_bounds * np.ones(self.d)
            ])    
        else:
            if self.hard_bounds.shape != (2, self.d):
                raise ValueError(f"hard_bounds must be 2×{self.d} array, got shape {self.hard_bounds.shape}")
            self.hard_bounds = self.hard_bounds               


    def compute_hessian(self,
                        func: Callable[[np.ndarray], float],
                        x: np.ndarray,
                        epsilon: float = 1e-5) -> np.ndarray:
        """
        Compute Hessian matrix via finite differences.

        Args:
            func: Objective function
            x: Point at which to compute Hessian
            epsilon: Step size for finite differences

        Returns:
            Hessian matrix (d × d)
        """
        n = len(x)
        H = np.zeros((n, n))

        # Compute gradient at x
        grad_x = approx_fprime(x, func, epsilon)

        # Compute gradient at x + epsilon*e_i for each dimension
        for i in range(n):
            x_step = x.copy()
            x_step[i] += epsilon
            grad_step = approx_fprime(x_step, func, epsilon)
            H[i, :] = (grad_step - grad_x) / epsilon

        # Symmetrize
        return (H + H.T) / 2

    def _single_optimization(self,
                             func: Callable[[np.ndarray], float],
                             x_init: np.ndarray) -> OptimizationResult:
        """
        Run a single optimization from given initial point.

        Args:
            func: Objective function
            x_init: Initial point

        Returns:
            OptimizationResult with hess=None (computed later for best run only)
        """
        # Track function evaluations for this run
        run_history_x = []
        run_history_f = []

        def func_wrapper(x):
            f = func(x)
            run_history_x.append(x.copy())
            run_history_f.append(f)
            return f

        # Convert range_bounds to scipy bounds format
        bounds = [(self.hard_bounds[0, i], self.hard_bounds[1, i])
                  for i in range(self.d)]

        # Run L-BFGS-B optimizer
        result = minimize(
            func_wrapper,
            x_init,
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': self.max_iter,
                'gtol': self.gtol,
                'ftol': self.ftol,
                'disp': False
            }
        )

        # Extract results
        x_opt = result.x
        f_opt = result.fun

        # Compute gradient at optimum using finite differences
        epsilon = 1e-8
        grad = approx_fprime(x_opt, func, epsilon)

        # Check if inverse Hessian from L-BFGS is positive definite
        # This is cheap and good enough for selecting best run
        try:
            # Convert to dense if needed
            if hasattr(result.hess_inv, 'todense'):
                hess_inv_dense = result.hess_inv.todense()
            elif hasattr(result.hess_inv, 'matvec'):
                n = self.d
                hess_inv_dense = np.zeros((n, n))
                for i in range(n):
                    e = np.zeros(n)
                    e[i] = 1.0
                    hess_inv_dense[:, i] = result.hess_inv.matvec(e)
            else:
                hess_inv_dense = result.hess_inv

            # Check if positive definite
            np.linalg.cholesky(hess_inv_dense)
            is_hess_pos = True
        except (np.linalg.LinAlgError, AttributeError):
            is_hess_pos = False

        # Compute mean absolute gradient
        abs_g = np.mean(np.abs(grad))

        # Store history temporarily (not in OptimizationResult)
        self._temp_history_x = run_history_x
        self._temp_history_f = run_history_f

        return OptimizationResult(
            x=x_opt,
            f=f_opt,
            hess=None,  # Computed later for best run only
            grad=grad,
            flag=0.0,  # Computed later
            success=result.success,
            nit=result.nit,
            n_runs=1,  # Single run
            is_hess_pos=is_hess_pos,
            abs_g=abs_g,
            x_init=x_init.copy()
        )

    def optimize(self,
                 func: Callable[[np.ndarray], float],
                 x_init: Optional[np.ndarray] = None) -> OptimizationResult:
        """
        Optimize the given function.

        If x_init is provided, uses it PLUS num_init random initializations.
        Otherwise, uses only num_init random initializations.

        The number of initializations adapts based on convergence quality:
        - If flag=1.0: Uses num_init initializations
        - If flag=0.5: Tries up to num_init_med initializations
        - If flag=0.0: Tries up to num_init_up initializations

        Args:
            func: Objective function that takes x (numpy array of length d) and returns scalar
            x_init: Optional initial point (length d array). If provided, will be used in addition to random starts

        Returns:
            OptimizationResult dataclass with all optimization results
        """
        self.all_results = []

        # Determine initial number of attempts
        n_attempts = self.num_init

        while True:
            # Generate list of initial points for this round
            init_points = []

            # Add user-provided initial point if given (only on first iteration)
            if x_init is not None and len(self.all_results) == 0:
                if len(x_init) != self.d:
                    raise ValueError(f"x_init must have length {self.d}, got {len(x_init)}")
                init_points.append(x_init)

            # Add user-provided initial point given through config
            if self.inits is not None and len(self.all_results) == 0:
                if len(self.inits) != self.d:
                    raise ValueError(f"inits must have length {self.d}, got {len(self.inits)}")
                init_points.append(self.inits)

            # Determine how many random inits to add this round
            n_random = n_attempts - len(self.all_results)

            # Fill remaining with random initializations
            n_needed = n_random - (len(init_points) - (1 if x_init is not None and len(self.all_results) == 0 else 0))
            if n_needed > 0:
                random_inits = np.random.uniform(
                    low=self.range_bounds[0, :],
                    high=self.range_bounds[1, :],
                    size=(n_needed, self.d)
                )
                for init_pt in random_inits:
                    init_points.append(init_pt)

            # Run optimization from each initial point
            best_f = np.inf
            best_result = None
            best_history_x = []
            best_history_f = []

            for i, x0 in enumerate(init_points):
                result = self._single_optimization(func, x0)
                self.all_results.append(result)

                # Keep track of best result (lowest function value).
                # Use `best_result is None` as the first-run guard so that NaN
                # f-values (which make NaN < inf == False) don't leave
                # best_result as None and crash the Hessian computation below.
                if best_result is None or (np.isfinite(result.f) and result.f < best_f):
                    best_f = result.f
                    best_result = result
                    best_history_x = self._temp_history_x
                    best_history_f = self._temp_history_f

            # Store history from best run
            self.history_x = best_history_x
            self.history_f = best_history_f

            # Compute Hessian only for the best result using finite differences
            hess = self.compute_hessian(func, best_result.x, epsilon=1e-5)

            # Re-check if Hessian is positive definite (using actual Hessian this time)
            try:
                np.linalg.cholesky(hess)
                is_hess_pos = True
            except np.linalg.LinAlgError:
                is_hess_pos = False

            # Determine flag based on convergence criteria
            flag = 0.0

            if best_result.success and is_hess_pos and (best_result.abs_g < self.tol_grad):
                flag = 1.0  # Full success
            elif is_hess_pos and (best_result.abs_g < self.tol_grad_liberal):
                flag = 0.5  # Partial success
            else:
                flag = 0.0  # Failed

            # Check if we need more attempts
            if flag == 1.0:
                # Success! We're done
                break
            elif flag == 0.5 and len(self.all_results) < self.num_init_med:
                # Partial success, try more initializations
                n_attempts = self.num_init_med
                continue
            elif flag == 0.0 and len(self.all_results) < self.num_init_up:
                # Failed, try even more initializations
                n_attempts = self.num_init_up
                continue
            else:
                # We've tried enough, stop here
                break

        # Throw warnings based on final flag
        if flag == 0.0:
            warnings.warn(f"--- No positive hessian found in spite of {len(self.all_results)} initialization.")
        elif flag == 0.5:
            warnings.warn(
                f"Positive hessian found, but not a good gradient in spite of {len(self.all_results)} initialization.")

        return OptimizationResult(
            x=best_result.x,
            f=best_result.f,
            hess=hess,
            grad=best_result.grad,
            flag=flag,
            success=best_result.success,
            nit=best_result.nit,
            n_runs=len(self.all_results),
            is_hess_pos=is_hess_pos,
            abs_g=best_result.abs_g,
            x_init=best_result.x_init
        )

    def get_all_results(self) -> List[OptimizationResult]:
        """
        Get detailed results from all optimization runs.

        Returns:
            List of OptimizationResult objects, one for each run (with hess=None)
        """
        return self.all_results

    def get_history(self) -> tuple:
        """
        Get optimization history from the best run.

        Returns:
            Tuple of (history_x, history_f) where:
                - history_x: List of x values tried
                - history_f: List of function values
        """
        return self.history_x, self.history_f


# Example usage
if __name__ == "__main__":
    # Define test function (Rosenbrock)
    def rosenbrock(x):
        """Rosenbrock function"""
        return np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2)


    # Create optimizer for 4-dimensional problem
    optimizer = BFGSOptimizer(
        d=4,
        range_bounds=np.array([[-2, -2, -2, -2], [2, 2, 2, 2]]),
        num_init=10
    )

    print("=" * 70)
    print("Test 1: Multiple Random Initializations (no x_init provided)")
    print("=" * 70)

    # Optimize without providing initial point (uses num_init random starts)
    result = optimizer.optimize(rosenbrock)

    print(f"Optimal x: {result.x}")
    print(f"Optimal f: {result.f:.6e}")
    print(f"Flag: {result.flag} (1.0=full success, 0.5=partial, 0.0=failed)")
    print(f"Success: {result.success}")
    print(f"Mean |grad|: {result.abs_g:.6e}")
    print(f"Number of runs: {result.n_runs}")
    print(f"Iterations (best run): {result.nit}")
    print(f"Hessian positive definite: {result.is_hess_pos}")
    print(f"Initial point of best run: {result.x_init}")

    print("\nHessian matrix:")
    print(result.hess)
    print(f"Condition number: {np.linalg.cond(result.hess):.2e}")
    print(f"Determinant: {np.linalg.det(result.hess):.2e}")

    print("\n" + "=" * 70)
    print("Test 2: With Provided Initial Point (x_init + num_init random)")
    print("=" * 70)

    # Optimize with specific initial point PLUS random initializations
    x_init = np.array([0.5, 0.5, 0.5, 0.5])
    result2 = optimizer.optimize(rosenbrock, x_init=x_init)

    print(f"Provided x_init: {x_init}")
    print(f"Total runs: {result2.n_runs} (1 from x_init + {result2.n_runs - 1} random)")
    print(f"Optimal x: {result2.x}")
    print(f"Optimal f: {result2.f:.6e}")
    print(f"Flag: {result2.flag}")
    print(f"Mean |grad|: {result2.abs_g:.6e}")
    print(f"Initial point of best run: {result2.x_init}")

    print("\nHessian matrix:")
    print(result2.hess)
    print(f"Condition number: {np.linalg.cond(result2.hess):.2e}")

    # Get optimization history
    history_x, history_f = optimizer.get_history()
    print(f"\nOptimization trajectory: {len(history_f)} function evaluations")
    print(f"Function value progress: {history_f[0]:.3e} -> {history_f[-1]:.3e}")

    print("\n" + "=" * 70)
    print("Accessing result fields")
    print("=" * 70)
    print(f"result.x = {result.x}")
    print(f"result.f = {result.f:.6e}")
    print(f"result.flag = {result.flag}")
    print(f"result.hess = {result.hess}")