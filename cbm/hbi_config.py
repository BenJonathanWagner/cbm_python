import os
import time
import math
import inspect
from dataclasses import dataclass, field
from typing import Optional, Union
from typing import Optional, Union, List, Tuple, Any #added


def _default_fname() -> str:
    
    t = time.time()
    # Get the directory of the calling script (not cwd)
    frame = inspect.currentframe()
    cbm_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        # Walk up the stack to find the first frame outside the cbm package
        current = frame
        while current is not None:
            frame_file = current.f_globals.get('__file__')
            if frame_file:
                frame_dir = os.path.dirname(os.path.abspath(frame_file))
                # Found a frame outside the cbm package
                if not frame_dir.startswith(cbm_dir):
                    return os.path.join(frame_dir, f"cbm_hbi_{t:0.4f}.pkl")
            current = current.f_back
    finally:
        del frame
    # Fallback to current directory if we can't determine caller
    return f"cbm_hbi_{t:0.4f}.pkl"


def _valid_fname(arg: Optional[str]) -> bool:
    """
    valid_fname:
      - empty or None → valid
      - otherwise directory must exist and extension must be '.pkl'
    """
    if arg is None or arg == "":
        return True
    try:
        fdir, fname = os.path.split(arg)
        _, fext = os.path.splitext(fname)
        if fdir == "":
            fdir = "."
        return os.path.isdir(fdir) and fext == ".pkl"
    except Exception:
        return False


def _valid_flog(arg: Optional[Union[str, int]]) -> bool:

    if arg is None or arg == "":
        return True
    if isinstance(arg, int) and arg == -1:
        return True
    if isinstance(arg, int) and arg == 1:
        return True
    if isinstance(arg, str):
        fdir, _ = os.path.split(arg)
        if fdir == "":
            fdir = "."
        return os.path.isdir(fdir)
    return False


@dataclass
class HBIConfig:

    verbose: int = 1
    covariance_blocks: Optional[List[List[Tuple[int, int]]]] = None # new switch
    fname_prog: Optional[str] = field(default_factory=_default_fname)
    flog: Optional[Union[str, int]] = None
    save_prog: int = 0
    initialize: str = "all_r_1"
    maxiter: int = 50
    tolx: float = 0.01
    tolL: float = -math.log(0.5)

    # -----------------------------------------------------------------
    # auto_nu0 / nu0_margin: opt-in auto-scaling of the Wishart hyperprior
    # dof for joint (covariance_blocks) models.
    #
    # The hardcoded default hyperprior dof is nu0=0.5 for every model,
    # regardless of how many parameters get linked into one covariance
    # block. A Wishart prior is only proper for a block of size Db when
    # nu0 > (Db-1)/2 -- so nu0=0.5 is already right at that boundary for
    # any 2-parameter block, and well past it for anything bigger (e.g. a
    # "full" within-task covariance_blocks selection). In practice this
    # doesn't corrupt the fitted covariance/correlation (nu0 cancels out
    # of the reported correlation, and the posterior dof nu_k=nu0+0.5*Nk
    # is what's actually used downstream), but it does silently zero out
    # the Wishart prior's normalizing term in the ELBO for any block that
    # crosses that boundary -- which matters if you compare log-evidence
    # (L) or exceedance probabilities across joint-vs-non-joint models.
    #
    # When auto_nu0=True, nu0 for a model is raised (never lowered) to
    # nu0 = max(nu0_default, (Db_max + nu0_margin) / 2), where Db_max is
    # the size of that model's LARGEST connected covariance_blocks
    # component. nu0_margin=1.0 (default) gives dof0 = Db_max + 1, a
    # standard "just proper, weakly informative" choice. nu0 stays a
    # single scalar per model either way (same as the current, unscaled
    # behaviour) -- only its VALUE changes, so nothing downstream that
    # reads qmutau[k].nu as a plain float is affected.
    #
    # Default False: existing fits/results are completely unaffected
    # unless you opt in.
    auto_nu0: bool = False
    nu0_margin: float = 1.0

    # recenter_a0: opt-in empirical-Bayes recentering of the group-mean
    # hyperprior. The Normal-Wishart scale update contains the term
    #   (Nk*beta0/beta_k) * (thetabar - a0)(thetabar - a0)^T
    # whose OFF-DIAGONAL entries are (tb_i - a0_i)(tb_j - a0_j). When the
    # true group mean sits far from the prior mean a0 in the SAME direction
    # for two linked parameters (common on real data, where a0 comes from
    # generic individual-fit priors), this injects spurious positive
    # covariance into exactly those linked pairs -- small at large N
    # (~beta0*diff_i*diff_j vs Nk*S_ij) but material at small N.
    #
    # With recenter_a0=True, a0 for each model is replaced at
    # initialization by the mean of that model's individual-fit MAP
    # estimates (finite subjects only), so diff stays near zero and the
    # term contributes almost nothing. This is standard empirical Bayes;
    # it only changes the hyperprior mean, never the update equations.
    # Default False: existing behaviour unchanged.
    recenter_a0: bool = False

    # ---------------------------------------------------------------------
    # ELBO divergence guard (keep_best_iterate / divergence_tol)
    # ---------------------------------------------------------------------
    # In exact variational EM the bound L is non-decreasing, so a large
    # NEGATIVE dL means something has gone wrong -- in practice, a single
    # subject whose Laplace refit finds no positive-definite Hessian
    # ("No positive hessian found in spite of N initialization"), which
    # makes its covariance block near-singular and blows up the bound's
    # slogdet/multigammaln terms. Once that happens the run can enter a
    # limit cycle: L swings by orders of magnitude every iteration while
    # the parameters barely move (dx stays tiny), and it never recovers.
    # This is most likely with large merged covariance blocks
    # (covariance_blocks linking many parameters) at moderate N.
    #
    # keep_best_iterate: on exit, restore the state (qmutau, qhquad, r, qm,
    #   bound, ...) from the iteration with the HIGHEST finite L, instead of
    #   returning whatever the last iteration happened to be. If a run
    #   converged cleanly and then blew up at iteration 18, this returns the
    #   converged iteration-17 solution rather than an arbitrary point in
    #   the post-divergence cycle. Costs nothing extra: the per-iteration
    #   snapshots are already retained.
    #
    # divergence_tol: if set to a positive number, terminate early as soon
    #   as dL < -divergence_tol. Since L should never decrease materially,
    #   this is a direct "the bound has broken" test rather than a
    #   heuristic. Something like 10-100 is reasonable; None disables it.
    #   Best used together with keep_best_iterate, so the run both stops
    #   early AND returns the last good state.
    #
    # Both default to off, so existing fits are unaffected.
    keep_best_iterate: bool = False
    divergence_tol: Optional[float] = None

    def __post_init__(self):
        # -----------------------
        # verbose
        # -----------------------
        if not isinstance(self.verbose, int):
            raise ValueError("verbose must be an integer")

        # -----------------------
        # save_prog (logical)
        # -----------------------
        self.save_prog = int(bool(self.save_prog))
        if self.save_prog not in (0, 1):
            raise ValueError("save_prog must be 0 or 1")

        # -----------------------
        # initialize ∈ {'all_r_1','cluster_r'}
        # -----------------------
        if self.initialize not in ("all_r_1", "cluster_r", "lme_softmax"):
            raise ValueError(
                "initialize must be 'all_r_1', 'cluster_r' or 'lme_softmax' "
                "('lme_softmax' seeds responsibilities from the individual "
                "fits' log model evidence -- required for population-mixture "
                "setups where K models share a likelihood and differ only in "
                "their priors; 'cluster_r' is accepted for compatibility but "
                "not implemented in this port)")

        # -----------------------
        # maxiter integer
        # -----------------------
        if not isinstance(self.maxiter, int):
            raise ValueError("maxiter must be an integer")

        # -----------------------
        # tolx, tolL scalar numerics
        # -----------------------
        if not isinstance(self.tolx, (int, float)):
            raise ValueError("tolx must be a scalar number")
        if not isinstance(self.tolL, (int, float)):
            raise ValueError("tolL must be a scalar number")

        # -----------------------
        # fname_prog validity
        # -----------------------
        if not _valid_fname(self.fname_prog):
            raise ValueError(f"Invalid fname_prog: {self.fname_prog}")

        # -----------------------
        # flog validity
        # -----------------------
        if not _valid_flog(self.flog):
            raise ValueError(f"Invalid flog: {self.flog}")

        # -----------------------
        if self.save_prog == 0:
            self.fname_prog = None

        # -----------------------
        # auto_nu0 / nu0_margin
        # -----------------------
        if not isinstance(self.auto_nu0, (bool, int)):
            raise ValueError("auto_nu0 must be a bool")
        self.auto_nu0 = bool(self.auto_nu0)
        if not isinstance(self.nu0_margin, (int, float)):
            raise ValueError("nu0_margin must be a scalar number")
        if self.nu0_margin <= 0:
            raise ValueError("nu0_margin must be positive")

        # -----------------------
        # recenter_a0
        # -----------------------
        if not isinstance(self.recenter_a0, (bool, int)):
            raise ValueError("recenter_a0 must be a bool")
        self.recenter_a0 = bool(self.recenter_a0)

        # -----------------------
        # keep_best_iterate / divergence_tol
        # -----------------------
        if not isinstance(self.keep_best_iterate, (bool, int)):
            raise ValueError("keep_best_iterate must be a bool")
        self.keep_best_iterate = bool(self.keep_best_iterate)
        if self.divergence_tol is not None:
            if not isinstance(self.divergence_tol, (int, float)):
                raise ValueError("divergence_tol must be a scalar number or None")
            if self.divergence_tol <= 0:
                raise ValueError("divergence_tol must be positive (or None to disable)")
            self.divergence_tol = float(self.divergence_tol)