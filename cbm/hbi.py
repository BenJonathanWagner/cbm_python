import os
import pickle
import warnings
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import numpy as np
from scipy.special import psi, gammaln

from cbm.hbi_exceedance import cbm_hbi_exceedance
from .hbi_types import (
    IndividualPosterior,
    ProgressChange,
    ProgressState,
    GaussianGammaDistribution,
    DirichletDistribution,
    BoundTerms,
    BoundQHZ,
    BoundQMutau,
    BoundQM,
    BoundState,
    HBIInput,
    HBIProfile,
    HBIMath,
    HBIOutput,
    HBIResult,
)

from .hbi_config import HBIConfig
from .hbi_updates import hbi_sumstats, hbi_qmutau, hbi_qm, hbi_qHZ, hbi_qhquad, hbi_bound
from .hbi_logging import hbi_log, log_header, log_iteration
__all__ = ["hbi_run", "hbi_init", "hbi_null", "HBIResult"]

# Private convergence helper (not exported)
def _hbi_prog(
    prog: List[ProgressState],
    L: float,
    alpha: np.ndarray,
    thetabar: List[np.ndarray],
    Sdiag: List[np.ndarray],
) -> Tuple[ProgressChange, List[ProgressState]]:
    last = prog[-1]
    L_pre = float(last.bound)
    alpha_pre = np.asarray(last.model_freq)
    x_pre = last.normalized_params

    thetabar_vec = np.concatenate([tb.ravel() for tb in thetabar])
    
    # --- NEW: Extract the diagonal if it's a 2D matrix! ---
    Sdiag_vec = np.concatenate([np.diag(sd) if sd.ndim == 2 else sd.ravel() for sd in Sdiag])
    
    x = thetabar_vec / np.sqrt(np.maximum(np.abs(Sdiag_vec), 1e-10))  # guard zero-variance from collapsed models

    dx = np.sqrt(np.mean((x - x_pre) ** 2))
    dL = float(L - L_pre)

    ibest = int(np.argmax(alpha))
    if np.isinf(alpha[ibest]):
        dalpha = np.nan
    else:
        dalpha = float(abs(alpha[ibest] - alpha_pre[ibest]))

    prog_change = ProgressChange(change_bound=dL, change_model_freq=dalpha, change_parameters=float(dx))
    prog.append(ProgressState(bound=float(L), model_freq=alpha.copy(), normalized_params=x))

    return prog_change, prog

def hbi_main(data: List[Any], models: List[Any], fcbm_maps: List[str], fname: str = "", config: Union[HBIConfig, Dict[str, Any]] = None, optimconfigs: List[Any] = None) -> HBIResult:
    """
    Main function to run HBI.

    Parameters
    ----------
    data : list
        List of subject-level data objects.
    models : list
        List of model functions.
    fcbm_maps : list
        List of file paths or dicts for CBM maps.
    config : HBIConfig or dict
        Configuration for HBI.
    optimconfigs : list
        List of optimization configurations for each model.
    fname : str, optional
        Filename to save the resulting CBM object, by default "".

    Returns
    -------
    HBIResult
        The result of the HBI run.
    """
    user_input = {
        "models": models,
        "fcbm_maps": fcbm_maps,
        "fname": fname,
        "config": config,
        "optimconfigs": optimconfigs,
    }

    # Hyper (prior) parameters
    b = 1.0
    v = 0.5
    s = 0.01
    hyper = {"b": b, "v": v, "s": s}

    # Initialize HBI. The responsibility initialization comes from
    # config.initialize (HBIConfig validates the allowed values); previously
    # this was hard-coded to 'all_r_1' and the config field was silently
    # ignored.
    _init_r = 'all_r_1'
    if config is not None:
        if isinstance(config, HBIConfig):
            _init_r = getattr(config, 'initialize', 'all_r_1') or 'all_r_1'
        elif isinstance(config, dict):
            _init_r = config.get('initialize', 'all_r_1') or 'all_r_1'
    inits, priors, opt_configs = hbi_init(
        fcbm_maps,
        hyper,
        limInf=0,
        initialize_r=_init_r,
    )

    # Optional empirical-Bayes recentering of the group-mean hyperprior
    # (config.recenter_a0): replace each model's a0 with the mean of its
    # individual-fit MAP estimates. This neutralizes the
    # (thetabar - a0)(thetabar - a0)^T term of the Normal-Wishart scale
    # update, whose off-diagonals otherwise inject spurious covariance
    # into linked parameter pairs when a0 is badly mis-centered (see
    # HBIConfig.recenter_a0). Only the hyperprior MEAN changes; all update
    # equations and every other hyperparameter stay untouched.
    _recenter = False
    if config is not None:
        if isinstance(config, HBIConfig):
            _recenter = bool(getattr(config, 'recenter_a0', False))
        elif isinstance(config, dict):
            _recenter = bool(config.get('recenter_a0', False))
    if _recenter:
        for k, _pmt in enumerate(priors["pmutau"]):
            _theta_k = np.asarray(inits["qh"].parameters[k], dtype=float)  # (D, N)
            _finite = np.all(np.isfinite(_theta_k), axis=0)
            if _finite.sum() >= 2:
                _pmt.a = _theta_k[:, _finite].mean(axis=1)

    # Run HBI
    cbm = hbi_run(data, user_input, inits, priors, opt_configs)
    return cbm

def hbi_run(data: List[Any], user_input: Dict[str, Any], inits: Dict[str, Any], priors: Dict[str, Any], opt_configs: List[Any]) -> HBIResult:
    models = user_input["models"]
    fcbm_maps = user_input["fcbm_maps"]
    fname = user_input.get("fname", None)
    config_in = user_input["config"]
    optconfigs_in = opt_configs

    K = len(models)
    N = len(data)

    qhquad = inits["qh"]
    r = np.asarray(inits["r"], dtype=float)
    bound = deepcopy(inits["bound"])

    hyper = priors["hyper"]
    pmutau = deepcopy(priors["pmutau"]) if isinstance(priors["pmutau"], list) else [deepcopy(priors["pmutau"])]
    pm = deepcopy(priors["pm"])

    isnull = bool(pm.limInf) if isinstance(pm, DirichletDistribution) else bool(pm["limInf"]) 

    if isinstance(config_in, HBIConfig):
        config = config_in
    else:
        config = HBIConfig(**config_in)

    flog = config.flog
    fname_prog = config.fname_prog
    save_prog = bool(config.save_prog)
    verbose = bool(config.verbose)
    maxiter = config.maxiter
    tolx = config.tolx
    tolL = config.tolL

    if (flog is None or flog == "") and fname:
        fdir, fn = os.path.split(fname)
        if fdir == "":
            fdir = "."
        flog = os.path.join(fdir, f"{os.path.splitext(fn)[0]}.log")

    fid_file = None
    if flog != -1 and isinstance(flog, str) and flog != "":
        fid_file = open(flog, "w")
    fid = fid_file

    verbose_multiK = bool(verbose and (K > 1) and (not isnull))
    fid_multiK = fid if (K > 1 and not isnull) else None

    optconfigs = []
    for k in range(K):
        d = len(pmutau[k].a) if isinstance(pmutau[k], GaussianGammaDistribution) else len(pmutau[k]["a"])
        optfigk = {}
        if len(optconfigs_in) > 0:
            optfigk = deepcopy(optconfigs_in[k])
        optfigk.num_init = 0
        optfigk.num_init_med = 0
        optfigk.num_init_up = 3
        optfigk.verbose = False
        optconfigs.append(optfigk)

    log_header(verbose, fid, K, N, fcbm_maps, isnull)

    prog = [
        ProgressState(
            bound=bound.bound.L,
            model_freq=np.asarray(pm.alpha, dtype=float) if isinstance(pm, DirichletDistribution) else np.asarray(pm["alpha"], dtype=float),
            normalized_params=np.nan,
        )
    ]

    terminate = False
    it = 0
    math_list: List[Dict[str, Any]] = []
    
    #added a block here
    
    auto_nu0 = bool(getattr(config, 'auto_nu0', False))
    nu0_margin = float(getattr(config, 'nu0_margin', 1.0))

    # ELBO divergence guard (see HBIConfig.keep_best_iterate / divergence_tol)
    keep_best_iterate = bool(getattr(config, 'keep_best_iterate', False))
    divergence_tol = getattr(config, 'divergence_tol', None)
    divergence_tol = None if divergence_tol is None else float(divergence_tol)
    diverged_at = None   # iteration index where the bound broke, if it did

    covariance_mask = None
    if hasattr(config, 'covariance_blocks') and config.covariance_blocks is not None:
        covariance_mask = [None] * K
        for k in range(K):
            if k < len(config.covariance_blocks) and config.covariance_blocks[k] is not None:
                # Find number of parameters for this specific model
                Dk = len(pmutau[k].a) if isinstance(pmutau[k], GaussianGammaDistribution) else len(pmutau[k]["a"])
                
                # Start with a diagonal of 1s (independent variances)
                mask_k = np.eye(Dk, dtype=float)
                
                # Add 1s symmetrically for any user-linked parameters
                for (i, j) in config.covariance_blocks[k]:
                    mask_k[i, j] = 1.0
                    mask_k[j, i] = 1.0
                    
                covariance_mask[k] = mask_k

    # -----------------------------------------------------------------------
    # recenter_a0 is not cosmetic once a covariance block is in play. The
    # (thetabar - a0)(thetabar - a0)^T term of the Normal-Wishart scale update
    # is RANK ONE, so its own correlation is exactly +1 whenever two linked
    # parameters' group means sit on the same side of a0 -- the usual case,
    # since a0 comes from generic individual-fit priors. It therefore inflates
    # the estimated correlation between exactly the pairs the user asked about.
    #
    # Measured over 20 paired replications (N=80, true r=0.5, one dataset fit
    # twice): bias +0.069 (t=4.3, p=0.0004) with recenter_a0=False, versus
    # -0.007 (t=-0.4) with it on; RMSE 0.098 vs 0.079. See mc_recenter_a0.py.
    #
    # The default is left at False so that existing scripts keep their
    # behaviour, so this warns instead. Only fires when the mask actually links
    # something -- an all-identity mask has no off-diagonal to corrupt.
    # -----------------------------------------------------------------------
    if covariance_mask is not None and not bool(getattr(config, 'recenter_a0', False)):
        _links = any(
            m is not None and np.any(np.asarray(m) - np.diag(np.diag(np.asarray(m))) != 0)
            for m in covariance_mask
        )
        if _links:
            _msg = (
                "covariance_blocks links at least one parameter pair but "
                "recenter_a0=False. The (thetabar - a0)(thetabar - a0)^T term of "
                "the scale update is rank one and inflates the correlation "
                "between linked parameters whenever their group means sit on the "
                "same side of a0. Measured bias +0.069 (t=4.3) with it off vs "
                "-0.007 (t=-0.4) with it on. Set recenter_a0=True unless you "
                "have a specific reason not to."
            )
            warnings.warn(_msg, RuntimeWarning, stacklevel=2)
            hbi_log(verbose, fid, "  WARNING: " + _msg + "\n")

    while not terminate and it <= maxiter:
        it += 1
        hbi_log(verbose, fid, f"Iteration {it:02d}\n")
        
        # ---> ADD THE MASK ARGUMENTS HERE <---
        Nbar, thetabar, Sdiag = hbi_sumstats(r, qhquad, covariance_mask=covariance_mask)
        qmutau, bound_qmutau = hbi_qmutau(
            pmutau, Nbar, thetabar, Sdiag,
            covariance_mask=covariance_mask,
            auto_nu0=auto_nu0,
            nu0_margin=nu0_margin,
        )
        
        bound.qmutau = bound_qmutau
        bound, _ = hbi_bound(bound, "qmutau")
        qm, bound_qm = hbi_qm(pm, Nbar)
        bound.qm = bound_qm
        bound, _ = hbi_bound(bound, "qm")
        qhquad = hbi_qhquad(models, data, optconfigs, qmutau, qhquad, fid)
        
        ##
    
        r, bound_qHZ = hbi_qHZ(qmutau, qm, qhquad, thetabar, Sdiag, covariance_mask=covariance_mask)
        
        ##
        
        bound.qHZ = bound_qHZ
        bound, _ = hbi_bound(bound, "qHZ")
        prog_change, prog = _hbi_prog(prog, bound.bound.L, qm.alpha, thetabar, Sdiag)
        if prog_change.change_parameters < tolx:
            terminate = True
        # Also converge when the ELBO itself stops moving (finite dL only)
        if it > 2 and np.isfinite(prog_change.change_bound) and abs(prog_change.change_bound) < tolL:
            terminate = True
        # Divergence guard: in exact variational EM the bound is
        # non-decreasing, so a large negative dL means the bound has broken
        # (typically one subject's Laplace refit losing positive-definiteness,
        # which makes its covariance block near-singular). Continuing from
        # here usually produces a limit cycle rather than a recovery, so stop
        # and -- with keep_best_iterate -- fall back to the best iterate.
        _dL = prog_change.change_bound
        if divergence_tol is not None and it > 1:
            if (not np.isfinite(_dL)) or (_dL < -divergence_tol):
                diverged_at = it
                terminate = True
                hbi_log(verbose, fid,
                        f"  ELBO divergence detected at iteration {it:02d} "
                        f"(dL={_dL:.2f} < -{divergence_tol:g}); stopping early.\n")
        if it > 1:
            log_iteration(verbose, fid, verbose_multiK, fid_multiK, it, Nbar, N, prog_change, terminate, K)
        math_iter = {
            "qhquad": deepcopy(qhquad),
            "r": r.copy(),
            "Nbar": Nbar.copy(),
            "thetabar": [tb.copy() for tb in thetabar],
            "Sdiag": [sd.copy() for sd in Sdiag],
            "pm": deepcopy(pm),
            "pmutau": deepcopy(pmutau),
            "qm": deepcopy(qm),
            "qmutau": deepcopy(qmutau),
            "bound": deepcopy(bound),
            "prog": deepcopy(prog),
            "prog_change": deepcopy(prog_change),
            "input": deepcopy(user_input),
            "hyper": deepcopy(hyper),
        }
        math_list.append(math_iter)
        if save_prog and fname_prog:
            with open(fname_prog, "wb") as f:
                pickle.dump(math_list, f)

    # -----------------------------------------------------------------------
    # keep_best_iterate: return the state from the iteration with the highest
    # finite bound, instead of whatever the last iteration happened to be.
    # The per-iteration snapshots in math_list are already full deepcopies, so
    # this is a pure selection step -- no extra copying, and it is a no-op
    # when the run converged normally (the last iteration IS the best one).
    # -----------------------------------------------------------------------
    if keep_best_iterate and len(math_list) > 0:
        _Ls = np.array([
            float(m["bound"].bound.L) if np.isfinite(np.float64(m["bound"].bound.L)) else -np.inf
            for m in math_list
        ], dtype=float)
        # IMPORTANT: do not take a global argmax. Once the bound breaks it can
        # swing in BOTH directions (the observed failure mode is a limit cycle
        # alternating by +-1e6), so the largest L in the whole run may itself be
        # a corrupted value. The trustworthy region is the initial prefix over
        # which the bound never dropped materially -- variational EM guarantees
        # a non-decreasing bound, so the first material drop marks the break.
        # A break is either (a) a material DROP in the bound, or (b) a jump of
        # either sign that is wildly out of scale with the run's own history --
        # the first corrupted iteration often sends L sharply UP, so a
        # drop-only rule would happily select it. In a healthy EM run |dL|
        # shrinks monotonically toward zero, so it never exceeds a large
        # multiple of the largest |dL| seen so far.
        _break_tol = divergence_tol if divergence_tol is not None else max(1.0, 10.0 * float(tolL))
        _BREAK_FACTOR = 100.0
        _cut = len(_Ls)
        _run_max = None
        for _i in range(1, len(_Ls)):
            _d = _Ls[_i] - _Ls[_i - 1]
            if (not np.isfinite(_Ls[_i])) or (not np.isfinite(_d)):
                _cut = _i
                break
            if _d < -_break_tol:
                _cut = _i
                break
            if _run_max is not None and abs(_d) > _BREAK_FACTOR * max(_run_max, _break_tol):
                _cut = _i
                break
            _run_max = abs(_d) if _run_max is None else max(_run_max, abs(_d))
        _Ls_pref = _Ls[:_cut]
        if len(_Ls_pref) > 0 and np.any(np.isfinite(_Ls_pref)):
            _best = int(np.nanargmax(_Ls_pref))
            if _best != len(math_list) - 1:
                _bm = math_list[_best]
                qhquad = _bm["qhquad"]
                r       = _bm["r"]
                Nbar    = _bm["Nbar"]
                thetabar = _bm["thetabar"]
                Sdiag   = _bm["Sdiag"]
                pm      = _bm["pm"]
                pmutau  = _bm["pmutau"]
                qm      = _bm["qm"]
                qmutau  = _bm["qmutau"]
                bound   = _bm["bound"]
                prog    = _bm["prog"]
                hbi_log(verbose, fid,
                        f"  keep_best_iterate: returning iteration {_best + 1:02d} "
                        f"(L={_Ls[_best]:.2f}) instead of the final iteration "
                        f"{len(math_list):02d} (L={_Ls[-1]:.2f}); the bound first "
                        f"broke at iteration {_cut + 1:02d}.\n"
                        if _cut < len(_Ls) else
                        f"  keep_best_iterate: returning iteration {_best + 1:02d} "
                        f"(L={_Ls[_best]:.2f}) instead of the final iteration "
                        f"{len(math_list):02d} (L={_Ls[-1]:.2f}).\n")
            else:
                hbi_log(verbose, fid,
                        "  keep_best_iterate: final iteration was already the best; "
                        "nothing to restore.\n")
        else:
            hbi_log(verbose, fid,
                    "  keep_best_iterate: no finite bound in any iteration; "
                    "returning the final state unchanged.\n")

    if diverged_at is not None and not keep_best_iterate:
        hbi_log(verbose, fid,
                f"  WARNING: run stopped on ELBO divergence at iteration "
                f"{diverged_at:02d} and keep_best_iterate is off, so the "
                f"returned state is the diverged one. Set keep_best_iterate="
                f"True to fall back to the last good iteration.\n")

    qmutau_list: List[GaussianGammaDistribution] = qmutau
    he_list: List[np.ndarray] = [None] * K
    nk_vec: np.ndarray = np.zeros(K, dtype=float)
    for k in range(K):
        nu = qmutau_list[k].nu
        beta = qmutau_list[k].beta
        sigma = np.asarray(qmutau_list[k].sigma)
        if sigma.ndim == 2:
            # Joint models store the full inverse-scale matrix; the
            # per-parameter hierarchical error bar uses its diagonal
            # (sqrt of the full matrix would produce NaNs off-diagonal).
            sigma = np.diag(sigma).copy()
        s2 = 2.0 * sigma / beta
        nk = 2.0 * nu
        he_list[k] = np.sqrt(s2 / nk)
        nk_vec[k] = nk

    exceedance = cbm_hbi_exceedance(qm.alpha, is_null = isnull)

    theta_list = qhquad.parameters
    r_mat = r
    r_out = r_mat.T

    a_list: List[np.ndarray] = [None] * K
    # he_list already calculated above, don't reinitialize
    # nk_vec already calculated above, don't reinitialize

    theta_out: List[np.ndarray] = [None] * K
    for k in range(K):
        theta_k = theta_list[k].T
        theta_out[k] = theta_k
        a_list[k] = qmutau[k].a.copy()

    xp = exceedance.xp
    pxp = exceedance.pxp

    output = HBIOutput(
        parameters=theta_out,
        responsibility=r_out,
        group_mean=a_list,
        group_hierarchical_errorbar=he_list,
        model_frequency=Nbar / N,
        exceedance_prob=xp,
        protected_exceedance_prob=pxp,
    )

    hyper_out = hyper
    profile = HBIProfile(
        datetime=datetime.now().isoformat(),
        filename="cbm_hbi_hbi",
        config=config,
        # Store as dicts to avoid pickle class-identity issues (same reason as
        # FitProfile.config).  hbi_null reads cbm.input.optimconfigs, not this
        # field, so nothing downstream is broken.
        optimconfigs=[cfg.__dict__.copy() if hasattr(cfg, '__dict__') else cfg
                      for cfg in optconfigs],
        hyperparameters=hyper_out,
    )

    cbm_input = HBIInput(
        models=user_input["models"],
        fcbm_maps=user_input["fcbm_maps"],
        fname=user_input.get("fname", ""),
        config=user_input["config"],
        optimconfigs=user_input.get("optimconfigs", None),
    )

    cbm_math = HBIMath(
        qhquad=qhquad,
        r=r,
        qmutau=qmutau,
        qm=qm,
        bound=bound,
        Nbar=Nbar,
        hyper=hyper,
        he_list=he_list,
        nk_vec=nk_vec,
        exceedance=exceedance,
    )

    cbm = HBIResult(
        method="hbi",
        input=cbm_input,
        profile=profile,
        math=cbm_math,
        output=output,
    )

    # log_final(verbose, fid, output)

    if fname:
        with open(fname, "wb") as f:
            pickle.dump(cbm, f)

    if fid_file is not None:
        fid_file.close()

    return cbm


def hbi_init(flap, hyper, limInf=0, initialize_r='all_r_1', families=None):
    if families is None:
        families = []
    b = hyper['b']
    v = hyper['v']
    s = hyper['s']
    K = len(flap)
    allfiles_map = True
    cbm_maps = []
    for k in range(K):
        fcbm_map = flap[k]
        if isinstance(fcbm_map, str):
            with open(fcbm_map, 'rb') as f:
                cbm = pickle.load(f)
                cbm_maps.append(cbm)
        elif isinstance(fcbm_map, dict):
            allfiles_map = allfiles_map and False
            cbm_maps.append(fcbm_map)
        else:
            raise ValueError(
                f"fcbm_map input has not properly been specified for model {k + 1}!"
            )
    bb = BoundTerms(
        ElogpX=np.nan,
        ElogpH=np.nan,
        ElogpZ=np.nan,
        Elogpmu=np.nan,
        Elogptau=np.nan,
        Elogpm=0.0,
        ElogqH=np.nan,
        ElogqZ=np.nan,
        Elogqmu=np.nan,
        Elogqtau=np.nan,
        Elogqm=0.0,
        pmlimInf=bool(limInf),
        lastmodule='',
        L=np.nan,
        dL=np.nan,
    )
    from .optimization import Config as _Config
    opt_configs = []
    for k in range(K):
        cbm_map = cbm_maps[k]
        opt_config = cbm_map.profile.config
        # FitProfile.config may be stored as a plain dict (new behaviour) or
        # as a Config object (old pickles). Normalise to Config here so the
        # rest of hbi_run can always use attribute access.
        if isinstance(opt_config, dict):
            opt_config = _Config(**opt_config)
        opt_configs.append(opt_config)
    
    logrho = []
    theta = []
    Ainvdiag = []
    Ainv_full = []  # <--- NEW: Master list for full inverse Hessians
    logdetA = []
    logf = []
    D = []
    a0 = []
    N = cbm_maps[0].output.parameters.shape[0]
    
    for k in range(K):
        cbm_map = cbm_maps[k]
        a0_k = np.asarray(cbm_map.profile.prior_mean).ravel()
        
        theta_k = cbm_map.math.parameters
        Ainvdiag_k = cbm_map.math.hessian_inv_diag
        hessian_k = cbm_map.math.hessian
        
        Dk = len(a0_k)
        Ainv_full_k = np.zeros((Dk, Dk, N), dtype=float)
        
        # =====================================================================
        # --- NEW: Aggressive Scrubber for Failed/Runaway Subjects ---
        # =====================================================================
        # =====================================================================
        # --- NEW: Aggressive Scrubber for Failed/Runaway Subjects ---
        # =====================================================================
        for n in range(N):
            # 1. Check if ANY output from the individual fit is corrupted (NaN/Inf)
            is_invalid = (
                not np.all(np.isfinite(theta_k[n])) or
                not np.all(np.isfinite(hessian_k[n])) or
                not np.all(np.isfinite(Ainvdiag_k[n])) or
                not np.isfinite(cbm_map.math.loglik[n])
            )
            # 2. Check if the optimizer "ran away" to a massive number
            is_runaway = np.any(np.abs(theta_k[n]) > 1000.0)
            
            if is_invalid or is_runaway:
                # If subject failed or ran away, replace with neutral prior fallbacks
                theta_k[n] = a0_k.copy()
                hessian_k[n] = np.eye(Dk) * 0.1
                Ainvdiag_k[n] = np.ones(Dk) * 10.0
                
                # Neutralize log-likelihoods so they don't break EM responsibilities
                cbm_map.math.loglik[n] = -1e6
                cbm_map.math.lme[n] = -1e6
                cbm_map.math.log_det_hessian[n] = 0.0
                
            # Safely invert the matrix
            Ainv_full_k[:, :, n] = np.linalg.pinv(hessian_k[n])
        # =====================================================================
        # =====================================================================
            
        Ainv_full.append(Ainv_full_k)
        theta.append(np.column_stack(theta_k))
        Ainvdiag.append(np.column_stack(Ainvdiag_k))
        
        logrho.append(np.asarray(cbm_map.math.lme))
        logf.append(np.asarray(cbm_map.math.loglik))
        logdetA.append(np.asarray(cbm_map.math.log_det_hessian))
        a0.append(a0_k)
        D.append(Dk)
        
    logf_mat = np.vstack(logf)
    logdetA_mat = np.vstack(logdetA)
    D = np.array(D)
    
    qh = IndividualPosterior(
        loglik=logf_mat,
        parameters=theta,
        hessian_inv_diag=Ainvdiag,
        log_det_hessian=logdetA_mat,
        hessian_inv=Ainv_full,  # <--- NEW: Inject it into the starting dataclass!
    )
    
    a = []
    beta = []
    sigma = []
    nu = []
    alpha0 = np.ones(K)
    for k in range(K):
        a.append(np.asarray(a0[k]))
        beta.append(b)
        if not isinstance(s, (list, tuple)):
            sigma.append(s * np.ones_like(a[k]))
        else:
            sigma_k = np.asarray(s[k])
            if sigma_k.shape != a[k].shape:
                raise ValueError(
                    f"length of s is not match with that for a for model {k + 1}"
                )
            sigma.append(sigma_k)
        nu.append(v)
    if len(families) > 0:
        families_arr = np.asarray(families)
        alpha0[:] = np.nan
        uf = np.unique(families_arr)
        for f in uf:
            mask = (families_arr == f)
            nf = mask.sum()
            alpha0[mask] = 1.0 / nf
    pmutau: List[GaussianGammaDistribution] = []
    for k in range(K):
        pmutau.append(
            GaussianGammaDistribution(
                a=a[k],
                beta=beta[k],
                sigma=sigma[k],
                nu=nu[k],
                Etau=np.zeros_like(a[k]),
                Elogtau=np.zeros_like(a[k]),
                logG=0.0,
            )
        )
    pm_alpha = alpha0.copy()
    pm = DirichletDistribution(
        limInf=bool(limInf),
        alpha=pm_alpha,
        Elogm=np.zeros_like(pm_alpha, dtype=float),
        logC=0.0,
    )
    for k in range(K):
        a_k = pmutau[k].a
        beta_k = pmutau[k].beta
        nu_k = np.asarray(pmutau[k].nu)
        sigma_k = np.asarray(pmutau[k].sigma)
        Elogtau = psi(nu_k) - np.log(sigma_k)
        Etau = nu_k / sigma_k
        logG = np.sum(-gammaln(nu_k) + nu_k * np.log(sigma_k))
        pmutau[k] = GaussianGammaDistribution(
            a=a_k,
            beta=beta_k,
            sigma=np.asarray(sigma_k),
            nu=float(nu_k),
            Etau=np.asarray(Etau),
            Elogtau=np.asarray(Elogtau),
            logG=float(logG),
        )
    alpha = pm.alpha
    alpha_star = np.sum(alpha)
    Elogm = psi(alpha) - psi(alpha_star)
    loggamma1 = gammaln(alpha)
    logC = gammaln(alpha_star) - np.sum(loggamma1)
    if pm.limInf:
        pm.alpha = np.full_like(alpha, np.inf, dtype=float)
        Elogm = np.full_like(alpha, np.inf, dtype=float)
        logC = 0.0
    pm.Elogm = Elogm
    pm.logC = float(logC)
    lme = np.vstack(logrho).T
    if initialize_r == 'all_r_1':
        r = np.ones((K, N))
    elif initialize_r == 'lme_softmax':
        # Break the K>1 symmetry using the individual fits' per-subject log
        # model evidence: each subject starts mostly assigned to the model
        # whose INDEPENDENT fit explains them better (softmax over models of
        # lme -- the same formula the E-step's r update uses, applied once at
        # initialization).
        #
        # Why this exists: with 'all_r_1' every subject starts fully assigned
        # to EVERY model, so when the K models share one likelihood and differ
        # only in their priors (a population-mixture setup), the first M-step
        # hands all components the same responsibility-weighted mean and they
        # merge immediately -- the responsibilities may still split later, but
        # the component means never separate. Evidence-based initialization
        # starts the components apart, at the subjects their own priors favor.
        lme_safe = np.where(np.isfinite(lme), lme, -1e10)   # (N, K)
        z = lme_safe - lme_safe.max(axis=1, keepdims=True)
        w = np.exp(z)
        w /= np.clip(w.sum(axis=1, keepdims=True), 1e-300, None)
        r = np.ascontiguousarray(w.T)                        # (K, N)
    else:
        raise NotImplementedError(
            f"initialize_r option '{initialize_r}' not implemented."
        )
    bound = BoundState(
        bound=bb,
        qHZ=BoundQHZ(
            ElogpX=np.full(K, np.nan),
            ElogpH=np.full(K, np.nan),
            ElogpZ=np.full(K, np.nan),
            ElogqH=np.full(K, np.nan),
            ElogqZ=np.full(K, np.nan),
        ),
        qmutau=BoundQMutau(
            ElogpH=np.full(K, np.nan),
            Elogpmu=np.full(K, np.nan),
            Elogptau=np.full(K, np.nan),
            Elogqmu=np.full(K, np.nan),
            Elogqtau=np.full(K, np.nan),
        ),
        qm=BoundQM(
            ElogpZ=np.full(K, np.nan),
            Elogpm=np.nan,
            Elogqm=np.nan,
        ),
    )
    inits = {
        'qh': qh,
        'r': r,
        'bound': bound,
    }
    priors = {
        'hyper': hyper,
        'pmutau': pmutau,
        'pm': pm,
    }
    return inits, priors, opt_configs


def hbi_null(
    data: List[Any],
    fname_cbm: Union[str, HBIResult],
) -> HBIResult:
    """
    Parameters
    ----------
    data : list
        List of subject-level data objects.
    fname_cbm : str or CBMResult
        If str: path to a saved cbm object (pickled).
        If CBMResult: already-loaded cbm structure.

    Returns
    -------
    cbm : CBMResult
        Original HBI result, updated with protected exceedance probs.
    cbm0 : CBMResult
        HBI result under the null hypothesis.
    """
    # ------------------------------------------------------------------
    # Load cbm if a filename is given
    # ------------------------------------------------------------------
    inputisfile = False
    fname = None

    if isinstance(fname_cbm, str):
        inputisfile = True
        fname = fname_cbm
        with open(fname, "rb") as f:
            loaded = pickle.load(f)
        # handle either cbm or {'cbm': cbm}
        if isinstance(loaded, dict) and "cbm" in loaded:
            cbm = loaded["cbm"]
        else:
            cbm = loaded
    elif isinstance(fname_cbm, HBIResult):
        cbm = fname_cbm
    else:
        raise TypeError("fname_cbm must be either a filename (str) or a CBMResult")

    # ------------------------------------------------------------------
    # Derive output filename for null model if we loaded from file
    # ------------------------------------------------------------------
    fname0 = None
    if inputisfile:
        fdir, fbase = os.path.split(fname)
        root, ext = os.path.splitext(fbase)
        if not ext:
            ext = ".pkl"
        fname0 = os.path.join(fdir, f"{root}_null{ext}")

    # ------------------------------------------------------------------
    # Extract input components
    # ------------------------------------------------------------------
    models = cbm.input.models
    fcbm_maps = cbm.input.fcbm_maps
    config = cbm.input.config
    optimconfigs = cbm.input.optimconfigs if cbm.input.optimconfigs is not None else []
    hyper = cbm.profile.hyperparameters
    isnull = 1  # used for limInf in cbm_hbi_init

    # ------------------------------------------------------------------
    # Adjust config for null run: flog and fname_prog get "_null"
    # ------------------------------------------------------------------
    # config may be an HBIConfig dataclass or a dict
    if isinstance(config, HBIConfig):
        config_null = deepcopy(config)
        # flog
        if isinstance(config_null.flog, str):
            fdir, fbase = os.path.split(config_null.flog)
            root, ext = os.path.splitext(fbase)
            config_null.flog = os.path.join(fdir or ".", f"{root}_null{ext}")
        # fname_prog
        if isinstance(config_null.fname_prog, str):
            fdir, fbase = os.path.split(config_null.fname_prog)
            root, ext = os.path.splitext(fbase)
            if not ext:
                ext = ".pkl"
            config_null.fname_prog = os.path.join(fdir or ".", f"{root}_null{ext}")
    else:
        # assume dict-like
        config_null = deepcopy(config)
        # flog
        flog = config_null.get("flog", None)
        if isinstance(flog, str):
            fdir, fbase = os.path.split(flog)
            root, ext = os.path.splitext(fbase)
            config_null["flog"] = os.path.join(fdir or ".", f"{root}_null{ext}")
        # fname_prog
        fname_prog = config_null.get("fname_prog", None)
        if isinstance(fname_prog, str):
            fdir, fbase = os.path.split(fname_prog)
            root, ext = os.path.splitext(fbase)
            if not ext:
                ext = ".pkl"
            config_null["fname_prog"] = os.path.join(fdir or ".", f"{root}_null{ext}")

    # ------------------------------------------------------------------
    # Build user_input for the null HBI run
    # ------------------------------------------------------------------
    user_input = {
        "models": models,
        "fcbm_maps": fcbm_maps,
        "fname": fname0,
        "config": config_null,
        "optimconfigs": optimconfigs,
    }

    # ensure we have an HBIConfig instance for initialization
    if isinstance(config_null, HBIConfig):
        config_for_init = config_null
    else:
        config_for_init = HBIConfig(**config_null)

    # ------------------------------------------------------------------
    # Initialize HBI under the null (limInf = isnull)
    # ------------------------------------------------------------------
    inits, priors, opt_configs = hbi_init(
        fcbm_maps,
        hyper,
        isnull,
        config_for_init.initialize,
    )

    # ------------------------------------------------------------------
    # Run HBI under null hypothesis
    # ------------------------------------------------------------------
    cbm0 = hbi_run(data, user_input, inits, priors, opt_configs)

    # ------------------------------------------------------------------
    # Use cbm0 to compute protected exceedance probability
    # ------------------------------------------------------------------
    alpha = np.asarray(cbm.math.qm.alpha, dtype=float)
    L = float(cbm.math.bound.bound.L)
    L0 = float(cbm0.math.bound.bound.L)

    exceedance = cbm_hbi_exceedance(alpha, L=L, L0=L0)

    # Update cbm with exceedance results
    cbm.math.exceedance = exceedance
    cbm.output.protected_exceedance_prob = exceedance.pxp

    # ------------------------------------------------------------------
    # Save updated cbm back to file if needed
    # ------------------------------------------------------------------
    if inputisfile and fname is not None:
        with open(fname, "wb") as f:
            pickle.dump(cbm, f)

    return cbm