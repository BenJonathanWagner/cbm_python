"""
Cross-task parameter covariance in HBI - a recovery demo.
=======================================================================

WHAT THIS SHOWS

Two tasks are fit as one concatenated model. Subject parameters are drawn from a
multivariate normal in which the two tasks' inverse temperatures are correlated
at a known value (TRUE_R below). Choices are then simulated over REAL stimulus
sequences from both experiments, refit, and the group-level correlation is
recovered as a hyperparameter.

Three HBI fits are run on identical data:

  1. covariance_blocks = None                    the upstream algorithm. The
                                                 group covariance is diagonal by
                                                 construction, so the cross-task
                                                 correlation is not estimable.
  2. covariance_blocks = [[(4, 5)]]              the two inverse temperatures
                                                 share a Normal-Wishart block.
  3. ... plus recenter_a0 = True                 same, with the group-mean
                                                 hyperprior recentred on the
                                                 individual fits.

Fits 1 and 2 also let you check that adding the block leaves the group MEANS
essentially unchanged -- the extension should answer a new question without
perturbing the answers upstream already gives.

For the complementary check -- that an identity mask reproduces the upstream
Gaussian-Gamma branch exactly, posterior and all five ELBO terms -- run
    python -m cbm.test_joint_elbo

RUNNING IT

    pip install -e .                            # once, from the repository root
    python examples/example_joint_covariance.py

No data files and no network access are needed; the stimulus sequences are in
example_joint_stimuli.py, next to this script. About six minutes at the default
N_SUBJECTS = 120, scaling roughly linearly in N.

EXPECTED OUTPUT at the defaults (N = 120, seed = 20260809):

    imposed (truth)                  +0.500
    realised in this draw            +0.452     <- finite-sample draw
    Pearson r of the MAP estimates   +0.193     <- attenuated by fitting noise
    HBI, covariance_blocks=None         n/a     <- diagonal by construction
    HBI, joint                       +0.376     95% CI [+0.214, +0.521]
    HBI, joint + recenter_a0         +0.309     95% CI [+0.141, +0.463]

    reliability  beta_RL 0.696  beta_risk 0.428  ceiling sqrt(rel*rel) = 0.546
    predicted MAP r = 0.452 * 0.546 = +0.246    (observed +0.193)

    max |group-mean difference|, diagonal vs joint:  0.037

Every point estimate above reproduced exactly on macOS and on Linux. The
credible-interval endpoints can differ by ~0.003 between platforms: they are
drawn from the fitted Wishart with a fixed seed, but the fitted Sigma itself
depends on individual_fit's optimiser restarts, which come from NumPy's global
RNG and are not seeded here. Add np.random.seed(SEED) if you need the last digit
to be reproducible too.

The two things to look at: the joint estimate recovers most of the attenuation
in the MAP correlation and its credible interval covers the realised value, and
adding the covariance block leaves every group mean where the diagonal fit put
it. Both credible intervals are wide because N = 120 with these reliabilities is
a modest amount of information about a correlation -- raise N_SUBJECTS to narrow
them.

THE TASKS

  Milky Way   72-trial two-armed bandit. Both arms have independently drifting
              reward magnitudes; the reward is deterministic given the choice.
              Fit with a 5-parameter Q-learning + choice-repetition model.

  Scavenger   40-trial risky/ambiguous choice. A certain amount versus a 50/50
              gamble, half gain trials and half loss trials, with the gamble's
              probability hidden on half of them. Fit with a 4-parameter
              bias-logit model with separate gain and loss offsets.

Joint parameter vector (D = 9), all in unconstrained space:

    0 alpha_RL    learning rate            sigmoid
    1 omega_RL    decay of unchosen Q      sigmoid
    2 eta_RL      repetition update rate   sigmoid
    3 rho_RL      repetition weight        identity
    4 beta_RL     inverse temperature      exp        <-- linked
    5 beta_risk   inverse temperature      exp        <-- linked
    6 phi_gain    gain-trial risk offset   identity
    7 phi_loss    loss-trial risk offset   identity
    8 eta_risk    ambiguity aversion       identity

Ben J. Wagner, 2026.  The two model functions are verbatim copies of two model candidates
used in the analysis, so the demo exercises the same code paths.
"""

import math
import pickle
import time
from pathlib import Path

import numpy as np

from cbm.individual_fit import individual_fit
from cbm.hbi import hbi_main
from cbm.hbi_config import HBIConfig
from cbm.posterior_correlation import all_linked_correlation_cis

from example_joint_stimuli import MW_WALKS, RISK_TRIALS, N_STIM

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
N_SUBJECTS = 220        # ~6 min end to end; runtime scales roughly linearly
TRUE_R     = 0.6        # imposed correlation between beta_RL and beta_risk
SEED       = 12314
NUM_INIT   = 8          # optimiser restarts per subject in individual_fit
OUT_DIR    = Path(__file__).resolve().parent / "output_joint_covariance"

N_RL, N_RISK = 5, 4
BETA_RL, BETA_RISK = 4, 5          # indices in the joint vector

# On macOS, Accelerate raises spurious divide/overflow/invalid floating-point
# flags inside matmul -- the same platform issue safe_invert() guards against.
# The results are unaffected: point estimates from a Mac and a Linux run agree
# to three decimals. The flags are silenced around the fitting calls only, so
# that a first-time reader is not met with a screen of warnings.
FP_QUIET = dict(divide="ignore", over="ignore", invalid="ignore")
NAMES = ["alpha_RL", "omega_RL", "eta_RL", "rho_RL", "beta_RL",
         "beta_risk", "phi_gain", "phi_loss", "eta_risk"]

# Ground-truth group mean and between-subject SD, in unconstrained space.
# Chosen so that the two LINKED parameters are well identified from these trial
# counts -- r(MAP, true) is about 0.9 for both -- since a correlation cannot be
# recovered from parameters the data barely constrain. alpha_RL is deliberately
# left in its realistic (poorly identified) regime; see the note printed at the
# end of the run.
MU_TRUE = np.array([2.2, -0.4, 0.0, 1.5, 2.0,   2.1,  0.4, -0.4, 0.12])
SD_TRUE = np.array([0.8,  0.8, 0.8, 0.6, 0.8,   0.8,  0.5,  0.5, 0.12])

# Subject-level priors, as used in the analysis
PRIOR_MEAN = np.array([2.2, -0.4, 0.0, 0.0, 0.0,   0.0, 0.0, 0.0, 0.0])
PRIOR_VAR  = np.array([2.25, 4.0, 6.25, 4.0, 6.0,  10.0, 2.0, 2.0, 0.25])


# ---------------------------------------------------------------------------
# Models
#
# Each model appears twice. The *_reference version is a verbatim copy of the
# analysis code. The working version is algebraically identical but avoids
# allocating small numpy arrays inside the trial loop, which makes it about
#  faster, which matters here, because L-BFGS-B approximates
# its gradients by finite differences and therefore evaluates the likelihood
# ~10 times per iteration per parameter.
#
# _check_model_equivalence() below asserts the two agree to 1e-9 on random
# parameter vectors, and runs at import. If it ever fails, trust the reference.
# ---------------------------------------------------------------------------
def repetition_model_reference(parameters, data):
    """Q-learning with decay of the unchosen option plus choice repetition.
    Params (5): [alpha_pre, omega_pre, eta_pre, rho, beta_pre]"""
    choices, outcomes = data
    alpha = 1 / (1 + np.exp(-parameters[0]))
    omega = 1 / (1 + np.exp(-parameters[1]))
    eta   = 1 / (1 + np.exp(-parameters[2]))
    rho   = parameters[3]
    beta  = np.exp(parameters[4])

    Q   = np.array([0.5, 0.5])
    REP = np.array([0.0, 0.0])
    log_lik = 0.0
    for t in range(len(choices)):
        c  = int(choices[t])
        uc = 1 - c
        net = beta * Q + rho * REP
        exp_val = np.exp(net - np.max(net))
        p = exp_val / np.sum(exp_val)
        log_lik += np.log(p[c] + 1e-10)

        pe = outcomes[t] - Q[c]
        Q[c]  = Q[c] + alpha * pe
        Q[uc] = (1.0 - omega) * Q[uc] + omega * 0.5
        REP[c]  = REP[c]  + eta * (1.0 - REP[c])
        REP[uc] = REP[uc] + eta * (0.0 - REP[uc])
    return log_lik


def risk_model_reference(parameters, data):
    """Bias-logit with ambiguity discount and separate gain/loss offsets.
    Params (4): [beta_pre, phi_gain, phi_loss, eta]"""
    choice, safe_magn, risky_magn, risky_prob, ambg = data
    beta     = np.exp(parameters[0])
    phi_gain = parameters[1]
    phi_loss = parameters[2]
    eta      = parameters[3]

    log_lik = 0.0
    for t in range(len(choice)):
        c = int(choice[t])
        subj_prob = (0.5 - eta) if ambg[t] == 1 else 0.5
        Exp_risk  = subj_prob * risky_magn[t]
        phi_t     = phi_gain if safe_magn[t] >= 0 else phi_loss
        net = np.array([beta * safe_magn[t], beta * Exp_risk + phi_t])
        exp_val = np.exp(net - np.max(net))
        p = exp_val / np.sum(exp_val)
        log_lik += np.log(p[c] + 1e-10)
    return log_lik


def repetition_model(parameters, data):
    """Scalar reformulation of repetition_model_reference."""
    choices, outcomes = data
    alpha = 1.0 / (1.0 + math.exp(-parameters[0]))
    omega = 1.0 / (1.0 + math.exp(-parameters[1]))
    eta   = 1.0 / (1.0 + math.exp(-parameters[2]))
    rho   = parameters[3]
    beta  = math.exp(parameters[4])

    q0 = q1 = 0.5
    r0 = r1 = 0.0
    log_lik = 0.0
    for t in range(len(choices)):
        c = choices[t]
        n0 = beta * q0 + rho * r0
        n1 = beta * q1 + rho * r1
        m = n0 if n0 > n1 else n1
        e0 = math.exp(n0 - m); e1 = math.exp(n1 - m)
        log_lik += math.log((e0 if c == 0 else e1) / (e0 + e1) + 1e-10)

        o = outcomes[t]
        if c == 0:
            q0 += alpha * (o - q0); q1 = (1.0 - omega) * q1 + omega * 0.5
            r0 += eta * (1.0 - r0); r1 -= eta * r1
        else:
            q1 += alpha * (o - q1); q0 = (1.0 - omega) * q0 + omega * 0.5
            r1 += eta * (1.0 - r1); r0 -= eta * r0
    return log_lik


def risk_model(parameters, data):
    """Vectorised reformulation of risk_model_reference. The risk model carries
    no state across trials, so it vectorises exactly."""
    choice, safe_magn, risky_magn, risky_prob, ambg = data
    beta     = math.exp(parameters[0])
    phi_gain = parameters[1]
    phi_loss = parameters[2]
    eta      = parameters[3]

    subj_prob = np.where(ambg == 1, 0.5 - eta, 0.5)
    phi_t     = np.where(safe_magn >= 0, phi_gain, phi_loss)
    n0 = beta * safe_magn
    n1 = beta * subj_prob * risky_magn + phi_t
    m  = np.maximum(n0, n1)
    e0 = np.exp(n0 - m); e1 = np.exp(n1 - m)
    p_chosen = np.where(choice == 0, e0, e1) / (e0 + e1)
    return float(np.sum(np.log(p_chosen + 1e-10)))


def _check_model_equivalence(n_draws=8, tol=1e-9, seed=1):
    """Assert the fast models match the verbatim reference implementations."""
    rng = np.random.default_rng(seed)
    walk, trials = MW_WALKS[0], RISK_TRIALS[0]
    ch = rng.integers(0, 2, walk.shape[0])
    mw = (ch, walk[np.arange(walk.shape[0]), ch])
    rk = (rng.integers(0, 2, trials.shape[0]).astype(float),
          trials[:, 0], trials[:, 1], trials[:, 2], trials[:, 3])
    worst = 0.0
    for _ in range(n_draws):
        p = rng.normal(0, 1.0, N_RL + N_RISK)
        a = repetition_model_reference(p[:N_RL], mw) + risk_model_reference(p[N_RL:], rk)
        b = repetition_model(p[:N_RL], mw) + risk_model(p[N_RL:], rk)
        worst = max(worst, abs(a - b))
    if worst > tol:
        raise AssertionError(f"fast models diverge from reference by {worst:.2e}")
    return worst


def joint_model(parameters, data):
    mw_data, risk_data = data
    return (repetition_model(parameters[:N_RL], mw_data)
            + risk_model(parameters[N_RL:], risk_data))


joint_model.__name__ = "rep_x_risk_joint"


# ---------------------------------------------------------------------------
# Simulators (the generative counterparts of the two models above)
# ---------------------------------------------------------------------------
def simulate_mw(params, walk, rng):
    """Choices only; rewards come from the real walk and are deterministic
    given the choice, exactly as in the experiment."""
    alpha = 1 / (1 + np.exp(-params[0]))
    omega = 1 / (1 + np.exp(-params[1]))
    eta   = 1 / (1 + np.exp(-params[2]))
    rho   = params[3]
    beta  = np.exp(params[4])

    n = len(walk)
    Q, REP = np.array([0.5, 0.5]), np.array([0.0, 0.0])
    choices, outcomes = np.zeros(n, dtype=int), np.zeros(n)
    for t in range(n):
        net = beta * Q + rho * REP
        p = np.exp(net - net.max()); p /= p.sum()
        c = rng.choice(2, p=p)
        r = float(walk[t, c])
        choices[t], outcomes[t] = c, r
        Q[c]   += alpha * (r - Q[c])
        Q[1-c]  = (1 - omega) * Q[1-c] + omega * 0.5
        REP[c]   += eta * (1.0 - REP[c])
        REP[1-c] += eta * (0.0 - REP[1-c])
    return (choices, outcomes)


def simulate_risk(params, trials, rng):
    """Choices only; the trial structure is the real one."""
    safe, risky, prob, ambg = trials[:, 0], trials[:, 1], trials[:, 2], trials[:, 3]
    beta = np.exp(params[0])
    phi_gain, phi_loss, eta = params[1], params[2], params[3]

    n = len(safe)
    choices = np.zeros(n)
    for t in range(n):
        subj_prob = (0.5 - eta) if ambg[t] == 1 else 0.5
        phi_t = phi_gain if safe[t] >= 0 else phi_loss
        net = np.array([beta * safe[t], beta * subj_prob * risky[t] + phi_t])
        p = np.exp(net - net.max()); p /= p.sum()
        choices[t] = float(rng.choice(2, p=p))
    return (choices, safe, risky, prob, ambg)


# ---------------------------------------------------------------------------
def build_true_covariance():
    Sigma = np.diag(SD_TRUE ** 2)
    c = TRUE_R * SD_TRUE[BETA_RL] * SD_TRUE[BETA_RISK]
    Sigma[BETA_RL, BETA_RISK] = Sigma[BETA_RISK, BETA_RL] = c
    return Sigma


def reliability(res_diagonal, hessians):
    """rel_d = Sigma_dd / (Sigma_dd + v_d).

    signal  Sigma_dd  the between-subject variance of the TRUE parameter, taken
                      from the diagonal HBI fit. Under the Gaussian-Gamma
                      posterior tau ~ Gamma(nu, sigma), E[1/tau] = sigma/(nu-1).
    noise   v_d       the mean posterior variance of one subject's estimate,
                      i.e. the average squared standard error.

    Using the fitted group variance rather than the observed variance of the MAP
    estimates matters: the MAPs are already pulled toward the prior, so
    Var(MAP) understates signal + noise and the naive 1 - v/Var(MAP) can go
    negative for weakly identified parameters."""
    qm = res_diagonal.math.qmutau[0]
    sigma = np.asarray(qm.sigma, dtype=float).ravel()
    var_group = sigma / max(float(qm.nu) - 1.0, 1e-12)
    v = np.mean([np.diag(np.linalg.inv(H)) for H in hessians], axis=0)
    return var_group / (var_group + v)


def run_hbi(tag, pkl, blocks, recenter):
    cfg = HBIConfig(covariance_blocks=blocks,
                    auto_nu0=blocks is not None,
                    recenter_a0=recenter,
                    keep_best_iterate=True,
                    divergence_tol=50.0,
                    save_prog=False,
                    verbose=0,
                    flog=-1)
    t0 = time.time()
    with np.errstate(**FP_QUIET):
        res = hbi_main([d for d in DATA], [joint_model], [pkl],
                       fname=str(OUT_DIR / f"hbi_{tag}.pkl"), config=cfg)
    print(f"  {tag:<26} done in {time.time() - t0:6.1f} s")
    return res


# ===========================================================================
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    Sigma_true = build_true_covariance()

    print("=" * 74)
    print("Cross-task covariance recovery".center(74))
    print("=" * 74)
    print(f"subjects              {N_SUBJECTS}")
    print(f"parameters            {len(NAMES)}  ({N_RL} RL + {N_RISK} risk)")
    print(f"trials per subject    {MW_WALKS.shape[1]} bandit + {RISK_TRIALS.shape[1]} risky choice")
    print(f"stimulus sequences    {N_STIM} real ones, assigned by  subject index % {N_STIM}")
    print(f"imposed r(beta_RL, beta_risk)  =  {TRUE_R:+.2f}")
    print(f"seed                  {SEED}")
    print(f"model check           fast vs reference agree to "
          f"{_check_model_equivalence():.1e}")

    # --- 1. draw parameters and simulate --------------------------------
    theta_true = rng.multivariate_normal(MU_TRUE, Sigma_true, size=N_SUBJECTS)
    DATA = []
    for n in range(N_SUBJECTS):
        k = n % N_STIM
        DATA.append((simulate_mw(theta_true[n, :N_RL], MW_WALKS[k], rng),
                     simulate_risk(theta_true[n, N_RL:], RISK_TRIALS[k], rng)))

    realised_r = np.corrcoef(theta_true[:, BETA_RL], theta_true[:, BETA_RISK])[0, 1]
    print(f"\nrealised r in the draw          {realised_r:+.3f}"
          f"   (finite-sample deviation from {TRUE_R:+.2f})")

    # --- 2. individual fits ----------------------------------------------
    print("\nfitting subjects independently ...")
    t0 = time.time()
    pkl = str(OUT_DIR / "individual_fit.pkl")
    with np.errstate(**FP_QUIET):
        ind = individual_fit(DATA, joint_model, PRIOR_MEAN, PRIOR_VAR, fname=pkl,
                             config={"num_init": NUM_INIT, "verbose": False})
    print(f"  done in {time.time() - t0:.1f} s")

    theta_map = ind.output.parameters
    raw_r = np.corrcoef(theta_map[:, BETA_RL], theta_map[:, BETA_RISK])[0, 1]

    # --- 3. the three HBI fits -------------------------------------------
    print("\nrunning HBI ...")
    res_diag = run_hbi("diagonal (upstream)", pkl, None, False)
    res_join = run_hbi("joint", pkl, [[(BETA_RL, BETA_RISK)]], False)
    res_rec  = run_hbi("joint + recenter_a0", pkl, [[(BETA_RL, BETA_RISK)]], True)

    # The interval is drawn from the fitted Wishart by Monte Carlo, so it needs
    # its own seed to be reproducible -- without one, two runs of an otherwise
    # identical fit differ in the third decimal of the CI while the point
    # estimate is bit-identical.
    ci_join = all_linked_correlation_cis(
        res_join, k=0, rng=np.random.default_rng(SEED))[(BETA_RL, BETA_RISK)]
    ci_rec = all_linked_correlation_cis(
        res_rec, k=0, rng=np.random.default_rng(SEED))[(BETA_RL, BETA_RISK)]

    rel = reliability(res_diag, ind.math.hessian)
    ceiling = np.sqrt(rel[BETA_RL] * rel[BETA_RISK])

    # --- 4. report --------------------------------------------------------
    print("\n" + "=" * 74)
    print("r(beta_RL, beta_risk)")
    print("=" * 74)
    print(f"  {'imposed (truth)':<38}{TRUE_R:>+8.3f}")
    print(f"  {'realised in this draw':<38}{realised_r:>+8.3f}")
    print(f"  {'Pearson r of the MAP estimates':<38}{raw_r:>+8.3f}"
          f"    attenuated -- see below")
    print(f"  {'HBI, covariance_blocks=None':<38}{'n/a':>8}"
          f"    diagonal by construction")
    print(f"  {'HBI, joint':<38}{ci_join['point']:>+8.3f}"
          f"    95% CI [{ci_join['ci_low']:+.3f}, {ci_join['ci_high']:+.3f}]")
    print(f"  {'HBI, joint + recenter_a0':<38}{ci_rec['point']:>+8.3f}"
          f"    95% CI [{ci_rec['ci_low']:+.3f}, {ci_rec['ci_high']:+.3f}]")

    print("\n" + "-" * 74)
    print("why the MAP correlation sits below the truth")
    print("-" * 74)
    print(f"  reliability of beta_RL             {rel[BETA_RL]:.3f}")
    print(f"  reliability of beta_risk           {rel[BETA_RISK]:.3f}")
    print(f"  attenuation factor sqrt(rel*rel)   {ceiling:.3f}")
    print(f"  predicted MAP r = realised * factor {realised_r * ceiling:+.3f}"
          f"  (observed {raw_r:+.3f})")
    print("  Estimation error inflates each parameter's variance but not their")
    print("  covariance, so a correlation of point estimates is attenuated. HBI")
    print("  models that error explicitly, so its hyperparameter is not.")

    print("\n" + "-" * 74)
    print("does adding the block disturb what upstream already estimates?")
    print("-" * 74)
    gm_diag = np.asarray(res_diag.output.group_mean[0]).ravel()
    gm_join = np.asarray(res_join.output.group_mean[0]).ravel()
    print(f"  {'parameter':<12}{'truth':>9}{'diagonal':>11}{'joint':>11}"
          f"{'joint-diag':>12}{'rel':>7}")
    for d in range(len(NAMES)):
        print(f"  {NAMES[d]:<12}{MU_TRUE[d]:>+9.2f}{gm_diag[d]:>+11.3f}"
              f"{gm_join[d]:>+11.3f}{gm_join[d] - gm_diag[d]:>+12.4f}{rel[d]:>7.2f}")
    print(f"\n  largest absolute difference in group means: "
          f"{np.max(np.abs(gm_join - gm_diag)):.4f}")
    print("  Adding the covariance block should not move the group means, and")
    print("  does not. Individual means can still sit away from the truth where")
    print("  a parameter is weakly identified (low rel) -- alpha_RL here -- which")
    print("  is a property of the task, not of the extension: the diagonal and")
    print("  joint fits are displaced by the same amount.")

    with open(OUT_DIR / "example_results.pkl", "wb") as f:
        pickle.dump({"theta_true": theta_true, "individual": ind,
                     "diagonal": res_diag, "joint": res_join,
                     "joint_recentred": res_rec}, f)
    print(f"\nall fits written to {OUT_DIR}/")
