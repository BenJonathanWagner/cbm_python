"""
Does recenter_a0 bias the recovered correlation downward?
=========================================================

THE QUESTION

`recenter_a0` removes this term from the Normal-Wishart scale update:

    (N_k beta0 / beta_k) * (thetabar - a0)(thetabar - a0)^T

It is rank one, so its own correlation is exactly +1 whenever the two offsets
share a sign -- which is the usual case, because `a0` comes from a generic
individual-fit prior and both group means typically sit on the same side of it.
It therefore always drags the estimate toward +1, and switching the flag on
always lowers it.

Two single runs of example_joint_covariance.py suggested that the *recentred*
estimate ends up further from the truth than the un-recentred one, which would
mean two errors were cancelling: a genuine downward bias in the sufficient
statistic S, offset by this spurious upward push. Two runs cannot establish
that. This script does it properly.

THE DESIGN

Paired: each replication simulates ONE dataset and fits it TWICE, with the flag
off and on. The comparison therefore removes between-replication variance
entirely, which matters because the replication-to-replication spread of a
correlation at these sample sizes is far larger than the effect being measured.

Each replication records

    realised   the correlation actually present in the drawn parameters
               (the right target -- the imposed value is only its expectation)
    raw_map    Pearson r of the independent MAP estimates, for context
    r_joint    covariance_blocks only
    r_recen    covariance_blocks + recenter_a0
    offset_ij  (thetabar_i - a0_i)(thetabar_j - a0_j), the size of the rank-one
               term, so the shift can be checked against its mechanism

and is appended to a CSV immediately, so the run is resumable: re-running picks
up after the last completed replication rather than starting over.

RUNNING IT

    pip install -e .                    # once, from the repository root
    python examples/mc_recenter_a0.py

At the defaults (20 replications, N = 80) expect
roughly 45-60 minutes. Reduce N_REPS for a first look; the summary is printed
from whatever rows exist, so you can stop it at any point and re-run to see
where it stands.

WHAT TO CONCLUDE

    bias_recen ~ 0, bias_joint > 0      recentring is correct and the flag
                                        should be the default
    bias_joint ~ 0, bias_recen < 0      the rank-one term is compensating a
                                        real downward bias elsewhere -- most
                                        likely the Laplace term in S
                                        under-correcting the attenuation
    both ~ 0                            neither matters at this N; the two
                                        single runs were noise
"""

import csv
import os
import time
from pathlib import Path

import numpy as np

from cbm.individual_fit import individual_fit
from cbm.hbi import hbi_main
from cbm.hbi_config import HBIConfig
from cbm.posterior_correlation import all_linked_correlation_cis

import example_joint_covariance as E

# ---------------------------------------------------------------------------
N_REPS     = 20
N_SUBJECTS = 80         # smaller than the demo's 120, for throughput
NUM_INIT   = 6
TRUE_R     = 0.5
BASE_SEED  = 90000
OUT_DIR    = Path(__file__).resolve().parent / "output_mc_recenter"
OUT_CSV    = OUT_DIR / "mc_recenter_a0.csv"

FIELDS = ["rep", "seed", "n_subjects", "true_r", "realised", "raw_map",
          "r_joint", "r_recen", "offset_ij", "secs"]


# ---------------------------------------------------------------------------
def simulate(seed):
    rng = np.random.default_rng(seed)
    Sigma = np.diag(E.SD_TRUE ** 2)
    c = TRUE_R * E.SD_TRUE[E.BETA_RL] * E.SD_TRUE[E.BETA_RISK]
    Sigma[E.BETA_RL, E.BETA_RISK] = Sigma[E.BETA_RISK, E.BETA_RL] = c

    with np.errstate(**E.FP_QUIET):     # macOS Accelerate flags; see example script
        theta = rng.multivariate_normal(E.MU_TRUE, Sigma, size=N_SUBJECTS)
        data = [(E.simulate_mw(theta[n, :E.N_RL], E.MW_WALKS[n % E.N_STIM], rng),
                 E.simulate_risk(theta[n, E.N_RL:], E.RISK_TRIALS[n % E.N_STIM], rng))
                for n in range(N_SUBJECTS)]
    return theta, data


def fit_once(data, pkl, recenter, seed):
    cfg = HBIConfig(covariance_blocks=[[(E.BETA_RL, E.BETA_RISK)]],
                    auto_nu0=True,
                    recenter_a0=recenter,
                    keep_best_iterate=True,
                    divergence_tol=50.0,
                    save_prog=False, verbose=0, flog=-1)
    tag = "recen" if recenter else "joint"
    with np.errstate(**E.FP_QUIET):
        res = hbi_main(data, [E.joint_model], [pkl],
                       fname=str(OUT_DIR / f"hbi_{tag}_tmp.pkl"), config=cfg)
        ci = all_linked_correlation_cis(
            res, k=0, rng=np.random.default_rng(seed))[(E.BETA_RL, E.BETA_RISK)]
    return res, ci["point"]


def run_replication(rep):
    seed = BASE_SEED + rep
    t0 = time.time()
    theta, data = simulate(seed)

    pkl = str(OUT_DIR / "ind_tmp.pkl")
    with np.errstate(**E.FP_QUIET):
        ind = individual_fit(data, E.joint_model, E.PRIOR_MEAN, E.PRIOR_VAR,
                             fname=pkl,
                             config={"num_init": NUM_INIT, "verbose": False})
    mp = ind.output.parameters

    res_j, r_joint = fit_once(data, pkl, False, seed)
    _,     r_recen = fit_once(data, pkl, True,  seed)

    # size of the rank-one term, using the un-recentred fit's group mean as a
    # stand-in for thetabar (they differ only by the beta0/beta_k shrinkage)
    gm = np.asarray(res_j.output.group_mean[0]).ravel()
    offset = float((gm[E.BETA_RL] - E.PRIOR_MEAN[E.BETA_RL])
                   * (gm[E.BETA_RISK] - E.PRIOR_MEAN[E.BETA_RISK]))

    return {
        "rep": rep, "seed": seed, "n_subjects": N_SUBJECTS, "true_r": TRUE_R,
        "realised": float(np.corrcoef(theta[:, E.BETA_RL],
                                      theta[:, E.BETA_RISK])[0, 1]),
        "raw_map": float(np.corrcoef(mp[:, E.BETA_RL],
                                     mp[:, E.BETA_RISK])[0, 1]),
        "r_joint": float(r_joint), "r_recen": float(r_recen),
        "offset_ij": offset, "secs": round(time.time() - t0, 1),
    }


def load_rows():
    if not OUT_CSV.exists():
        return []
    with open(OUT_CSV, newline="") as f:
        return list(csv.DictReader(f))


def append_row(row):
    new = not OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def summarise():
    rows = load_rows()
    if not rows:
        print("no replications yet")
        return
    A = {k: np.array([float(r[k]) for r in rows]) for k in
         ("realised", "raw_map", "r_joint", "r_recen", "offset_ij")}
    n = len(rows)

    def ms(x):
        return x.mean(), x.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan

    print("\n" + "=" * 72)
    print(f"recenter_a0 Monte Carlo   {n} replications, "
          f"N = {rows[0]['n_subjects']}, imposed r = {rows[0]['true_r']}")
    print("=" * 72)
    print(f"  {'quantity':<34}{'mean':>10}{'SE':>10}")
    for label, key in [("realised r (the target)", "realised"),
                       ("Pearson r of MAP estimates", "raw_map"),
                       ("HBI joint", "r_joint"),
                       ("HBI joint + recenter_a0", "r_recen")]:
        m, s = ms(A[key])
        print(f"  {label:<34}{m:>+10.4f}{s:>10.4f}")

    print(f"\n  {'bias vs the realised value':<34}{'mean':>10}{'SE':>10}{'t':>8}")
    for label, key in [("joint", "r_joint"), ("joint + recenter_a0", "r_recen")]:
        d = A[key] - A["realised"]
        m, s = ms(d)
        print(f"  {label:<34}{m:>+10.4f}{s:>10.4f}{m/s if s else np.nan:>8.2f}")

    d = A["r_joint"] - A["r_recen"]
    m, s = ms(d)
    print(f"\n  {'paired shift, joint - recentred':<34}{m:>+10.4f}{s:>10.4f}"
          f"{m/s if s else np.nan:>8.2f}")
    print(f"  {'mean rank-one offset (i)(j)':<34}{A['offset_ij'].mean():>+10.4f}")
    print("\n  The paired shift is the effect of the flag with simulation noise")
    print("  removed; |t| > ~2 means it is resolved. Read the two bias rows to")
    print("  decide which setting is right -- the one nearer zero.")


# ===========================================================================
if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    done = {int(r["rep"]) for r in load_rows()}
    if done:
        print(f"resuming: {len(done)} replication(s) already in {OUT_CSV}")

    for rep in range(N_REPS):
        if rep in done:
            continue
        row = run_replication(rep)
        append_row(row)
        print(f"  rep {rep:>3}/{N_REPS - 1}   realised {row['realised']:+.3f}"
              f"   joint {row['r_joint']:+.3f}   recentred {row['r_recen']:+.3f}"
              f"   ({row['secs']:.0f} s)")

    summarise()
