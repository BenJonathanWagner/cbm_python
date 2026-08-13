"""
Posterior uncertainty for group-level correlations in joint (covariance_blocks)
HBI fits.

Why this exists: the joint branch reports a single point estimate of the
correlation between two linked parameters (from sigma_k or inv(Etau) --
identical up to scale, which cancels in a correlation). But the fitted
Normal-Wishart posterior q(mu, Lambda) carries a full DISTRIBUTION over the
group covariance, and therefore over every correlation coefficient. For
weakly identified parameter pairs that distribution can be very wide (e.g.
+-0.3), in which case run-to-run scatter of the point estimate is expected
behaviour rather than a defect -- and any single fit's correlation should be
quoted with its credible interval.

Parameterization note: hbi_qmutau's blockwise updates follow the standard
Normal-Wishart equations under the correspondence
    dof   = 2 * nu_k
    scale = (2 * sigma_b)^{-1}
(see the comment above the blockwise ELBO section in hbi_updates.py), so
E[Lambda_b] = 2*nu_k * (2*sigma_b)^{-1} = nu_k * sigma_b^{-1}, matching the
Etau computed there. Sampling therefore uses
    Lambda_b ~ Wishart(df=2*nu_k, scale=(2*sigma_b)^{-1})
    Sigma_b  = Lambda_b^{-1}
    r_ij     = Sigma_ij / sqrt(Sigma_ii * Sigma_jj).

Only pairs inside the same connected covariance block have a modeled
correlation; asking for an unlinked pair raises a ValueError (the model's
posterior for that correlation is the structural zero, not a distribution).
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import wishart

from .hbi_updates import _mask_blocks


def _get_qmutau(result, k: int):
    """Accept either an HBIResult or a bare qmutau-like object list holder."""
    qm = result.math.qmutau[k]
    sigma = np.asarray(qm.sigma, dtype=float)
    if sigma.ndim != 2:
        raise ValueError(
            "This result was fit WITHOUT covariance_blocks (sigma is diagonal-"
            "only); group-level correlations are not modeled, so there is no "
            "posterior to sample. Refit with covariance_blocks to use this."
        )
    nu = float(qm.nu)
    return sigma, nu


def correlation_posterior_samples(
    result,
    i: int,
    j: int,
    k: int = 0,
    n_samples: int = 20000,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Draw samples of the group-level correlation r(i, j) from the fitted
    Normal-Wishart posterior of model k.

    Parameters
    ----------
    result : HBIResult
        A joint-mode (covariance_blocks) hbi_main result.
    i, j : int
        Parameter indices in the model's joint parameter vector. Must lie
        in the same connected covariance block.
    k : int
        Model index (default 0).
    n_samples : int
        Number of posterior draws.
    rng : np.random.Generator, optional
        Source of randomness (default: fresh default_rng()).

    Returns
    -------
    np.ndarray of shape (n_samples,) with correlation draws in [-1, 1].
    """
    if i == j:
        raise ValueError("i and j must be different parameter indices")
    sigma, nu = _get_qmutau(result, k)

    # Recover the block structure directly from sigma's exact-zero pattern
    # (off-block entries are structural zeros by construction in hbi_qmutau).
    blocks = _mask_blocks((sigma != 0).astype(float))
    block = next((b for b in blocks if (i in b) and (j in b)), None)
    if block is None:
        raise ValueError(
            f"parameters {i} and {j} are not linked by any covariance block "
            f"in this fit -- their modeled correlation is a structural zero."
        )

    bix = np.ix_(block, block)
    Db = len(block)
    df = 2.0 * nu
    if df <= Db - 1:
        raise ValueError(
            f"Wishart dof 2*nu={df:.2f} <= Db-1={Db - 1}; posterior is "
            f"improper for this block -- cannot sample."
        )

    sigma_b = np.asarray(sigma[bix], dtype=float)
    sigma_b = (sigma_b + sigma_b.T) / 2.0
    scale = np.linalg.inv(2.0 * sigma_b)
    scale = (scale + scale.T) / 2.0

    if rng is None:
        rng = np.random.default_rng()
    lam = wishart.rvs(df=df, scale=scale, size=n_samples, random_state=rng)
    if lam.ndim == 2:  # size=1 edge case
        lam = lam[np.newaxis, :, :]
    cov = np.linalg.inv(lam)  # batched inverse: (n_samples, Db, Db)

    bl = list(block)
    ii, jj = bl.index(i), bl.index(j)
    r = cov[:, ii, jj] / np.sqrt(cov[:, ii, ii] * cov[:, jj, jj])
    return r


def correlation_credible_interval(
    result,
    i: int,
    j: int,
    k: int = 0,
    ci: float = 0.95,
    n_samples: int = 20000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, float]:
    """
    Credible interval for the group-level correlation r(i, j).

    Returns a dict with:
      point    -- the usual point estimate from sigma (identical to the
                  inv(Etau)-based value, since scale cancels)
      mean     -- posterior mean of r
      median   -- posterior median of r
      ci_low   -- lower quantile bound
      ci_high  -- upper quantile bound
      ci       -- the requested mass (e.g. 0.95)
      prob_pos -- posterior probability that r > 0
    """
    sigma, _ = _get_qmutau(result, k)
    point = float(sigma[i, j] / np.sqrt(sigma[i, i] * sigma[j, j]))
    r = correlation_posterior_samples(result, i, j, k=k, n_samples=n_samples, rng=rng)
    lo, hi = np.quantile(r, [(1.0 - ci) / 2.0, 1.0 - (1.0 - ci) / 2.0])
    return {
        "point": point,
        "mean": float(r.mean()),
        "median": float(np.median(r)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci": float(ci),
        "prob_pos": float(np.mean(r > 0)),
    }


def all_linked_correlation_cis(
    result,
    k: int = 0,
    ci: float = 0.95,
    n_samples: int = 20000,
    rng: Optional[np.random.Generator] = None,
) -> Dict[Tuple[int, int], Dict[str, float]]:
    """
    Credible intervals for EVERY linked (within-block, off-diagonal) pair of
    model k, keyed by (i, j) with i < j. One Wishart sampling pass per block.
    """
    sigma, nu = _get_qmutau(result, k)
    blocks = _mask_blocks((sigma != 0).astype(float))
    if rng is None:
        rng = np.random.default_rng()

    out: Dict[Tuple[int, int], Dict[str, float]] = {}
    for block in blocks:
        Db = len(block)
        if Db < 2:
            continue
        bix = np.ix_(block, block)
        sigma_b = np.asarray(sigma[bix], dtype=float)
        sigma_b = (sigma_b + sigma_b.T) / 2.0
        scale = np.linalg.inv(2.0 * sigma_b)
        scale = (scale + scale.T) / 2.0
        lam = wishart.rvs(df=2.0 * nu, scale=scale, size=n_samples, random_state=rng)
        if lam.ndim == 2:
            lam = lam[np.newaxis, :, :]
        cov = np.linalg.inv(lam)
        d = np.sqrt(np.einsum("nii->ni", cov))
        bl = list(block)
        for a_ in range(Db):
            for b_ in range(a_ + 1, Db):
                i, j = bl[a_], bl[b_]
                if sigma[i, j] == 0:
                    # structural zero inside a merged block (e.g. overlapping
                    # pairs merged into one component): still sampled, but
                    # flag nothing -- the posterior handles it.
                    pass
                r = cov[:, a_, b_] / (d[:, a_] * d[:, b_])
                lo, hi = np.quantile(r, [(1 - ci) / 2, 1 - (1 - ci) / 2])
                # plain ints, not np.int64, so that printing the dict is readable
                out[(int(i), int(j))] = {
                    "point": float(sigma[i, j] / np.sqrt(sigma[i, i] * sigma[j, j])),
                    "mean": float(r.mean()),
                    "median": float(np.median(r)),
                    "ci_low": float(lo),
                    "ci_high": float(hi),
                    "ci": float(ci),
                    "prob_pos": float(np.mean(r > 0)),
                }
    return out
