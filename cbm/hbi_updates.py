from typing import Any, Dict, List, Tuple
import numpy as np
from scipy.special import psi, gammaln, multigammaln
from copy import deepcopy

from .hbi_types import (
    IndividualPosterior,
    GaussianGammaDistribution,
    DirichletDistribution,
    BoundQMutau,
    BoundQM,
    BoundQHZ,
    GaussianDistribution,
    BoundState,
    BoundTerms,
)

# new hbi_sumstats
# either default mode or covariance mask for joint modeling/masked parameters stay independent

def safe_invert(mat: np.ndarray) -> np.ndarray:
    """
    Bypasses Mac SVD hardware crashes by using LU decomposition (inv) 
    instead of SVD (pinv), with adaptive Bayesian jitter.
    """
    # 1. Clean any impossible numbers that slipped through
    if not np.all(np.isfinite(mat)):
        mat = np.nan_to_num(mat, nan=0.0, posinf=1e6, neginf=-1e6)
        
    # 2. Force strict symmetry
    mat = (mat + mat.T) / 2.0
    
    # 3. Adaptive LU Inversion — also check for silent NaN returns (Mac BLAS issue)
    jitter = 1e-8
    for _ in range(10):
        try:
            result = np.linalg.inv(mat + np.eye(mat.shape[0]) * jitter)
            if np.all(np.isfinite(result)):
                return result
            # inv returned NaN/Inf silently (Mac BLAS issue) — increase jitter and retry
        except np.linalg.LinAlgError:
            pass
        jitter *= 100

    # 4. Ultimate fallback if the matrix is completely destroyed
    diag = np.diag(mat).copy()
    diag[diag <= 0] = 1e-8
    return np.diag(1.0 / diag)


def make_positive_definite(mat: np.ndarray, min_eig: float = 1e-6) -> np.ndarray:
    """
    Guarantee a symmetric, strictly positive-definite matrix via eigenvalue
    flooring.

    Why this is needed: zeroing out entries of a PSD matrix (e.g. applying a
    covariance_mask to keep only specific cross-parameter blocks) is NOT
    guaranteed to preserve positive-definiteness — the masked result can be
    singular or indefinite even though the unmasked matrix was fine. A
    singular matrix passes `np.isfinite()` checks (it's not NaN/Inf) but
    makes `np.linalg.slogdet` return -inf (or a negative sign for
    indefinite matrices). That -inf then silently poisons everything
    downstream the moment it's added/subtracted against another finite or
    -inf value (e.g. `rarg = logrho - logrho[k, :]` => -inf - (-inf) = NaN),
    without ever tripping an `isfinite` guard at the point of creation.

    Eigenvalue flooring repairs this directly: decompose, clip every
    eigenvalue up to `min_eig`, and reassemble. The result is provably PD,
    so slogdet/inv/Cholesky on it are always well-defined afterward.

    Hardened the same way safe_invert() already is against the documented
    Mac SVD/BLAS hardware issue (silent NaN/Inf returns from LAPACK on some
    Mac hardware, noted in safe_invert's docstring): retry np.linalg.eigh
    with escalating jitter and an explicit isfinite check on the result,
    since eigh is exactly the kind of decomposition that issue affects, and
    previously this function had none of safe_invert's protection at all.
    """
    if not np.all(np.isfinite(mat)):
        mat = np.nan_to_num(mat, nan=0.0, posinf=1e6, neginf=-1e6)
    mat = (mat + mat.T) / 2.0

    jitter = 0.0
    for _ in range(10):
        try:
            eigvals, eigvecs = np.linalg.eigh(mat + np.eye(mat.shape[0]) * jitter)
            if np.all(np.isfinite(eigvals)) and np.all(np.isfinite(eigvecs)):
                eigvals_clipped = np.clip(eigvals, min_eig, None)
                repaired = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
                repaired = (repaired + repaired.T) / 2.0
                if np.all(np.isfinite(repaired)):
                    return repaired
            # eigh returned NaN/Inf silently (the Mac BLAS issue) -- retry with jitter
        except np.linalg.LinAlgError:
            pass
        jitter = jitter * 10 if jitter > 0 else 1e-8

    # Ultimate fallback if eigh is completely unusable on this matrix:
    # keep only the (floored) diagonal, mirroring safe_invert's fallback.
    diag = np.diag(mat).copy()
    diag = np.where(np.isfinite(diag) & (diag >= min_eig), diag, min_eig)
    return np.diag(diag)


def _cap_eigenvalues(mat: np.ndarray, max_eig: float) -> np.ndarray:
    """
    Cap a symmetric matrix's eigenvalues at `max_eig` -- the mirror image of
    make_positive_definite's flooring, used here as a ceiling instead.

    Why this is needed: in hbi_sumstats' joint branch, theta_k (each
    subject's MAP parameters) is already clipped to +-1e4 before entering the
    outer product, guarding against a runaway individual fit. But the
    per-subject inverse-Hessian (Ainv_k, i.e. that subject's approximate
    posterior covariance) had no equivalent cap -- only a post-hoc
    isfinite() check that catches NaN/Inf but not a large-but-finite
    pathological value (e.g. a subject with a near-flat likelihood along one
    weakly-identified parameter). Under a small covariance block that value
    would only distort its own small pair; under a large merged block (e.g.
    WITHIN_TASK_MODE="full", which links every parameter into one block) it
    can distort every OTHER parameter's precision too, since the block's
    eigendecomposition (used for PD-repair) mixes all of the block's
    coordinates together. Capping bounds how much any single subject can
    dominate the joint sufficient statistics, while leaving well-conditioned
    subjects completely untouched.
    """
    if not np.all(np.isfinite(mat)):
        mat = np.nan_to_num(mat, nan=0.0, posinf=max_eig, neginf=0.0)
    mat = (mat + mat.T) / 2.0

    jitter = 0.0
    for _ in range(10):
        try:
            eigvals, eigvecs = np.linalg.eigh(mat + np.eye(mat.shape[0]) * jitter)
            if np.all(np.isfinite(eigvals)) and np.all(np.isfinite(eigvecs)):
                eigvals_capped = np.clip(eigvals, None, max_eig)
                capped = eigvecs @ np.diag(eigvals_capped) @ eigvecs.T
                capped = (capped + capped.T) / 2.0
                if np.all(np.isfinite(capped)):
                    return capped
        except np.linalg.LinAlgError:
            pass
        jitter = jitter * 10 if jitter > 0 else 1e-8

    diag = np.diag(mat).copy()
    diag = np.where(np.isfinite(diag), diag, 0.0)
    diag = np.clip(diag, None, max_eig)
    return np.diag(diag)


def _mask_blocks(mask: np.ndarray) -> List[np.ndarray]:
    """
    Decompose a binary covariance mask into connected components (blocks).

    The joint branch treats q(mu, Lambda) as factorizing over these blocks:
    linked parameters form a small Normal-Wishart block, and singleton
    blocks reduce exactly to the Gaussian-Gamma of the default branch.
    All Wishart expectations and ELBO terms must therefore be computed
    blockwise — treating the masked matrix as one full D-dimensional
    Wishart uses the wrong multivariate digamma/gamma terms.

    Note: overlapping pairs, e.g. (0,1) and (1,2), merge into one block
    {0,1,2}; the (0,2) entry then stays a structural zero inside that
    block's scale matrix, which is an approximation to exact conjugacy.

    Returns a list of sorted index arrays, one per block.
    """
    D = mask.shape[0]
    adj = (mask != 0)
    seen = np.zeros(D, dtype=bool)
    blocks: List[np.ndarray] = []
    for start in range(D):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = [start]
        while stack:
            i = stack.pop()
            for j in np.flatnonzero(adj[i]):
                if not seen[j]:
                    seen[j] = True
                    stack.append(j)
                    comp.append(j)
        blocks.append(np.array(sorted(comp), dtype=int))
    return blocks

def hbi_sumstats(
    r: np.ndarray, 
    qh: IndividualPosterior,
    covariance_mask: List[np.ndarray] = None
) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
    
    theta = qh.parameters
    K, N = r.shape
    
    thetabar: List[np.ndarray] = [None] * K
    S_out: List[np.ndarray] = [None] * K
    Nbar = np.zeros(K, dtype=float)

    for k in range(K):
        r_k = r[k, :]
        Nk = float(r_k.sum())
        Nbar[k] = Nk

        # Guard: if a model has collapsed to zero responsibility (Nk≈0),
        # skip its parameter update entirely — thetabar and S stay zero,
        # which is equivalent to the model contributing nothing to the
        # group-level sufficient statistics this iteration.
        theta_k = theta[k]
        D_k = theta_k.shape[0]
        if Nk < 1e-8:
            thetabar[k] = np.zeros(D_k)
            if covariance_mask is not None and covariance_mask[k] is not None:
                S_out[k] = np.zeros((D_k, D_k))
            else:
                S_out[k] = np.zeros(D_k)
            continue

        # theta_k shape: (D, N)
        thetabar_k = np.sum(theta_k * r_k[np.newaxis, :], axis=1) / Nk #new
        thetabar[k] = thetabar_k

        if covariance_mask is not None and covariance_mask[k] is not None:
            # when joint modeling is applied

            # Weighted outer product of subject parameters.
            # Sanitize theta first: replace any non-finite value with 0 so
            # that overflow/NaN from a bad individual fit doesn't poison the
            # group-level covariance estimate.
            # Clip theta before outer product to prevent overflow
            # (individual-fit parameters can occasionally be large-but-finite)
            theta_k_safe = np.where(np.isfinite(theta_k), theta_k, 0.0)
            theta_k_safe = np.clip(theta_k_safe, -1e4, 1e4)
            param_outer = (theta_k_safe * r_k) @ theta_k_safe.T

            # Weighted sum of the full inverse Hessians
            # Assumes qh.hessian_inv is a list of (D, D, N) arrays
            #
            # Cap each subject's inverse-Hessian eigenvalues before summing
            # (see _cap_eigenvalues docstring) -- closes the one asymmetric
            # gap versus theta_k_safe's clip above, and matters most once
            # many parameters share one merged covariance block.
            AINV_MAX_EIG = 1e4  # matches theta_k_safe's clip order of magnitude
            Ainv_k = np.asarray(qh.hessian_inv[k], dtype=float).copy()
            for _n in range(Ainv_k.shape[2]):
                Ainv_k[:, :, _n] = _cap_eigenvalues(Ainv_k[:, :, _n], AINV_MAX_EIG)
            hessian_sum = np.sum(Ainv_k * r_k[np.newaxis, np.newaxis, :], axis=2)
            if not np.all(np.isfinite(hessian_sum)):
                hessian_sum = np.nan_to_num(hessian_sum, nan=0.0, posinf=0.0, neginf=0.0)

            # Combine, divide by Nk, and center it
            tb_col = thetabar_k.reshape(-1, 1)
            S_k = (param_outer + hessian_sum) / Nk - (tb_col @ tb_col.T)

            # Sanitize S_k: NaN/Inf here propagates to sigma_k → slogdet → r → Nk
            # → nu_k = NaN, which breaks everything from iteration 2 onwards.
            if not np.all(np.isfinite(S_k)):
                S_k = np.nan_to_num(S_k, nan=0.0, posinf=0.0, neginf=0.0)

            # Apply the binary mask to isolate specific covariance blocks
            # S_k * mask performs element-wise multiplication, zeroing out unlinked parameters
            S_out[k] = S_k * covariance_mask[k]
            
        else:
            # default cbm
            Ainvdiag_k = qh.hessian_inv_diag[k]
            Sdiag_k = (
                np.sum((theta_k ** 2 + Ainvdiag_k) * r_k[np.newaxis, :], axis=1) / Nk  # <--- keepdims=True removed!
                - thetabar_k ** 2
            )
            S_out[k] = Sdiag_k

    return Nbar, thetabar, S_out

# hbi_qmutau

def hbi_qmutau(
    pmutau: List[GaussianGammaDistribution],
    Nbar: np.ndarray,
    thetabar: List[np.ndarray],
    S_in: List[np.ndarray],  # Previously Sdiag, now holds the full S matrix
    covariance_mask: List[np.ndarray] = None,
    auto_nu0: bool = False,
    nu0_margin: float = 1.0,
) -> Tuple[List[GaussianGammaDistribution], BoundQMutau]:

    K = len(Nbar)
    ElogpH = np.zeros(K)   # zero (not nan) so collapsed models contribute 0 to bound
    Elogpmu = np.zeros(K)
    Elogqmu = np.zeros(K)
    Elogptau = np.zeros(K)
    Elogqtau = np.zeros(K)
    qmutau_out: List[GaussianGammaDistribution] = []

    for k in range(K):
        a0k = np.asarray(pmutau[k].a, dtype=float)
        beta0k = float(pmutau[k].beta)
        nu0k = float(pmutau[k].nu)
        sigma0k = np.asarray(pmutau[k].sigma, dtype=float)
        Nk = float(Nbar[k])
        tb_k = thetabar[k]
        S_k = S_in[k]
        Dk = len(a0k)

        # Precompute this model's covariance blocks once (reused below) and,
        # if auto_nu0 is on, raise nu0k (never lower it) so the Wishart prior
        # stays proper for the LARGEST linked block: nu0 = max(nu0_default,
        # (Db_max + nu0_margin) / 2). nu0k/nu_k stay plain scalars throughout
        # -- only the VALUE changes -- so every downstream consumer that
        # reads qmutau[k].nu as a float (including outside this file) is
        # unaffected when auto_nu0=False (the default).
        blocks_k = None
        if covariance_mask is not None and covariance_mask[k] is not None:
            blocks_k = _mask_blocks(covariance_mask[k])
            if auto_nu0 and len(blocks_k) > 0:
                Db_max = max(len(b) for b in blocks_k)
                nu0k = max(nu0k, (Db_max + nu0_margin) / 2.0)

        # Mean and Degrees of Freedom update (Identical for 1D and 2D)
        beta_k = beta0k + Nk
        a_k = (beta0k * a0k + Nk * tb_k) / beta_k
        nu_k = nu0k + 0.5 * Nk

        # If model has zero responsibility, reset to prior and zero all ELBO terms.
        # Avoids NaN from 0/0 and discontinuous ELBO jumps when a model collapses.
        if Nk < 1e-8:
            a_k     = a0k.copy()
            beta_k  = beta0k
            nu_k    = nu0k
            sigma0k_mat = np.diagflat(sigma0k) if sigma0k.ndim == 1 else sigma0k.copy()
            sigma_k = sigma0k_mat.copy() if (covariance_mask is not None and covariance_mask[k] is not None) else sigma0k.copy()
            Etau_k  = np.eye(Dk) if (covariance_mask is not None and covariance_mask[k] is not None) else np.ones(Dk)
            Elogtau_k = np.zeros(1) if (covariance_mask is not None and covariance_mask[k] is not None) else np.zeros(Dk)
            logG_k  = 0.0
            # ELBO terms stay at 0 (initialized above) — collapsed model contributes nothing
            qmutau_out.append(GaussianGammaDistribution(
                a=a_k, beta=beta_k, sigma=sigma_k, nu=nu_k,
                Etau=Etau_k, Elogtau=Elogtau_k, logG=logG_k,
            ))
            continue

        if covariance_mask is not None and covariance_mask[k] is not None:
            # --- JOINT MODELING (NORMAL-WISHART MATRIX MATH) ---
            diff = tb_k.reshape(-1, 1) - a0k.reshape(-1, 1)
            
            sigma0k_mat = np.diagflat(sigma0k) if sigma0k.ndim == 1 else sigma0k
            
            ## 1. Update the inverse scale matrix
            sigma_k = sigma0k_mat + 0.5 * (Nk * S_k + (Nk * beta0k / beta_k) * (diff @ diff.T))

            # 2. Re-apply the mask to zero-out the dense diff outer product!
            #    NOTE: masking a PSD matrix is NOT guaranteed to keep it PSD —
            #    it can become singular/indefinite even when sigma_k before
            #    masking was perfectly fine. That singularity doesn't show up
            #    as NaN/Inf (np.isfinite passes), it shows up two steps later
            #    as slogdet returning -inf, which then poisons everything
            #    downstream the moment it's subtracted from another value.
            sigma_k = sigma_k * covariance_mask[k]

            # 3. Decompose the mask into independent blocks (connected
            #    components) and repair each block via eigenvalue flooring.
            #    Blockwise flooring GUARANTEES strict positive-definiteness
            #    while keeping off-block entries exactly zero, so the block
            #    factorization of q(Lambda) is preserved (full-matrix
            #    flooring could silently re-introduce masked-out couplings).
            blocks = blocks_k  # precomputed above (also used for auto_nu0)
            for idx in blocks:
                bix = np.ix_(idx, idx)
                sigma_k[bix] = make_positive_definite(sigma_k[bix], min_eig=1e-6)

            # 4. E[Lambda] = nu_k * sigma_k^{-1}, inverted blockwise so that
            #    off-block entries stay exactly zero.
            Etau_k = np.zeros_like(sigma_k)
            for idx in blocks:
                bix = np.ix_(idx, idx)
                Etau_k[bix] = nu_k * safe_invert(sigma_k[bix])

            # 5. Force strict symmetry
            Etau_k = (Etau_k + Etau_k.T) / 2.0

            # 6. Guard against NaN/Inf in Etau_k (belt-and-suspenders; should
            #    no longer trigger now that sigma_k is guaranteed PD)
            if not np.all(np.isfinite(Etau_k)):
                Etau_k = np.eye(Dk)

            # Blockwise expected log-determinant and Wishart normalizers.
            # The sufficient-statistic updates above (beta_k, a_k, nu_k,
            # sigma_k with the 0.5 factors) are exactly the standard
            # Normal-Wishart updates (Bishop 10.59-10.63) under the
            # correspondence  dof = 2*nu_k, scale = (2*sigma_b)^{-1},
            # applied independently per mask block. Hence for each block b
            # of dimension Db:
            #   E[log|Lambda_b|] = sum_{i=1..Db} psi(nu_k + (1-i)/2) - log|sigma_b|
            #   log-normalizer   = nu_k*log|sigma_b| - multigammaln(nu_k, Db)
            # Both reduce exactly to the Gamma-branch formulas for Db=1.
            # (The previous full-D formula had a spurious +Dk*log(2) and used
            # full-D digamma/gamma terms, which biased the bound and, with
            # K>1, the responsibilities of masked models.)
            ElogdetT = 0.0
            logG_k = 0.0
            Elogptau_blocks = 0.0
            Elogqtau_blocks = 0.0
            for idx in blocks:
                Db = len(idx)
                bix = np.ix_(idx, idx)
                sign_b, ld_b = np.linalg.slogdet(sigma_k[bix])
                if sign_b <= 0 or not np.isfinite(ld_b):
                    ld_b = float(np.sum(np.log(np.clip(np.diag(sigma_k[bix]), 1e-12, None))))
                psi_b = float(np.sum([psi(nu_k + 0.5 * (1 - i)) for i in range(1, Db + 1)]))
                ElogdetT_b = psi_b - ld_b
                ElogdetT += ElogdetT_b

                # Posterior normalizer: nu_k > (Db-1)/2 holds whenever the
                # model carries responsibility mass; guard anyway.
                try:
                    logG_b = nu_k * ld_b - multigammaln(nu_k, Db)
                except ValueError:
                    logG_b = 0.0
                logG_k += logG_b

                # Prior normalizer: with the default hyperprior v=0.5 the
                # implied block-Wishart prior (dof 2v=1) is improper for
                # Db>=2 (multigammaln pole). It is a constant of the model,
                # so it is dropped for those blocks; use v > (Db-1)/2 + eps
                # (e.g. v=1.0) if a fully proper ELBO is needed, e.g. for
                # comparing joint vs non-joint models via L or BOR.
                sign0_b, ld0_b = np.linalg.slogdet(sigma0k_mat[bix])
                if sign0_b <= 0 or not np.isfinite(ld0_b):
                    ld0_b = float(np.sum(np.log(np.clip(np.diag(sigma0k_mat[bix]), 1e-12, None))))
                try:
                    logG0_b = nu0k * ld0_b - multigammaln(nu0k, Db)
                except ValueError:
                    logG0_b = 0.0

                # Per-block ElogdetT coefficients of E[log p(Lambda)] and
                # E[log q(Lambda)]; trace terms are added outside the loop.
                Elogptau_blocks += ((2.0 * nu0k - Db - 1.0) / 2.0) * ElogdetT_b + logG0_b
                Elogqtau_blocks += ((2.0 * nu_k - Db - 1.0) / 2.0) * ElogdetT_b + logG_b
            
            # Stored as a 1-element array so downstream np.sum() works seamlessly
            Elogtau_k = np.array([ElogdetT])




            # E_q[log p(mu | Lambda)]: Normal prior on group mean
            diff_a = (a_k - a0k).reshape(-1, 1)
            # Written as an elementwise sum rather than float(x.T @ A @ x):
            # the triple product is a (1,1) array, and converting an ndim>0
            # array to a Python scalar is deprecated from NumPy 1.25 and an
            # error from NumPy 2.3 onward. Same style as the trace terms below.
            quad_mu0 = float(np.sum(diff_a * (Etau_k @ diff_a)))
            Elogpmu[k] = (
                (Dk / 2) * np.log(beta0k / (2 * np.pi))
                + 0.5 * ElogdetT
                - (beta0k / 2) * (Dk / beta_k + quad_mu0)
            )

            # E_q[log p(Lambda)]: blockwise Wishart prior on precision.
            # ElogdetT coefficients and prior normalizers were accumulated
            # per block above; the trace term tr(sigma0 E[Lambda]) equals
            # the full elementwise sum because Etau_k is exactly
            # block-diagonal. Reduces to
            # (nu0-1)*ElogdetT - sum(sigma0*Etau) + logG0 for all-singleton
            # masks, matching the default Gaussian-Gamma branch.
            Elogptau[k] = Elogptau_blocks - float(np.sum(sigma0k_mat * Etau_k))

            # E_q[log q(mu | Lambda)]: entropy of variational Normal
            Elogqmu[k] = (
                (Dk / 2) * np.log(beta_k / (2 * np.pi))
                + 0.5 * ElogdetT
                - Dk / 2
            )

            # E_q[log q(Lambda)]: blockwise entropy term of the variational
            # Wishart. tr(sigma_b E[Lambda_b]) = nu_k * Db per block, which
            # sums to nu_k * Dk. Reduces to (nu-1)*ElogdetT - nu*Dk + logG
            # for all-singleton masks, matching the default branch.
            Elogqtau[k] = Elogqtau_blocks - nu_k * Dk

            # E_q[log p(H | mu, Lambda)] at the qmutau stage (same trace
            # trick as in hbi_qHZ, with the current sufficient statistics).
            # Previously left at 0 in the joint branch, which distorted the
            # intermediate bound reported right after the qmutau step.
            diff_h = tb_k.reshape(-1, 1) - a_k.reshape(-1, 1)
            trace_penalty = float(np.sum(Etau_k * (S_k + diff_h @ diff_h.T)))
            ElogpH[k] = (
                0.5 * Nk * ElogdetT
                - 0.5 * Nk * Dk * np.log(2 * np.pi)
                - 0.5 * Nk * Dk / beta_k
                - 0.5 * Nk * trace_penalty
            )

        else:
            # --- DEFAULT CBM (GAUSSIAN-GAMMA 1D MATH) ---
            sigma_k = sigma0k + 0.5 * (
                Nk * S_k + Nk * beta0k / beta_k * (tb_k - a0k) ** 2
            )
            Elogtau_k = psi(nu_k) - np.log(sigma_k)
            Etau_k = nu_k / sigma_k
            logG_k = np.sum(-gammaln(nu_k) + nu_k * np.log(sigma_k))
            ElogdetT = np.sum(Elogtau_k)
            
            
            
        
        # --- ELBO Bound Calculations (1D case only; joint case computed above) ---
        if covariance_mask is None or covariance_mask[k] is None:
            logG0 = np.sum(-gammaln(nu0k) + nu0k * np.log(sigma0k))
            diff_a = a_k - a0k
            quad_term = beta0k * np.sum(Etau_k * diff_a ** 2)
            Elogpmu[k] = (
                -Dk / 2 * np.log(2 * np.pi) + 0.5 * Dk * np.log(beta0k)
                + 0.5 * ElogdetT - 0.5 * quad_term - Dk / 2 * beta0k / beta_k
            )
            Elogptau[k] = (nu0k - 1) * ElogdetT - np.sum(sigma0k * Etau_k) + logG0
            Elogqmu[k] = (
                -Dk / 2 * np.log(2 * np.pi) + 0.5 * Dk * np.log(beta_k)
                + 0.5 * ElogdetT - Dk / 2
            )
            Elogqtau[k] = (nu_k - 1) * ElogdetT - Dk * nu_k + logG_k
            ElogpH[k] = (
                0.5 * Nk * ElogdetT - 0.5 * Nk * Dk * np.log(2 * np.pi)
                - 0.5 * Nk * Dk / beta_k
                - 0.5 * np.sum(Etau_k * (Nk * S_k + Nk * (tb_k - a_k) ** 2))
            )

        qmutau_out.append(
            GaussianGammaDistribution(
                a=a_k,
                beta=beta_k,
                sigma=sigma_k,
                nu=nu_k,
                Etau=Etau_k,
                Elogtau=Elogtau_k,
                logG=logG_k,
            )
        )
        
    bound = BoundQMutau(
        ElogpH=ElogpH, Elogpmu=Elogpmu, Elogptau=Elogptau, 
        Elogqmu=Elogqmu, Elogqtau=Elogqtau
    )
    return qmutau_out, bound

# hbi_qm

def hbi_qm(pm: DirichletDistribution, Nbar: np.ndarray) -> Tuple[DirichletDistribution, BoundQM]:
    limInf = bool(pm.limInf)
    logC0 = float(pm.logC)
    alpha0 = np.asarray(pm.alpha, dtype=float)
    alpha = alpha0 + Nbar
    alpha_star = np.sum(alpha)
    if ~np.isfinite(alpha_star):
        Elogm = np.nan * np.ones_like(alpha)
        logC = np.nan
    else:
        Elogm = psi(alpha) - psi(alpha_star)
        loggamma = gammaln(alpha)
        logC = gammaln(alpha_star) - np.sum(loggamma)
    Elogpm = logC0 + np.sum((alpha0 - 1) * Elogm)
    Elogqm = logC + np.sum((alpha - 1) * Elogm)
    ElogpZ = Nbar * Elogm
    if limInf:
        K = len(alpha)
        alpha = np.full(K, np.inf)
        Elogm = np.log(np.ones(K) / K)
        logC = np.inf
        Elogpm = np.nan
        Elogqm = np.nan
        ElogpZ = Nbar * Elogm
    qm = DirichletDistribution(
        limInf=limInf,
        alpha=alpha,
        Elogm=Elogm,
        logC=logC,
    )
    bound = BoundQM(
        ElogpZ=ElogpZ,
        Elogpm=Elogpm,
        Elogqm=Elogqm,
    )
    return qm, bound

# hbi_qHZ

def hbi_qHZ(
    qmutau: List[GaussianGammaDistribution],
    qm: DirichletDistribution,
    qh: IndividualPosterior,
    thetabar: List[np.ndarray],
    S_in: List[np.ndarray], # Changed name from Sdiag to accept full matrices
    covariance_mask: List[np.ndarray] = None
) -> Tuple[np.ndarray, BoundQHZ]:
    
    qmlimInf = bool(qm.limInf)
    logf = np.asarray(qh.loglik, dtype=float)
    logdetA = np.asarray(qh.log_det_hessian, dtype=float)
    K, N = logf.shape
    r = np.zeros((K, N), dtype=float)
    ElogpH = np.full(K, np.nan)
    ElogpZ = np.full(K, np.nan)
    ElogpX = np.full(K, np.nan)
    ElogqH = np.full(K, np.nan)
    ElogqZ = np.full(K, np.nan)
    D = np.array([len(qmutau[k].a) for k in range(K)], dtype=float)
    ElogdetT = np.array([np.sum(qmutau[k].Elogtau) for k in range(K)], dtype=float)
    
    # Safely handle the log determinant of the expected precision
    logdetET_list = []
    for k in range(K):
        if covariance_mask is not None and covariance_mask[k] is not None:
            Etau_k = qmutau[k].Etau
            sign, ldet = np.linalg.slogdet(Etau_k)
            # Defensive guard: Etau should be PD by construction now that
            # sigma_k is repaired via eigenvalue flooring in hbi_qmutau, but
            # guard anyway — a -inf/NaN here silently poisons `rarg` below
            # via -inf - (-inf) = NaN without ever tripping an isfinite check
            # at its point of creation.
            if sign <= 0 or not np.isfinite(ldet):
                ldet = float(np.sum(np.log(np.clip(np.diag(Etau_k), 1e-12, None))))
            logdetET_list.append(ldet)
        else:
            logdetET_list.append(np.sum(np.log(qmutau[k].Etau)))
    logdetET = np.array(logdetET_list, dtype=float)

    beta = np.array([qmutau[k].beta for k in range(K)], dtype=float)
    lambda_vec = 0.5 * ElogdetT - 0.5 * logdetET - 0.5 * D / beta
    shift = 0.5 * D * np.log(2 * np.pi) + lambda_vec + qm.Elogm
    logrho = logf - 0.5 * logdetA
    logrho = logrho + shift[:, np.newaxis]

    # Final safety net: any remaining -inf/inf/NaN here (e.g. from a
    # degenerate logdetA) would otherwise silently poison `rarg` below via
    # inf - inf = NaN, which cascades into r -> Nbar -> nu_k -> Etau every
    # iteration from this point on. Replace with a very low (not -inf,
    # not NaN) log-density so the affected model/subject is effectively
    # assigned ~0 responsibility instead of corrupting the whole batch.
    if not np.all(np.isfinite(logrho)):
        logrho = np.nan_to_num(logrho, nan=-1e10, posinf=1e10, neginf=-1e10)

    if qmlimInf:
        r[:, :] = 1.0 / K
    else:
        for k in range(K):
            rarg = logrho - logrho[k, :][np.newaxis, :]
            # Clip before exp to prevent overflow when one model dominates
            r[k, :] = 1.0 / np.sum(np.exp(np.clip(rarg, -500, 500)), axis=0)
            
    logeps = np.exp(np.log1p(-1 + np.finfo(float).eps))
    
    for k in range(K):
        Nk = float(r[k, :].sum())
        Dk = D[k]
        ElogdetT_k = ElogdetT[k]
        beta_k = beta[k]
        Etau_k = qmutau[k].Etau
        a_k = qmutau[k].a
        S_k = S_in[k]
        tb_k = thetabar[k]
        
        if covariance_mask is not None and covariance_mask[k] is not None:
            # --- JOINT MODELING (THE TRACE TRICK) ---
            
            # 1. Outer product for the subject mean distance
            diff = tb_k.reshape(-1, 1) - a_k.reshape(-1, 1)
            mean_dist_matrix = diff @ diff.T
            
            # 2. Element-wise multiply with E[Lambda] and sum 
            # (Mathematically identical to the Trace)
            trace_penalty = np.sum(Etau_k * (S_k + mean_dist_matrix))
            
            ElogpH[k] = (
                0.5 * Nk * ElogdetT_k
                - 0.5 * Nk * Dk * np.log(2 * np.pi)
                - 0.5 * Nk * Dk / beta_k
                - 0.5 * Nk * trace_penalty
            )
        else:
            # --- DEFAULT CBM (1D MATH) ---
            Sd_k = S_k.ravel()
            tb_k_ravel = tb_k.ravel()
            a_k_ravel = a_k.ravel()
            
            ElogpH[k] = (
                0.5 * Nk * ElogdetT_k
                - 0.5 * Nk * Dk * np.log(2 * np.pi)
                - 0.5 * Nk * Dk / beta_k
                - 0.5 * Nk * np.sum(Etau_k * (Sd_k + (tb_k_ravel - a_k_ravel) ** 2))
            )
            
        ElogpZ[k] = Nk * qm.Elogm[k]
        ElogpXH = np.sum(r[k, :] * (logf[k, :] - 0.5 * Dk + lambda_vec[k]))
        ElogpX[k] = ElogpXH - ElogpH[k]
        r_k = r[k, :]
        # Use np.where to avoid evaluating log(0) — 0*log(0) = 0 by convention
        rlogr = np.where(r_k > logeps, r_k * np.log(r_k), 0.0)
        ElogqH[k] = np.sum(
            r_k * (-Dk / 2 - Dk / 2 * np.log(2 * np.pi) + 0.5 * logdetA[k, :])
        )
        ElogqZ[k] = np.sum(rlogr)
        
    bound = BoundQHZ(
        ElogpX=ElogpX, ElogpH=ElogpH, ElogpZ=ElogpZ,
        ElogqH=ElogqH, ElogqZ=ElogqZ,
    )
    return r, bound


# hbi_qhquad (moved from hbi_all)

def hbi_qhquad(
    models: List[Any],
    data: List[Any],
    pconfig: List[Dict[str, Any]],
    qmutau: List[GaussianGammaDistribution],
    qh: IndividualPosterior,
    fid,
) -> IndividualPosterior:
    N = len(data)
    K = len(models)
    verbose_vec = np.zeros(K, dtype=int)
    if not np.all(verbose_vec > 0):
        fid = None
    theta_list = []
    Ainvdiag_list = []
    Ainv_list = []  # <--- NEW: List to hold the full matrices for all models
    logf = np.zeros((K, N), dtype=float)
    flag = np.zeros((K, N), dtype=int)
    logdetA = np.zeros((K, N), dtype=float)
    for k in range(K):
        a_k = np.asarray(qmutau[k].a)
        Etau_k = np.asarray(qmutau[k].Etau)
        Dk = len(a_k)
        
        # --- NEW: Safely handle 1D vs 2D expected precisions ---
        if Etau_k.ndim == 1:
            prior_prec = np.diagflat(Etau_k)
        else:
            prior_prec = Etau_k

        # Guard: if the precision matrix contains NaN/Inf (e.g. from a
        # degenerate Normal-Wishart update in the first iteration), fall back
        # to the identity so optimize_map receives a valid prior.
        if not np.all(np.isfinite(prior_prec)):
            prior_prec = np.eye(Dk)
        a_k_safe = np.where(np.isfinite(a_k), a_k, 0.0)
        prior = GaussianDistribution(mean=a_k_safe, precision=prior_prec)
        # -------------------------------------------------------
        
        
        cfg = deepcopy(pconfig[k])
        theta_k = np.zeros((Dk, N), dtype=float)
        Ainvdiag_k = np.zeros((Dk, N), dtype=float)
        
        Ainv_full_k = np.zeros((Dk, Dk, N), dtype=float) # <--- NEW: 3D array for full matrices
        
        for n in range(N):
            cfg.inits = qh.parameters[k][:, n]
            from .map_estimation import optimize_map, log_posterior
            
            logf_kn, theta_kn, A_kn, _, flag_kn = optimize_map(
                data[n], models[k], cfg, prior.mean.flatten(), prior.precision, 'LAP'
            )
            
            # =====================================================================
            # --- NEW: Ultimate Scrubber & Jitter ---
            # =====================================================================
            is_invalid = (
                not np.all(np.isfinite(theta_kn)) or 
                not np.all(np.isfinite(A_kn)) or
                not np.isfinite(logf_kn)
            )
            is_runaway = np.any(np.abs(theta_kn) > 1000.0)
            
            if flag_kn == 0 or is_invalid or is_runaway:
                # Revert to the stable group prior to prevent matrix poisoning
                theta_kn = prior.mean.flatten()
                A_kn = prior.precision
                try:
                    logf_kn = log_posterior(theta_kn, models[k], data[n], prior.mean.flatten(), prior.precision)
                    if not np.isfinite(logf_kn):
                        logf_kn = -1e6
                except Exception:
                    logf_kn = -1e6  # Ultimate fallback
            
            # Force strict symmetry and add jitter to prevent -inf log-determinants
            A_kn = (A_kn + A_kn.T) / 2.0
            A_kn = A_kn + np.eye(Dk) * 1e-8
            # =====================================================================
            
            logf[k, n] = logf_kn
            theta_k[:, n] = theta_kn
            flag[k, n] = flag_kn
            
            # --- NEW: Robust log-determinant and safe LU inverse ---
            # --- NEW: Robust log-determinant and safe LU inverse ---
            sign, logdetA_kn = np.linalg.slogdet(A_kn)
            if not np.isfinite(logdetA_kn):
                logdetA_kn = 0.0
                
            Ainv = safe_invert(A_kn)  # <--- FIXED: No more SVD crash here!
            
            Ainvdiag_k[:, n] = np.diag(Ainv)
            Ainv_full_k[:, :, n] = Ainv  # Save the full matrix for this subject
            
            logdetA[k, n] = logdetA_kn
            
        Ainvdiag_list.append(Ainvdiag_k)
        Ainv_list.append(Ainv_full_k)  # <--- NEW: Append to the master list
        theta_list.append(theta_k)
        
    qh_new = IndividualPosterior(
        loglik=logf,
        parameters=theta_list,
        hessian_inv_diag=Ainvdiag_list,
        log_det_hessian=logdetA,
        hessian_inv=Ainv_list  # <--- NEW: Add to the returned dataclass
    )
    return qh_new

# hbi_bound (moved from hbi_all)

def hbi_bound(bound: BoundState, lastmodule: str) -> Tuple[BoundState, float]:
    bb = bound.bound
    pmlimInf = bool(bb.pmlimInf)
    Elogpm_Elogqm = bb.Elogpm - bb.Elogqm
    if pmlimInf:
        Elogpm_Elogqm = 0.0
    L_pre = (
        bb.ElogpX
        + bb.ElogpH
        + bb.ElogpZ
        + bb.Elogpmu
        + bb.Elogptau
        - bb.ElogqH
        - bb.ElogqZ
        - bb.Elogqmu
        - bb.Elogqtau
        + Elogpm_Elogqm
    )
    if lastmodule == "qHZ":
        bh = bound.qHZ
        bb.ElogpX = float(np.sum(bh.ElogpX))
        bb.ElogpH = float(np.sum(bh.ElogpH))
        bb.ElogpZ = float(np.sum(bh.ElogpZ))
        bb.ElogqH = float(np.sum(bh.ElogqH))
        bb.ElogqZ = float(np.sum(bh.ElogqZ))
    elif lastmodule == "qmutau":
        bq = bound.qmutau
        bb.ElogpH = float(np.sum(bq.ElogpH))
        bb.Elogpmu = float(np.sum(bq.Elogpmu))
        bb.Elogptau = float(np.sum(bq.Elogptau))
        bb.Elogqmu = float(np.sum(bq.Elogqmu))
        bb.Elogqtau = float(np.sum(bq.Elogqtau))
    elif lastmodule == "qm":
        bm = bound.qm
        bb.ElogpZ = float(np.sum(bm.ElogpZ))
        bb.Elogpm = float(bm.Elogpm)
        bb.Elogqm = float(bm.Elogqm)
    Elogpm_Elogqm = bb.Elogpm - bb.Elogqm
    if pmlimInf:
        Elogpm_Elogqm = 0.0
    L = (
        bb.ElogpX
        + bb.ElogpH
        + bb.ElogpZ
        + bb.Elogpmu
        + bb.Elogptau
        - bb.ElogqH
        - bb.ElogqZ
        - bb.Elogqmu
        - bb.Elogqtau
        + Elogpm_Elogqm
    )
    dL = float(L - L_pre)
    bb.lastmodule = lastmodule
    bb.L = float(L)
    bb.dL = dL
    return bound, dL
