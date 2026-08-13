# Joint (cross-task) hierarchical modelling for `cbm_python`

**A fork of `payampiray/cbm_python` that lets the HBI group level carry a
non-diagonal covariance over a user-specified set of parameter pairs.**

This document is a complete, file-by-file account of every difference between
this fork and the upstream Python port, written so that the changes can be
reviewed without diffing the code. Sections 7 and 8 list the places where we
are *not* confident and would value your judgement.

Contact: Ben Wagner (postdoc with Tobias Hauser and Peter Dayan;
ben.wagner@tuebingen.mpg.de).

---

## 0. Provenance of the baseline

Everything below is diffed against the following exact revision of
`https://github.com/payampiray/cbm_python`:

```
repo    payampiray/cbm_python
branch  main
commit  ccb8aa8cd03a0c3edd30fd6d39de39511f6a4d6e   ("Update examples", 2025-12-16)
tree    5d9fc1540a58479b5fae2aef97478dd5c2a8a7d4
```

| file | upstream blob SHA | upstream bytes | this fork, bytes | changed? |
|---|---|---:|---:|---|
| `cbm/__init__.py`        | `a428d52` | 142 | 283 | yes (3 new exports) |
| `cbm/hbi.py`             | `aec475b` | 20 608 | 33 923 | **yes, substantially** |
| `cbm/hbi_bound.py`       | `316401f` | 1 694 | 1 694 | no |
| `cbm/hbi_config.py`      | `e855928` | 3 918 | 10 712 | **yes** (6 new fields) |
| `cbm/hbi_exceedance.py`  | `02b3c63` | 3 965 | 3 965 | no |
| `cbm/hbi_logging.py`     | `cde079d` | 2 653 | 2 653 | no |
| `cbm/hbi_types.py`       | `516c49b` | 2 763 | 2 882 | yes (1 new field) |
| `cbm/hbi_updates.py`     | `96885ee` | 10 834 | 36 436 | **yes, substantially** |
| `cbm/individual_fit.py`  | `ab4d35f` | 10 950 | 11 063 | yes (1 change) |
| `cbm/map_estimation.py`  | `afc73c4` | 3 584 | 3 699 | yes (1 change) |
| `cbm/model_selection.py` | `b929ee4` | 8 771 | 8 771 | no |
| `cbm/optimization.py`    | `de69af5` | 19 421 | 19 702 | yes (1 change) |

New files not present upstream:

| file | purpose |
|---|---|
| `cbm/posterior_correlation.py` | credible intervals for group-level correlations (Wishart sampling) |
| `cbm/test_joint_elbo.py` | regression tests for the blockwise ELBO (`python -m cbm.test_joint_elbo`) |
| `examples/example_joint_covariance.py` | end-to-end ground-truth recovery demo, ~6 min, no data files needed |
| `examples/example_joint_stimuli.py` | the real trial sequences the demo simulates over — **stimulus structure only**: reward walks and risky-choice magnitudes, with no choices, no reaction times and no identifiers |
| `examples/mc_recenter_a0.py` | the paired Monte Carlo behind the `recenter_a0` measurement in §3 |

The four files with identical byte counts (`hbi_bound.py`,
`hbi_exceedance.py`, `hbi_logging.py`, `model_selection.py`) were additionally
read line-by-line against upstream; they are untouched.

---

## 1. What the extension does

Upstream, the group-level variational posterior factorises over parameters:

```
q(mu, tau) = prod_d  NormalGamma(mu_d, tau_d)
```

so the group covariance is diagonal by construction, and the model asserts that
any two parameters are independent across subjects. That is exactly what you
want when the D parameters belong to one task and you care about group means —
but it makes cross-task questions unaskable. If a subject's inverse temperature
in task A and inverse temperature in task B are fit in one concatenated model,
upstream HBI will estimate both group means correctly and will still hold their
group-level correlation at zero by construction.

This fork adds an opt-in switch, `HBIConfig.covariance_blocks`, that replaces
the Normal-Gamma factor over a designated set of parameters with a
**Normal-Wishart** factor over that set:

```
q(mu, Lambda) = prod_b  NormalWishart(mu_b, Lambda_b)      b = covariance blocks
```

Parameters not named in `covariance_blocks` keep singleton blocks, and a
singleton Normal-Wishart block is algebraically identical to the upstream
Normal-Gamma factor — so with `covariance_blocks=None` (the default) the joint
code path is never entered at all, and with an identity mask it is entered but
reproduces upstream exactly (test 1 in `test_joint_elbo.py`).

What this makes possible:

* estimating the group-level correlation between any parameter pair, within or
  across tasks, as a hyperparameter with its own posterior;
* de-attenuation — because the group covariance is built from between-subject
  scatter **plus** mean within-subject Laplace covariance (law of total
  covariance), the estimate is corrected for measurement error in the
  individual estimates, unlike a Pearson correlation of MAP estimates;
* credible intervals on those correlations, by sampling the fitted Wishart
  (`cbm/posterior_correlation.py`);
* all of the above while leaving responsibilities, model frequencies and
  exceedance probabilities intact.

---

## 2. The maths that changed

### 2.1 Sufficient statistics (`hbi_sumstats`)

Upstream computes, per parameter d,

```
Sdiag_d = sum_n r_n (theta_nd^2 + [A_n^-1]_dd) / N_k  -  thetabar_d^2
```

The joint branch computes the matrix generalisation

```
S = ( sum_n r_n theta_n theta_n^T  +  sum_n r_n A_n^-1 ) / N_k  -  thetabar thetabar^T
```

then multiplies elementwise by the binary mask. `diag(S)` is identical to
upstream's `Sdiag`, so the diagonal is untouched; the mask decides which
off-diagonal entries survive.

The second term is the law of total covariance: `theta_n` is a distribution,
not a point, so the group-level second moment is
`E[theta_n theta_n^T] = theta_n theta_n^T + A_n^-1`. It matters because the
E-step means are shrunk toward the group mean and therefore under-disperse;
adding each subject's posterior spread back is what makes the fixed point an
estimate of the true group covariance rather than of the shrunken scatter. The
consequence is that the reported correlation is de-attenuated relative to a
Pearson correlation of the independent, pre-HBI MAP estimates, whose variances
are inflated by estimation error while their covariance is not.

This needs the **full** per-subject inverse Hessian, which upstream does not
retain — hence the new `IndividualPosterior.hessian_inv` field (§4.3).

### 2.2 M-step (`hbi_qmutau`)

For each connected block `b` of size `D_b`:

```
beta_k  = beta0 + N_k                                   (unchanged)
a_k     = (beta0 a0 + N_k thetabar) / beta_k            (unchanged)
nu_k    = nu0 + N_k/2                                   (unchanged)
Sigma_b = sigma0_b + 0.5 ( N_k S_b + (N_k beta0/beta_k) (thetabar_b - a0_b)(thetabar_b - a0_b)^T )
```

`Sigma_b` is the matrix version of upstream's `sigma_k`; setting `D_b = 1`
recovers upstream's scalar line for line.

The variational Wishart is parameterised as

```
q(Lambda_b) = Wishart( dof = 2 nu_k,  scale = (2 Sigma_b)^-1 )
```

which gives `E[Lambda_b] = 2 nu_k (2 Sigma_b)^-1 = nu_k Sigma_b^-1`, matching
upstream's `Etau = nu_k / sigma_k` at `D_b = 1`. Under that parameterisation the
two Wishart quantities the ELBO needs are

```
E[ log |Lambda_b| ] = sum_{i=1..D_b} psi( nu_k + (1-i)/2 )  -  log |Sigma_b|
log-normaliser      = nu_k log |Sigma_b|  -  log Gamma_{D_b}( nu_k )
```

Both reduce exactly to `psi(nu_k) - log sigma_k` and
`-gammaln(nu_k) + nu_k log sigma_k` at `D_b = 1`. The `D_b log 2` terms of the
standard Wishart formulas cancel against the factor 2 in the scale — an earlier
version of this fork carried a spurious `+ D log 2` and used full-`D` rather
than per-block digamma/multigamma terms, which biased `L` and, with `K > 1`, the
responsibilities of masked models. That is fixed; `test_joint_elbo.py` test 2
is the regression test.

The four ELBO terms, with trace terms summed over the whole (block-diagonal)
matrix:

```
E[log p(mu|Lambda)] = (D/2) log(beta0/2pi) + 0.5 E[log|Lambda|]
                      - (beta0/2)( D/beta_k + (a_k-a0)^T E[Lambda] (a_k-a0) )
E[log q(mu|Lambda)] = (D/2) log(beta_k/2pi) + 0.5 E[log|Lambda|] - D/2
E[log p(Lambda)]    = sum_b [ (2 nu0 - D_b - 1)/2 * E[log|Lambda_b|] + logG0_b ]
                      - sum( sigma0 .* E[Lambda] )
E[log q(Lambda)]    = sum_b [ (2 nu_k - D_b - 1)/2 * E[log|Lambda_b|] + logG_b ]
                      - nu_k D
```

All four reduce to upstream's expressions for an all-singleton mask.

### 2.3 E-step (`hbi_qHZ`, `hbi_qhquad`)

* `hbi_qHZ`: the responsibility shift needs `log |E[Lambda]|`, which is
  `sum(log Etau)` for a diagonal `Etau` and `slogdet(Etau)` for a matrix. The
  `E[log p(H|mu,Lambda)]` term becomes
  `-0.5 N_k * sum( E[Lambda] .* ( S + (thetabar-a)(thetabar-a)^T ) )` — the
  trace written as an elementwise sum, identical to upstream for a diagonal
  `Etau`.
* `hbi_qhquad`: the per-subject Laplace prior becomes `N(a_k, E[Lambda]^-1)`
  with a **full** precision matrix rather than `diagflat(Etau)`. This is where
  the cross-task coupling reaches the individual level: a subject's task-A
  parameters now inform their task-B parameters through the group prior.

---

## 3. Configuration API added

All six new fields default to values that leave upstream behaviour unchanged.

```python
HBIConfig(
    covariance_blocks = None,   # List[List[Tuple[int,int]]] | None -- one entry per model k
    auto_nu0          = False,  # bool
    nu0_margin        = 1.0,    # float
    recenter_a0       = False,  # bool
    keep_best_iterate = False,  # bool
    divergence_tol    = None,   # float | None
)
```

**`covariance_blocks`** — for model `k`, a list of `(i, j)` index pairs into
that model's joint parameter vector. `hbi_run` turns each into a `D_k x D_k`
binary mask, `eye(D_k)` plus symmetric ones at the named pairs. Connected
components of that mask are the Wishart blocks. Example, for a 7-parameter RL
model concatenated with a 4-parameter risk model, linking only the two inverse
temperatures (indices 6 and 7):

```python
HBIConfig(covariance_blocks=[[(6, 7)]])
```

Full within- and cross-task covariance is
`[[(i, j) for i in range(11) for j in range(i+1, 11)]]`, which collapses to a
single 11-dimensional Wishart block.

**`auto_nu0` / `nu0_margin`** — the hyperprior dof is hard-coded upstream as
`v = 0.5`, i.e. Wishart dof `2v = 1`. A Wishart prior is proper only for
`dof > D_b - 1`, so `v = 0.5` sits exactly on the boundary at `D_b = 2` and is
improper beyond it. With `auto_nu0=True`, `nu0` is raised (never lowered) to
`max(v, (D_b_max + nu0_margin)/2)`, where `D_b_max` is the largest block for
that model; `nu0_margin = 1` gives `dof0 = D_b_max + 1`, the standard "just
proper" choice. `nu0` stays a scalar, so nothing downstream that reads
`qmutau[k].nu` as a float is affected. **This matters for `L` and for exceedance
probabilities, not for the fitted correlations** — see §7.1.

**`recenter_a0`** — replaces each model's hyperprior mean `a0` with the mean of
that model's individual-fit MAP estimates (finite subjects only) before the loop
starts. Rationale: the scale update contains
`(thetabar - a0)(thetabar - a0)^T`, whose off-diagonals are
`(tb_i - a0_i)(tb_j - a0_j)`. If the true group mean sits away from `a0` in the
same direction for two linked parameters — common on real data, where `a0` comes
from generic individual-fit priors — this injects positive covariance into
exactly the linked pairs. The magnitude is `~ beta0 * diff_i * diff_j` against
`N_k * S_ij`, so it is negligible at large `N` and material at small `N`.
Recentring is standard empirical Bayes and changes only the hyperprior mean, not
any update equation.

A worked example, with CBM's own hyperprior (`beta0 = 1`, `sigma0 = 0.01 I`),
four subjects and two linked parameters. Suppose the E-step leaves the
sufficient statistics at:

```
thetabar = (1, 1)      S = [[1.00, 0.25],
                            [0.25, 1.00]]      -> r = 0.25
```

With the default `a0 = (0, 0)`, the three pieces of the scale update are:

```
0.5 * N_k * S                            = [[2.00, 0.50], [0.50, 2.00]]
0.5 * (N_k beta0/beta_k) * (1,1)(1,1)^T  = [[0.40, 0.40], [0.40, 0.40]]
sigma0                                   = [[0.01, 0.00], [0.00, 0.01]]
                                           ---------------------------
Sigma_b                                  = [[2.41, 0.90], [0.90, 2.41]]   -> r = 0.373
```

With `recenter_a0=True`, `a0 = thetabar`, the middle term vanishes, and:

```
Sigma_b = [[2.01, 0.50], [0.50, 2.01]]                                     -> r = 0.249
```

The data said 0.25; with the default `a0` the fit reports 0.373. The update
itself is the correct conjugate one — the whole difference is the off-diagonal
of `(thetabar - a0)(thetabar - a0)^T`, which is 1 x 1 = 1 here only because both
group means happen to sit above the generic prior mean of zero. That is a prior
belief nobody intended to express. Its weight relative to the data term is
`diff_i diff_j / ((1 + N_k) S_ij)`, so it fades as 1/N — but it is largest
exactly where a spurious correlation does most damage: when the group mean is
far from `a0` and the true covariance is small.

This is measured, not just argued. Twenty **paired** replications — one
simulated dataset per replication, fit twice with the flag off and on, so
between-replication variance drops out of the comparison — at `N = 80` with a
true correlation of 0.5:

```
                                bias vs realised r        t     RMSE
covariance_blocks only                    +0.069       4.28    0.098
covariance_blocks + recenter_a0           -0.007      -0.40    0.079
```

Without recentring the estimator is biased upward and 17 of 20 replications
overshoot (p = 0.0004); with it the bias is indistinguishable from zero and the
sign test is 9 of 20. The paired shift is +0.076 with `t = 14`, i.e. the flag
does the same thing on every dataset — as a rank-one artefact must.
`mc_recenter_a0.py` reproduces this.

Scope matters here: the bias affects the **joint branch only**. In the default
diagonal branch the same term inflates the group variance, which is standard
conjugate behaviour and arguably intended, but there is no off-diagonal for it
to corrupt. It is a hazard this extension introduces by meeting an existing
default, not a defect in the original.

Because leaving it off is the wrong choice in almost every joint fit, `hbi_run`
now warns when `covariance_blocks` links at least one pair and `recenter_a0` is
`False`. We chose a warning over flipping the default so that existing scripts
keep their behaviour; see §7.8 for the question of whether that is the right
call.

**`keep_best_iterate` / `divergence_tol`** — see §5.3.

---

## 4. File-by-file diff

### 4.1 `cbm/hbi_updates.py` (10.8 kB → 36.4 kB)

**New imports.** `multigammaln` added to the `scipy.special` import.

**New module-level helpers (all four are ours; none exist upstream):**

| function | what it does |
|---|---|
| `safe_invert(mat)` | symmetrise, then LU-invert with escalating jitter (1e-8, ×100, 10 tries) and an explicit `isfinite` check on the *result*; diagonal fallback. Written to avoid `np.linalg.pinv`, whose SVD path returned silent NaN on some Apple-silicon BLAS builds (§5.4). |
| `make_positive_definite(mat, min_eig=1e-6)` | eigenvalue **flooring**. Needed because zeroing entries of a PSD matrix (i.e. applying the mask) does not preserve positive-definiteness; the resulting singularity does not show up as NaN, it shows up two steps later as `slogdet → -inf`, which then poisons `rarg = logrho - logrho[k]` via `-inf - (-inf) = NaN` without tripping any `isfinite` guard at the point of creation. |
| `_cap_eigenvalues(mat, max_eig)` | eigenvalue **ceiling**, applied to each subject's inverse Hessian before it enters the joint sufficient statistic (`AINV_MAX_EIG = 1e4`). Bounds how much one weakly-identified subject can dominate a large merged block. |
| `_mask_blocks(mask)` | connected-component decomposition of the binary mask, returning a list of index arrays. |

**`hbi_sumstats(r, qh)` → `hbi_sumstats(r, qh, covariance_mask=None)`**

* new `Nk < 1e-8` guard: a model that has collapsed to zero responsibility gets
  zero statistics instead of `0/0`;
* `keepdims=True` removed from `thetabar_k` and `Sdiag_k`, so both are now 1-D
  of length `D` rather than `(D, 1)`. See §7.5 — this is a package-wide
  convention change, not a local one;
* joint branch as in §2.1, with `theta` sanitised (`nan → 0`) and clipped to
  `±1e4` before the outer product;
* return name changed `Sdiag → S_out`; entries are 1-D vectors in default mode
  and `D x D` matrices in joint mode.

**`hbi_qmutau(pmutau, Nbar, thetabar, Sdiag)` → `hbi_qmutau(pmutau, Nbar, thetabar, S_in, covariance_mask=None, auto_nu0=False, nu0_margin=1.0)`**

* the five bound accumulators are initialised to `np.zeros(K)` instead of
  `np.full(K, np.nan)`, so a collapsed model contributes 0 rather than NaN;
* `Nk < 1e-8` short-circuit: reset to prior, contribute nothing to the bound;
* `auto_nu0` block (§3);
* joint branch: mask re-applied after the dense `diff @ diff.T`, then blockwise
  PD repair, then blockwise inversion, then the blockwise ELBO terms of §2.2;
* the default branch carries upstream's arithmetic unchanged, moved under an
  `if covariance_mask is None or covariance_mask[k] is None:` guard;
* `ElogpH[k]` is now also computed in the joint branch (it was previously left
  at zero there, which distorted the intermediate bound reported right after the
  `qmutau` step). Upstream computes it unconditionally; no change for non-joint
  runs.

**`hbi_qm`** — unchanged.

**`hbi_qHZ(..., thetabar, Sdiag)` → `hbi_qHZ(..., thetabar, S_in, covariance_mask=None)`**

* `logdetET` computed via `slogdet(Etau)` in joint mode, `sum(log(Etau))`
  otherwise, with a diagonal fallback if the sign is non-positive;
* a `nan_to_num(logrho, nan=-1e10, posinf=1e10, neginf=-1e10)` net before the
  responsibility softmax;
* `rarg` clipped to `±500` before `exp` (upstream: unclipped);
* `rlogr` rewritten from `r*log1p(r-1)` with a post-hoc `rlogr[r<eps]=0` to
  `np.where(r > eps, r*log(r), 0.0)` — algebraically identical, but it does not
  evaluate `log(0)` first;
* joint branch for `ElogpH[k]` as in §2.3.

**`hbi_qhquad`** — signature unchanged; body changed in five places:

* prior precision is `Etau` itself when 2-D, `diagflat(Etau)` when 1-D, with an
  identity fallback if non-finite;
* failure handling extended from upstream's `if flag_kn == 0` to
  `flag_kn == 0 or is_invalid or is_runaway`, where `is_invalid` is any
  non-finite `theta`/`A`/`logf` and `is_runaway` is `max|theta| > 1000`;
* every returned Hessian is symmetrised and given `+1e-8 * I` jitter;
* `cholesky` + `2*sum(log(diag))` replaced by `slogdet`, with `0.0` on
  non-finite — upstream **raises** `LinAlgError` here if a subject's Hessian is
  not positive definite, which aborts the whole fit;
* `np.linalg.inv` replaced by `safe_invert`, and the full inverse is stored into
  a new `Ainv_full_k` array of shape `(D, D, N)`.

**`hbi_bound`** — unchanged. (It is defined identically in both
`hbi_updates.py` and `hbi_bound.py` upstream; `hbi.py` imports the
`hbi_updates` copy. We left the duplication alone.)

### 4.2 `cbm/hbi.py` (20.6 kB → 33.9 kB)

**`_hbi_prog`** — `Sdiag_vec` now takes `np.diag(sd)` when `sd` is 2-D, and the
normalisation is `sqrt(max(|Sdiag|, 1e-10))` instead of `sqrt(Sdiag)`. See §7.4.

**`hbi_main`** —

* `initialize_r` is now read from `config.initialize` instead of being
  hard-coded to `'all_r_1'`. Upstream accepts the field, validates it, and then
  ignores it;
* optional `recenter_a0` step between `hbi_init` and `hbi_run`;
* the hyperparameters `b = 1.0, v = 0.5, s = 0.01` are unchanged and still
  hard-coded.

**`hbi_run`** —

* reads `config.tolL`, `config.auto_nu0`, `config.nu0_margin`,
  `config.keep_best_iterate`, `config.divergence_tol`;
* builds `covariance_mask` from `config.covariance_blocks` and threads it
  through `hbi_sumstats`, `hbi_qmutau`, `hbi_qHZ`;
* **new convergence criterion**: `abs(dL) < tolL` also terminates, from
  iteration 3 onwards. Upstream terminates on `dx < tolx` only, and never reads
  `tolL` despite carrying it in the config. See §7.4;
* divergence guard: if `divergence_tol` is set and `dL < -divergence_tol` (or
  `dL` is non-finite), stop and log;
* `keep_best_iterate` restore, §5.3;
* `he_list` takes `diag(sigma)` when `sigma` is 2-D;
* `profile.optimconfigs` stored as plain dicts rather than `Config` objects, to
  avoid pickle class-identity failures across sessions.

**`hbi_init`** —

* `opt_config` is coerced back from `dict` to `Config` (the counterpart of the
  `individual_fit` change in §4.4), so old and new pickles both load;
* `a0_k = np.asarray(cbm_map.profile.prior_mean).ravel()` — upstream keeps the
  `(D, 1)` column-vector shape that `individual_fit.Prior` imposes. §7.5;
* **per-subject scrubber**: any subject whose `theta`, `hessian`,
  `hessian_inv_diag` or `loglik` is non-finite, or whose `max|theta| > 1000`,
  has its parameters replaced by `a0`, its Hessian by `0.1 * I`, its
  `hessian_inv_diag` by `10`, its `loglik` and `lme` by `-1e6` and its
  `log_det_hessian` by `0`. This mutates the loaded `cbm_map` object in place;
* computes `pinv(hessian_n)` for every subject and stores the result in the new
  `hessian_inv` field;
* new `initialize_r == 'lme_softmax'` branch: responsibilities are seeded with a
  softmax over models of each subject's individual-fit log model evidence. This
  exists because with `'all_r_1'` every subject starts fully assigned to every
  model, so when the `K` models share one likelihood and differ only in their
  priors — a population-mixture setup, which is how we test for latent subgroups
  — the first M-step hands all components the same responsibility-weighted mean
  and they merge immediately and never separate.

**`hbi_null`** — unchanged, including the docstring/implementation mismatch
noted in §7.6.

### 4.3 `cbm/hbi_types.py`

One field added:

```python
@dataclass
class IndividualPosterior:
    loglik: np.ndarray
    parameters: List[np.ndarray]
    hessian_inv_diag: List[np.ndarray]
    log_det_hessian: np.ndarray
    hessian_inv: Optional[List[np.ndarray]] = None   # NEW: list of (D, D, N)
```

Defaulted to `None`, so old pickles unpickle and non-joint code paths never
touch it.

### 4.4 `cbm/individual_fit.py`

One change: `FitProfile.config` is typed `Any` and populated with
`config.__dict__.copy()` rather than the `Config` instance, so that a `.pkl`
written in one session loads in another without the `Config` class identity
having to match. `hbi_init` converts it back (§4.2).

### 4.5 `cbm/map_estimation.py`

One change, in `log_posterior`:

```python
# upstream
log_det_precision = np.log(np.linalg.det(prior_precision))
# here
log_det_precision = np.linalg.slogdet(prior_precision)[1]
```

`det` of a `D = 11` precision matrix underflows to 0 for plausible parameter
scales, making `log(det)` `-inf` and the whole log posterior `-inf`. `slogdet`
is exact in log space. Mathematically identical wherever `det` does not
underflow.

### 4.6 `cbm/optimization.py`

One change, in `BFGSOptimizer.optimize`:

```python
# upstream
if result.f < best_f:
# here
if best_result is None or (np.isfinite(result.f) and result.f < best_f):
```

`NaN < inf` is `False`, so upstream leaves `best_result = None` if every restart
returns NaN, and then raises `AttributeError` on `best_result.x` in the Hessian
call rather than returning `flag = 0`. Behaviourally identical whenever any
restart produces a finite objective.

### 4.7 `cbm/__init__.py`

Three new exports from `posterior_correlation`. Existing exports unchanged.

### 4.8 `cbm/posterior_correlation.py` (new)

```python
correlation_posterior_samples(result, i, j, k=0, n_samples=20000, rng=None) -> np.ndarray
correlation_credible_interval(result, i, j, k=0, ci=0.95, ...) -> dict
all_linked_correlation_cis(result, k=0, ci=0.95, ...) -> dict[(i,j) -> dict]
```

Recovers the block structure from the exact-zero pattern of `sigma`, samples
`Lambda_b ~ Wishart(2 nu_k, (2 Sigma_b)^-1)`, inverts, and reports
`r = Sigma_ij / sqrt(Sigma_ii Sigma_jj)` with quantiles and `P(r > 0)`.
Requesting an unlinked pair raises, since the model's posterior for that
correlation is a structural zero rather than a distribution.

---

## 5. Robustness changes, and why they were needed

None of these were part of the design; each was added after a specific failure
on real data (`N` ≈ 200–500, `D = 11`, two concatenated tasks).

### 5.1 Masking breaks positive-definiteness

Elementwise-masking a PSD matrix can produce a singular or indefinite one.
`slogdet` then returns `-inf`, which survives every `isfinite` check until it
reaches `rarg = logrho - logrho[k]`, where `-inf - (-inf) = NaN` silently
corrupts the entire responsibility matrix from that iteration onwards. Fixed by
blockwise eigenvalue flooring (`make_positive_definite`) applied to `Sigma_b`
after masking. Flooring is done per block so that off-block entries stay exactly
zero — full-matrix flooring would silently re-introduce the couplings the mask
was there to remove.

### 5.2 One bad subject can break a large block

A subject whose likelihood is flat along one weakly-identified parameter gets a
large-but-finite inverse Hessian. In a 2-parameter block that only distorts its
own pair; in an 11-parameter merged block it distorts everything, because the
block's eigendecomposition mixes all coordinates. Fixed by
`_cap_eigenvalues(A_n^-1, 1e4)`, symmetric with the pre-existing `±1e4` clip on
`theta`.

### 5.3 ELBO limit cycles

With a large merged block at moderate `N` we saw a characteristic failure: the
run converges normally for ~17 iterations, then one subject's Laplace refit
fails to find a positive-definite Hessian, and from that point `L` swings by
`±1e6` every iteration while `dx` stays tiny, so `dx < tolx` never fires and the
run burns through `maxiter` in a cycle.

`divergence_tol` stops the run at the first `dL < -tol`. `keep_best_iterate`
returns the state from the best *trustworthy* iterate.

One detail worth stating: **we do not take a global `argmax(L)`.** The first
corrupted iteration frequently sends `L` sharply *up*, so a global argmax
happily selects a corrupted state — we found this in testing before it reached
any analysis. Instead the code scans forward from iteration 1 and cuts at the
first iteration that either drops materially or produces a `|dL|` more than
100× the largest `|dL|` seen so far (in a healthy EM run `|dL|` decreases
monotonically toward zero), then takes the argmax of that prefix. The `100×`
factor and the drop tolerance are heuristics — §7.3.

### 5.4 Platform-specific LAPACK failures

`np.linalg.pinv` (SVD) returned silent NaN, not an exception, on some
Apple-silicon BLAS builds for matrices that `np.linalg.inv` (LU) handled fine.
`safe_invert` and the retry loops in `make_positive_definite` /
`_cap_eigenvalues` therefore check `isfinite` on the *result*, not just catch
`LinAlgError`. If this is not a problem you have seen, these loops are inert
overhead and could reasonably be reduced to a single call.

---

## 6. Validation performed

1. **`test_joint_elbo.py` test 1** — with an identity mask, the joint branch
   must reproduce the default Gaussian-Gamma branch exactly: same posterior
   (`a`, `beta`, `nu`, `diag(Sigma)`, `E[Lambda]`) *and* all five ELBO terms. It
   does.
2. **`test_joint_elbo.py` test 2** — with a linked pair and a proper prior
   (`nu0 = 1.0`), every Wishart ELBO term is checked against Monte-Carlo
   expectations under `q(Lambda) = Wishart(2 nu, (2 Sigma_b)^-1)`. This is a
   check of internal consistency between the closed forms and the assumed `q`,
   not a check that the factorisation is KL-optimal — §7.2.
3. **`test_joint_elbo.py` test 3** — mask-to-block decomposition.
4. **Simulation recovery.** Drawing subject parameters from a known MVN,
   simulating choices, refitting: the fitted group-level `rho` is unbiased at
   moderate-to-large `N` (2000 replications, bias `+0.001`). A positive bias
   appears only when small `N` *and* low per-subject reliability coincide
   (`+0.113` at `N = 50` with reliability 0.6; `~0` at `N = 500`), consistent
   with the ordinary finite-sample bias of a correlation estimated from noisy
   scores rather than with anything specific to the algorithm.
5. **Cross-engine agreement.** On our real data (`N` = 200–500) this fork, Stan
   (NUTS) and JAGS (Gibbs) return the same group-level correlations for the same
   joint model, within each other's credible intervals. The variational
   intervals are narrower, as expected for a mean-field approximation.
6. The correlation of MAP estimates sits *below* the fitted group correlation by
   roughly the amount the Spearman attenuation formula predicts given the
   per-subject posterior variances — i.e. the de-attenuation mechanism in §2.1
   is doing what it should rather than inflating.

---

## 7. Uncertainties, approximations and open questions

These are the points we would most like a second opinion on.

### 7.1 The prior Wishart normaliser is dropped for `D_b >= 2` when `auto_nu0=False`

With the hard-coded `v = 0.5`, the implied block-Wishart prior has `dof0 = 1`,
and `multigammaln(0.5, D_b)` hits a pole for `D_b >= 2`. The code catches the
`ValueError` and sets `logG0_b = 0`.

`logG0_b` is a constant of the model, so this does not affect the fitted
`Sigma`, the correlations, or the responsibilities *within* a fit. It **does**
mean that `L` for a joint model is missing a model-dependent constant, so `L`,
BOR and protected exceedance probabilities are **not** comparable between a
joint and a non-joint model unless `auto_nu0=True` (or `v` is raised). We say so
in the code comment, but a user who compares `L` across joint and non-joint fits
without reading it will get a wrong answer. **Is silently dropping the constant
the right call, or should this raise?**

### 7.2 The blockwise factorisation for chained masks is imposed, not derived

If the user links `(0,1)` and `(1,2)`, the connected component is `{0,1,2}` and
we fit a 3×3 Wishart, but `Sigma[0,2]` stays a structural zero because the mask
is re-applied after the dense outer product. That zero is *not* a conjugate
constraint — the exactly-conjugate object for a 3-dimensional block is a full
3×3 Wishart with no zeros. So for chained masks the fitted `q` is a projection
rather than the exact variational optimum, and the reported `L` is a bound under
a slightly different family than the one the update equations solve.

Two sub-points we are unsure about:

* the constraint we impose is `Sigma_ij = 0`, i.e. zero **marginal** covariance,
  whereas the natural sparse-Gaussian object would put the zero in the
  **precision** (zero partial correlation). We chose the covariance version
  because it is what masking the sufficient statistic gives for free and because
  it is the quantity a user wants to read off. Is that defensible?
* `make_positive_definite` reconstructs the block as `V diag(clip(eig)) V^T`,
  which preserves the structural zero **only if no eigenvalue was actually
  clipped**. If clipping fires, the zero can be filled in. In practice clipping
  fires rarely, but the constraint is therefore not strictly enforced.

Neither issue arises for the two cases we actually use: disjoint pairs (exactly
conjugate 2×2 Wisharts) and `"full"` (one `D`-dimensional Wishart with no
structural zeros, i.e. the textbook multivariate HBI).

### 7.3 Magic constants

None of these are user-configurable, and all were chosen empirically:

| constant | value | where |
|---|---|---|
| `theta` clip | `±1e4` | `hbi_sumstats` |
| `AINV_MAX_EIG` | `1e4` | `hbi_sumstats` |
| runaway threshold | `\|theta\| > 1000` | `hbi_qhquad`, `hbi_init` |
| Hessian jitter | `1e-8 * I` | `hbi_qhquad` |
| `min_eig` | `1e-6` | `make_positive_definite` |
| `rarg` clip | `±500` | `hbi_qHZ` |
| non-finite `logrho` replacement | `-1e10` | `hbi_qHZ` |
| failed-subject `loglik` | `-1e6` | `hbi_qhquad`, `hbi_init` |
| `_BREAK_FACTOR` | `100` | `keep_best_iterate` |

They should probably be config fields. The `-1e6` loglik in particular is a
hard-coded number on the same scale as real log-likelihoods for short
experiments, which is a latent bug for anyone with few trials per subject.

### 7.4 Two behavioural changes that are **not** opt-in

We would flag these especially, because they alter standard, non-joint runs:

1. **`abs(dL) < tolL` now terminates the loop** (from iteration 3). Upstream
   carries `tolL = -log(0.5) ≈ 0.693` in the config and never uses it. Adding
   the criterion typically shortens runs; it can also stop a run that upstream
   would have continued. We believe this is what `tolL` was meant for, but it
   *is* a change in default behaviour.
2. **`dx` is computed as `thetabar / sqrt(max(|S_dd|, 1e-10))`** instead of
   `thetabar / sqrt(S_dd)`. The `abs` and the floor were needed because a
   collapsed model can produce `S_dd <= 0`; the consequence is that `dx`, and
   therefore the `dx < tolx` stopping rule, differ marginally from upstream even
   for well-behaved runs.

In addition, the Hessian jitter (`+1e-8 I`), `safe_invert`'s jitter and the
`hbi_init` / `hbi_qhquad` scrubbers all run unconditionally, so
`hessian_inv_diag` and `log_det_hessian` differ from upstream in the last few
digits even with `covariance_blocks=None`. We have not seen this change any
result, but it does mean this fork is not bit-identical to upstream on a
non-joint fit. **If you would prefer all of these gated behind a flag, that is
an easy change and we are happy to make it.**

### 7.5 Array-shape convention

Upstream carries `a0`, `thetabar` and `a_k` as `(D, 1)` column vectors, because
`individual_fit.Prior.__post_init__` reshapes `prior_mean` and `hbi_init`
propagates it. This fork flattens to `(D,)` throughout (`.ravel()` in
`hbi_init`, `keepdims=True` removed in `hbi_sumstats`), because the matrix
algebra in the joint branch needs explicit `reshape(-1, 1)` at the few points
where a column is wanted, and mixing the two conventions produced silent
`(D, D)` broadcasts.

**Consequence:** `cbm.output.group_mean[k]` is now a length-`D` 1-D array rather
than `(D, 1)`. Any downstream code that indexes it as `[:, 0]` will break. This
is the one change that is not backward-compatible at the API level.

### 7.6 Things we observed upstream but did not change

* `hbi_null`'s docstring documents a `(cbm, cbm0)` return, but the
  implementation returns `cbm` only. Unpacking as documented raises
  `TypeError`. We work around it in our calling code.
* `hbi_main` computes exceedance probabilities without `L`/`L0`, so
  `output.protected_exceedance_prob` is NaN unless `hbi_null` is run afterwards.
  We call `hbi_null` explicitly.
* `hbi_qhquad` sets `cfg.inits = qh.parameters[k][:, n]` per subject, but
  `optimize_map` calls `optimizer.optimize(objective, x_init=prior_mean)`.
  `BFGSOptimizer` does read `config.inits`, so the warm start is used — but as
  one initialisation among many rather than as *the* start.
* `hbi_bound` is defined identically in `hbi_bound.py` and `hbi_updates.py`.
* `hbi_config` validates `initialize in ("all_r_1", "cluster_r")`, but
  `hbi_init` raises `NotImplementedError` for `'cluster_r'`. We kept
  `'cluster_r'` accepted-but-unimplemented and added `'lme_softmax'`.

### 7.7 Not yet established

* Whether the mean-field factorisation between `q(mu, Lambda)` and `q(H)`
  narrows the correlation's credible interval relative to MCMC by the standard
  `sqrt(1 - rho^2)` factor or by more. Our intervals are narrower than Stan's
  and JAGS's; we have not quantified whether the gap matches theory.
* Whether `auto_nu0` raising `nu0` to satisfy the *largest* block is too
  conservative for a model containing one large and several small blocks — the
  small blocks then carry a stronger prior than they need.
* Whether `recenter_a0` should also apply inside `hbi_null`. It currently does
  not (it is implemented in `hbi_main`, and `hbi_null` calls `hbi_init`
  directly), so a joint fit with `recenter_a0=True` is compared against a null
  fit using the original `a0`. We expect this to affect `L0` and therefore the
  BOR, and we have not measured how much. Now that `recenter_a0` is known to
  matter (§3) this is more pressing than when we wrote it.

### 7.8 Should `recenter_a0` default to True?

The measurement in §3 says that leaving it off biases the recovered correlation
by +0.069 (`t = 4.3`) while turning it on gives −0.007 (`t = −0.4`). On that
evidence the flag should arguably not be a flag at all for joint fits.

We did not flip the default, for two reasons. It would silently change the
numbers produced by any existing script that sets `covariance_blocks`, and
`recenter_a0` also affects non-joint fits (it changes the hyperprior mean for
every parameter, not only linked ones), where we have not measured anything. So
`hbi_run` warns instead.

Three ways this could go, and we do not have a strong view:

1. keep the warning, as now;
2. default `recenter_a0=True` whenever `covariance_blocks` links a pair, leaving
   non-joint fits untouched — narrow, but a config field whose default depends
   on another field is unusual;
3. leave it entirely to the user and document it only.

We would take your preference here.

---

## 8. Usage

```python
from cbm.individual_fit import individual_fit
from cbm.hbi import hbi_main, hbi_null
from cbm.hbi_config import HBIConfig
from cbm.posterior_correlation import all_linked_correlation_cis

# 1. individual fits of the concatenated (task A + task B) model, as usual
ind = individual_fit(data, joint_model, prior_mean, prior_var, fname="fit.pkl")

# 2. HBI with a group-level covariance between the two inverse temperatures
cfg = HBIConfig(
    covariance_blocks = [[(6, 7)]],   # model 0: link parameters 6 and 7
    auto_nu0          = True,         # proper Wishart prior -> L stays comparable
    recenter_a0       = True,
    keep_best_iterate = True,
    divergence_tol    = 50.0,
    save_prog         = False,
)
res = hbi_main(data, [joint_model], ["fit.pkl"], fname="hbi.pkl", config=cfg)

# 3. the fitted group correlation, with a credible interval
for (i, j), s in all_linked_correlation_cis(res, k=0).items():
    print(f"r({i},{j}) = {s['point']:+.3f}  "
          f"[{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]  P(r>0) = {s['prob_pos']:.3f}")
```

Setting `covariance_blocks=None` reverts to the upstream algorithm, modulo the
non-opt-in differences catalogued in §7.4.

---

## 9. Attribution

The joint-modelling extension was implemented by Ben J. Wagner, with substantial
use of LLM assistance for the derivations, the numerical hardening and the
documented changes (this document). The HBI algorithm and both the MATLAB and Python implementations are
Payam Piray's; the upstream code in every unmodified region is his verbatim.

Piray P., Dezfouli A., Heskes T., Frank M. J., Daw N. D. (2019). Hierarchical
Bayesian inference for concurrent model fitting and comparison for group
studies. *PLoS Computational Biology* 15(6): e1007043.
