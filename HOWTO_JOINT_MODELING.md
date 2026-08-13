# How to fit a joint (cross-task) model

A practical guide to `HBIConfig.covariance_blocks`. For what was changed in the
toolbox and why, see `README_JOINT_MODELING.md`. For the caveats you should know
before publishing a result, see its section 7.

---

## What this does

Standard HBI factorises the group-level posterior over parameters, so the group
covariance is diagonal and any two parameters are independent across subjects.
That is fine for one task. It makes cross-task questions unaskable: fit two
tasks as one concatenated model and HBI will get both group means right while
holding their correlation at zero by construction.

`covariance_blocks` names pairs of parameters that are allowed to covary. For
those, the Normal-Gamma factor is replaced by a Normal-Wishart one, and the
group-level correlation becomes a hyperparameter with its own posterior.

---

## Install

```bash
git clone -b feature-joint-modeling https://github.com/BenJonathanWagner/cbm_python.git
cd cbm_python
python3 -m venv .venv && source .venv/bin/activate
pip install numpy scipy      # NOT declared in pyproject.toml -- install them first
pip install -e .
```

`pip install -e .` alone will appear to succeed and then fail on `import cbm`,
because `pyproject.toml` declares no dependencies. Install numpy and scipy first.

Check it works:

```bash
python -m cbm.test_joint_elbo                  # ~10 s, prints "All tests passed."
python examples/example_joint_covariance.py    # ~6 min, recovers a known correlation
```

The first is the one that matters: it verifies that with an identity mask the
joint code path reproduces the standard branch exactly — same posterior, same
ELBO terms. If that passes, the extension is not silently changing your
non-joint results.

---

## A complete example you can run

Copy this into a file and run it. It needs no data and finishes in about two
seconds. Each "task" is deliberately trivial — a handful of noisy observations of
one parameter — so that nothing distracts from the wiring.

```python
import numpy as np
from cbm.individual_fit import individual_fit
from cbm.hbi import hbi_main
from cbm.hbi_config import HBIConfig
from cbm.posterior_correlation import all_linked_correlation_cis

# Two toy "tasks": each gives a few noisy observations of one parameter.
def model_A(p, d):  return float(-0.5 * np.sum((d - p[0]) ** 2))
def model_B(p, d):  return float(-0.5 * np.sum((d - p[0]) ** 2))

def joint_model(parameters, data):          # 2 params: [theta_A, theta_B]
    data_A, data_B = data
    return model_A(parameters[:1], data_A) + model_B(parameters[1:], data_B)

# Simulate 200 subjects whose two parameters correlate at 0.5
rng   = np.random.default_rng(0)
TRUE  = 0.5
Sigma = np.array([[1.0, TRUE], [TRUE, 1.0]])
theta = rng.multivariate_normal([0.0, 0.0], Sigma, size=200)
data  = [(rng.normal(t[0], 1.0, 5), rng.normal(t[1], 1.0, 5)) for t in theta]

ind = individual_fit(data, joint_model, np.zeros(2), np.full(2, 10.0),
                     fname="min_fit.pkl", config={"num_init": 2, "verbose": False})
res = hbi_main(data, [joint_model], ["min_fit.pkl"], fname="min_hbi.pkl",
               config=HBIConfig(covariance_blocks=[[(0, 1)]], recenter_a0=True,
                                auto_nu0=True, save_prog=False, verbose=0, flog=-1))

s  = all_linked_correlation_cis(res, k=0, rng=np.random.default_rng(1))[(0, 1)]
mp = ind.output.parameters
print(f"  true correlation imposed   {TRUE:+.3f}")
print(f"  realised in the draw       {np.corrcoef(theta[:,0], theta[:,1])[0,1]:+.3f}")
print(f"  Pearson r of MAP estimates {np.corrcoef(mp[:,0], mp[:,1])[0,1]:+.3f}  (attenuated)")
print(f"  HBI group correlation      {s['point']:+.3f}  "
      f"95% CI [{s['ci_low']:+.3f}, {s['ci_high']:+.3f}]")
```

Output:

```
  true correlation imposed   +0.500
  realised in the draw       +0.479
  Pearson r of MAP estimates +0.442  (attenuated)
  HBI group correlation      +0.454  95% CI [+0.339, +0.557]
```

The correlation of the point estimates (+0.442) sits below the correlation
actually present in the drawn parameters (+0.479); the group-level
hyperparameter (+0.454) recovers it, and its interval covers it. That gap is the
whole point of estimating the correlation hierarchically rather than correlating
MAP estimates.

For a realistic version — two real cognitive models, real trial sequences, nine
parameters — see `examples/example_joint_covariance.py`.

## The pattern for your own models

The example above is a sketch of the shape your own code needs. Three rules:

```python
# 1. ONE model function. Its parameter vector is task A's parameters followed by
#    task B's, and its log-likelihood is the sum of the two.
def joint_model(parameters, data):
    data_A, data_B = data
    return model_A(parameters[:5], data_A) + model_B(parameters[5:], data_B)

# 2. data[n] is a tuple: (task A's data for subject n, task B's data for subject n)

# 3. priors are concatenated the same way
prior_mean = np.concatenate([prior_mean_A, prior_mean_B])
prior_var  = np.concatenate([prior_var_A,  prior_var_B])
```

Only subjects with data from both tasks can be included — match them on ID before
building `data`.

---

## Specifying `covariance_blocks`

**Indices are positions in the concatenated parameter vector**, not within each
task. If task A has 5 parameters and task B has 4, then task B's first parameter
is index 5, not 0. Getting this wrong is the most common mistake, and it fails
silently — you get a correlation between two parameters, just not the ones you
meant.

**The outer list is over models**, one entry per model in `models`. With a
single model you always have a doubly-nested list:

```python
covariance_blocks = [[(4, 5)]]              # one model, one linked pair
covariance_blocks = [[(4, 5)], [(3, 4)]]    # two models, different pairs each
covariance_blocks = [[(4, 5)], None]        # link in model 0, leave model 1 diagonal
```

Common patterns, for a 5 + 4 parameter joint model:

```python
# just the two inverse temperatures, across tasks
[[(4, 5)]]

# several independent cross-task pairs
[[(4, 5), (2, 7)]]

# full covariance within task A only
[[(i, j) for i in range(5) for j in range(i + 1, 5)]]

# everything with everything (one 9-dimensional Wishart block)
[[(i, j) for i in range(9) for j in range(i + 1, 9)]]
```

Overlapping pairs merge. `[(0,1), (1,2)]` becomes one block over `{0,1,2}`, and
that case is an approximation rather than exact conjugacy — see
`README_JOINT_MODELING.md` §7.2. Disjoint pairs and "everything" are both exact.

> **Careful with merged blocks.** Once `{0,1,2}` is one block,
> `all_linked_correlation_cis` reports **three** pairs, including `(0,2)` — which
> you never linked. Its point estimate is exactly 0 because it is a structural
> zero, but it still comes with a credible interval that looks like a result:
>
> ```
> r(0, 1)  -0.0006   CI [-0.371, +0.361]
> r(0, 2)  -0.0000   CI [-0.372, +0.367]   <- constraint, not an estimate
> r(1, 2)  +0.2651   CI [-0.115, +0.576]
> ```
>
> Do not report `(0,2)` as evidence of no correlation. The model was told it is
> zero. If you want a genuine estimate for all three, link all three pairs.

---

## Settings you should not leave at their defaults

**`recenter_a0=True`** — the single most important one. The scale update
contains `(thetabar - a0)(thetabar - a0)^T`, which is rank one and therefore has
correlation exactly +1 whenever two linked parameters' group means sit on the
same side of `a0`. Since `a0` comes from your generic individual-fit prior, that
is the normal case, and it inflates precisely the correlations you are trying to
measure. Measured over 20 paired replications at N = 80: bias **+0.069**
(t = 4.3) with it off, **−0.007** (t = −0.4) with it on. `hbi_run` warns if you
forget. Do not ignore the warning.

**`auto_nu0=True`** if you will compare `L`, BOR or protected exceedance
probabilities between a joint and a non-joint model. The hard-coded hyperprior
dof makes the block-Wishart prior improper for blocks of two or more, and its
normalising constant is then dropped, so `L` is missing a model-dependent
constant. Correlations are unaffected either way.

**`keep_best_iterate=True` and `divergence_tol=50.0`** for large blocks. With
many parameters merged into one block, a single subject whose Laplace refit
loses positive-definiteness can send the bound into a limit cycle that never
triggers the convergence test. These stop the run at the first material drop and
return the last good iterate.

---

## Reading and sanity-checking the output

```python
res.output.group_mean[0]          # group means, 1-D array of length D
res.output.parameters[0]          # per-subject estimates, (N, D)
res.math.qmutau[0].sigma          # the fitted (D, D) inverse-scale matrix
```

The correlation is `sigma[i,j] / sqrt(sigma[i,i] * sigma[j,j])`, which is what
`all_linked_correlation_cis` reports as `point`. Any overall scaling cancels.

**Three checks worth making every time:**

1. **Fit it once with `covariance_blocks=None` and compare group means.** Adding
   a covariance block should not move them. In the bundled example the largest
   difference is 0.037 on a parameter with reliability 0.35. If your means move
   substantially, something else is wrong.

2. **Compute the reliability of each linked parameter.** The maximum correlation
   you could ever observe between two measures is `sqrt(rel_i * rel_j)`. A null
   result against a ceiling of 0.6 means something quite different from a null
   against 0.95. See the `reliability()` helper in
   `examples/example_joint_covariance.py`.

3. **Check the credible interval, not just the point estimate.** For weakly
   identified parameters it is wide, and a point estimate quoted without it is
   misleading. Note that variational intervals are narrower than MCMC's.

---

## Things that will bite you

**`output.group_mean[k]` is now a 1-D array**, not `(D, 1)`. Existing code
indexing it as `[:, 0]` breaks. This is the one backward-incompatible change.

**More subjects does not improve reliability.** It narrows the interval around
whatever correlation is estimable, but the ceiling `sqrt(rel_i * rel_j)` is set
by trials per subject and task design. Going from N = 200 to N = 500 cannot lift
a ceiling of 0.6; doubling trials can.

**A Pearson correlation of the MAP estimates is not comparable** to the HBI
correlation, and will be lower. Estimation error inflates each parameter's
variance but not their covariance, so the point-estimate correlation is
attenuated. The hyperparameter is not. Do not present them side by side without
saying which is which.

**Chained masks are approximate.** See §7.2 of the review document.

**Small N with low reliability biases correlations upward**, as it does for any
correlation estimated from noisy scores. At N = 50 with reliability 0.6 the bias
is about +0.11; by N = 500 it is negligible.

---

## Getting help / reporting problems

This is an extension to Payam Piray's `cbm_python`, not part of the official
toolbox. Problems with the joint-modelling code path belong with
Ben J. Wagner (ben.wagner@tuebingen.mpg.de), not upstream.

Piray P., Dezfouli A., Heskes T., Frank M. J., Daw N. D. (2019). Hierarchical
Bayesian inference for concurrent model fitting and comparison for group
studies. *PLoS Computational Biology* 15(6): e1007043.
