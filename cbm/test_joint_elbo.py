"""Regression tests for the blockwise joint-modeling ELBO in hbi_updates.py.

Run from the parent directory of the cbm package:  python -m cbm.test_joint_elbo

TEST 1  An identity (eye) covariance mask must reproduce the default
        Gaussian-Gamma branch exactly: same posterior AND same ELBO terms.
TEST 2  With a linked pair, every Wishart ELBO term must match Monte-Carlo
        expectations under q(Lambda) = Wishart(2*nu, (2*sigma_b)^{-1}) per
        block (uses a proper prior nu0=1.0 so E[ln p(Lambda)] is testable).
TEST 3  Mask-to-block decomposition sanity.
"""
import numpy as np
from scipy.special import gammaln, multigammaln
from scipy.stats import wishart, gamma as gamma_dist

from cbm.hbi_types import GaussianGammaDistribution
from cbm.hbi_updates import hbi_qmutau, _mask_blocks

rng = np.random.default_rng(3)

# ============================================================================
# TEST 1: eye-mask joint branch must reproduce the default 1-D branch exactly
# ============================================================================
D, N = 4, 37.0
a0 = rng.standard_normal(D)
pmutau = [GaussianGammaDistribution(a=a0, beta=1.0, sigma=0.01 * np.ones(D),
                                    nu=0.5, Etau=np.zeros(D), Elogtau=np.zeros(D), logG=0.0)]
Nbar = np.array([N])
tb = [rng.standard_normal(D)]
Q = rng.standard_normal((D, D))
S_full = Q @ Q.T / D + 0.5 * np.eye(D)

q_def, b_def = hbi_qmutau(pmutau, Nbar, tb, [np.diag(S_full).copy()], covariance_mask=None)
q_eye, b_eye = hbi_qmutau(pmutau, Nbar, tb, [S_full * np.eye(D)], covariance_mask=[np.eye(D)])

print("TEST 1: eye-mask joint == default 1-D branch")
checks = {
    "a": (q_def[0].a, q_eye[0].a),
    "beta": (q_def[0].beta, q_eye[0].beta),
    "nu": (q_def[0].nu, q_eye[0].nu),
    "sigma": (q_def[0].sigma, np.diag(q_eye[0].sigma)),
    "Etau": (q_def[0].Etau, np.diag(q_eye[0].Etau)),
    "sum(Elogtau)": (np.sum(q_def[0].Elogtau), np.sum(q_eye[0].Elogtau)),
    "logG": (q_def[0].logG, q_eye[0].logG),
    "Elogpmu": (b_def.Elogpmu, b_eye.Elogpmu),
    "Elogptau": (b_def.Elogptau, b_eye.Elogptau),
    "Elogqmu": (b_def.Elogqmu, b_eye.Elogqmu),
    "Elogqtau": (b_def.Elogqtau, b_eye.Elogqtau),
    "ElogpH": (b_def.ElogpH, b_eye.ElogpH),
}
ok = True
for name, (x, y) in checks.items():
    # rtol 1e-6: safe_invert adds a 1e-8 diagonal jitter, so the joint path
    # differs from exact division by ~1e-9 relative — numerically irrelevant
    match = np.allclose(np.asarray(x, dtype=float), np.asarray(y, dtype=float), rtol=1e-6, atol=1e-8)
    ok &= match
    print(f"  {name:<14}: {'OK' if match else 'MISMATCH  ' + str(x) + ' vs ' + str(y)}")
print("  -> TEST 1", "PASSED" if ok else "FAILED")

# ============================================================================
# TEST 2: pair-mask ELBO terms vs Monte-Carlo (proper prior nu0=1.0)
# ============================================================================
D = 3
nu0, s0 = 1.0, 0.01
a0 = rng.standard_normal(D)
pmutau = [GaussianGammaDistribution(a=a0, beta=1.0, sigma=s0 * np.ones(D),
                                    nu=nu0, Etau=np.zeros(D), Elogtau=np.zeros(D), logG=0.0)]
mask = np.eye(D); mask[0, 1] = mask[1, 0] = 1.0
tb = [rng.standard_normal(D)]
Q = rng.standard_normal((D, D))
S_full = Q @ Q.T / D + 0.4 * np.eye(D)

q_j, b_j = hbi_qmutau(pmutau, Nbar, tb, [S_full * mask], covariance_mask=[mask])
sig = q_j[0].sigma
nu_k = q_j[0].nu

pair = np.ix_([0, 1], [0, 1])
lam_pair = wishart.rvs(df=2 * nu_k, scale=np.linalg.inv(2 * sig[pair]), size=60_000, random_state=5)
tau_3 = gamma_dist.rvs(a=nu_k, scale=1.0 / sig[2, 2], size=60_000, random_state=6)

mc_Elogdet = np.mean([np.linalg.slogdet(l)[1] for l in lam_pair]) + np.mean(np.log(tau_3))
code_Elogdet = float(np.sum(q_j[0].Elogtau))
print("\nTEST 2: pair-mask ELBO terms vs MC (nu0=1.0, D=3, blocks {0,1},{2})")
print(f"  E[log|Lambda|]  MC {mc_Elogdet: .5f}   code {code_Elogdet: .5f}   diff {code_Elogdet-mc_Elogdet:+.5f}")

def logq(lam2, t3):
    _, ldW = np.linalg.slogdet(2 * sig[pair])
    _, ldL = np.linalg.slogdet(lam2)
    lq_pair = (0.5 * (2 * nu_k - 3) * ldL - 0.5 * np.trace(2 * sig[pair] @ lam2)
               + nu_k * ldW - nu_k * 2 * np.log(2) - multigammaln(nu_k, 2))
    lq_3 = nu_k * np.log(sig[2, 2]) - gammaln(nu_k) + (nu_k - 1) * np.log(t3) - sig[2, 2] * t3
    return lq_pair + lq_3

def logp(lam2, t3):
    s0_pair = s0 * np.eye(2)
    _, ldL = np.linalg.slogdet(lam2)
    lp_pair = (0.5 * (2 * nu0 - 3) * ldL - 0.5 * np.trace(2 * s0_pair @ lam2)
               + nu0 * np.linalg.slogdet(2 * s0_pair)[1] - nu0 * 2 * np.log(2) - multigammaln(nu0, 2))
    lp_3 = nu0 * np.log(s0) - gammaln(nu0) + (nu0 - 1) * np.log(t3) - s0 * t3
    return lp_pair + lp_3

mc_Elogq = np.mean([logq(lam_pair[i], tau_3[i]) for i in range(40_000)])
mc_Elogp = np.mean([logp(lam_pair[i], tau_3[i]) for i in range(40_000)])
print(f"  E[ln q(Lambda)] MC {mc_Elogq: .4f}   code {b_j.Elogqtau[0]: .4f}   diff {b_j.Elogqtau[0]-mc_Elogq:+.4f}")
print(f"  E[ln p(Lambda)] MC {mc_Elogp: .4f}   code {b_j.Elogptau[0]: .4f}   diff {b_j.Elogptau[0]-mc_Elogp:+.4f}")

t2_ok = (abs(code_Elogdet - mc_Elogdet) < 0.02 and abs(b_j.Elogqtau[0] - mc_Elogq) < 0.05
         and abs(b_j.Elogptau[0] - mc_Elogp) < 0.05)
print("  -> TEST 2", "PASSED" if t2_ok else "FAILED")

# ============================================================================
# TEST 3: _mask_blocks sanity
# ============================================================================
m = np.eye(5); m[0, 1] = m[1, 0] = 1; m[2, 4] = m[4, 2] = 1
blocks = _mask_blocks(m)
expected = [[0, 1], [2, 4], [3]]
t3_ok = [b.tolist() for b in blocks] == expected
print("\nTEST 3: blocks of eye(5)+links(0-1, 2-4):", [b.tolist() for b in blocks],
      "PASSED" if t3_ok else "FAILED")

assert ok and t2_ok and t3_ok, "some tests failed"
print("\nAll tests passed.")
