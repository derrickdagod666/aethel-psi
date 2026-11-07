"""CAMB joint Lambda-CDM refit with the A5 ripple, on REAL ACT DR6 CMB-only
data (dr6_data_cmbonly.fits, all 135 TT+TE+EE points, full covariance).

Two least-squares fits (scipy, whitened residuals):
  (1) LCDM only: 5 params (ombh2, omch2, cosmomc_theta, logA, ns), tau=0.0544 fixed.
  (2) LCDM + ripple: same 5 + Ac, As at FIXED omega = 9.75 (the primary
      A5 omega_hat), ripple applied to the TT theory only:
      D_TT -> D_TT * (1 + Ac*cos(om*x) + As*sin(om*x)), x = ln(ell/2200).
Theory: CAMB lensed Cls, CMB_unit='muK', binned with the SACC bandpower
windows. Starting point / bounds: DR6-ACT-lite yamls/parameters/cosmology.yaml
ref values and priors. Recombination: CAMB default (Recfast); CosmoRec is
NOT installed (no gfortran after the VM restart; apt update still running)
-- the CosmoRec-vs-Recfast difference at ACT precision is far below the
ripple amplitude being tested, and is stated here, not hidden.

Writes camb_joint_refit.json + prints progress (least_squares verbose=2).
"""
import json
import os
import time

import numpy as np
import sacc
import camb
from scipy.linalg import cholesky
from scipy.optimize import least_squares

HERE = os.path.dirname(os.path.abspath(__file__))
SACC_FILE = os.path.join(HERE, "dr6_data_cmbonly.fits")
OUT_JSON = os.path.join(HERE, "camb_joint_refit.json")
OMEGA_RIPPLE = 9.75  # primary A5 omega_hat (TT, CMB-only)

s = sacc.Sacc.load_fits(SACC_FILE)
# collect per-spectrum blocks in sacc order
blocks = []
for dt, col in (("cl_00", 0), ("cl_0e", 3), ("cl_ee", 1)):
    idx = [i for i, d in enumerate(s.data) if d.data_type == dt]
    ells = np.array([s.data[i].get_tag("ell") for i in idx], dtype=float)
    obs = np.array([s.data[i].value for i in idx])
    w = s.get_bandpower_windows(idx)
    W = np.asarray(w.weight)
    v = np.asarray(w.values).astype(int)
    blocks.append({"dt": dt, "col": col, "idx": idx, "ells": ells,
                   "obs": obs, "W": W, "v": v})
    print(f"{dt}: n={len(idx)} window ell support "
          f"{v[W.any(axis=1)].min()}..{v[W.any(axis=1)].max()}", flush=True)

obs_all = np.concatenate([b["obs"] for b in blocks])
# NOTE: sacc loads the covariance big-endian (>f8); scipy's LAPACK Cholesky
# fails on it (potrf info=3) while numpy's succeeds. Force native order.
C = np.ascontiguousarray(s.covariance.covmat, dtype=np.float64)
L = cholesky(C, lower=True)
Linv = np.linalg.inv(L)


def theory_binned(pars, ripple=None):
    ombh2, omch2, theta, logA, ns = pars[:5]
    p = camb.CAMBparams()
    p.set_cosmology(ombh2=ombh2, omch2=omch2, cosmomc_theta=theta, tau=0.0544)
    p.InitPower.set_params(As=1e-10 * np.exp(logA), ns=ns)
    p.set_for_lmax(8500, lens_potential_accuracy=1)  # window values run to 8501
    res = camb.get_results(p)
    cl = res.get_total_cls(8500, CMB_unit="muK")  # cols TT, EE, BB, TE.
    # NOTE: with CMB_unit set, CAMB returns D_l = l(l+1)C_l/2pi directly
    # (verified: D_TT(220)=5799 vs cmb.dat 5700; D_TT(2000)=237.7 vs 234.5).
    # Do NOT multiply by l(l+1)/2pi again.
    ell_grid = np.arange(8501)
    Dl = {0: cl[:, 0], 1: cl[:, 1], 3: cl[:, 3]}
    out = []
    for b in blocks:
        # window values run to ell=8501, one past CAMB's lmax=8500 output;
        # weights there are exactly zero (support ends 6325), so clipping
        # the index is exact.
        vidx = np.clip(np.searchsorted(ell_grid, b["v"]), 0, len(ell_grid) - 1)
        D = b["W"].T @ Dl[b["col"]][vidx]
        if ripple is not None and b["dt"] == "cl_00":
            Ac, As = ripple
            x = np.log(b["ells"] / 2200.0)
            D = D * (1 + Ac * np.cos(OMEGA_RIPPLE * x)
                     + As * np.sin(OMEGA_RIPPLE * x))
        out.append(D)
    return np.concatenate(out)


def resid(pars, ripple_on):
    if ripple_on:
        th = theory_binned(pars[:5], ripple=pars[5:7])
    else:
        th = theory_binned(pars)
    return Linv @ (obs_all - th)


x0 = np.array([0.02239, 0.12, 0.0104170, 3.064, 0.9729])
lb = np.array([0.017, 0.09, 0.01030, 2.6, 0.9])
ub = np.array([0.027, 0.15, 0.01050, 3.5, 1.1])

t0 = time.time()
r0 = resid(x0, False)
print(f"start-point chi2 (LCDM only): {float(r0 @ r0):.1f} / 135 dof "
      f"(CAMB eval {time.time() - t0:.0f}s)", flush=True)

sol1 = least_squares(resid, x0, args=(False,), bounds=(lb, ub),
                     x_scale=np.array([0.001, 0.005, 1e-5, 0.05, 0.01]),
                     verbose=2, max_nfev=80)
chi2_1 = float(2 * sol1.cost)
print(f"FIT1 LCDM-only: chi2={chi2_1:.1f} params={sol1.x} nfev={sol1.nfev} "
      f"({time.time() - t0:.0f}s)", flush=True)

x02 = np.concatenate([sol1.x, [0.0, 0.0]])
lb2 = np.concatenate([lb, [-0.1, -0.1]])
ub2 = np.concatenate([ub, [0.1, 0.1]])
sol2 = least_squares(resid, x02, args=(True,), bounds=(lb2, ub2),
                     x_scale=np.array([0.001, 0.005, 1e-5, 0.05, 0.01, 0.005, 0.005]),
                     verbose=2, max_nfev=100)
chi2_2 = float(2 * sol2.cost)
Ac, As = sol2.x[5], sol2.x[6]
A_rip = float(np.hypot(Ac, As))
print(f"FIT2 LCDM+ripple: chi2={chi2_2:.1f} dchi2={chi2_1 - chi2_2:.2f} "
      f"Ac={Ac:.5f} As={As:.5f} A={A_rip:.5f} params={sol2.x} nfev={sol2.nfev} "
      f"({time.time() - t0:.0f}s)", flush=True)

with open(OUT_JSON, "w") as f:
    json.dump({
        "omega_ripple_fixed": OMEGA_RIPPLE,
        "tau_fixed": 0.0544,
        "recombination": "CAMB default (Recfast); CosmoRec not installed",
        "start_chi2": float(r0 @ r0),
        "fit_lcdm_only": {"chi2": chi2_1, "params": sol1.x.tolist(),
                          "nfev": int(sol1.nfev)},
        "fit_lcdm_ripple": {"chi2": chi2_2, "dchi2_vs_lcdm": chi2_1 - chi2_2,
                            "params": sol2.x.tolist(), "Ac": float(Ac),
                            "As": float(As), "A_ripple": A_rip,
                            "nfev": int(sol2.nfev)},
        "seconds": time.time() - t0,
    }, f, indent=2)
print("Wrote", OUT_JSON)
