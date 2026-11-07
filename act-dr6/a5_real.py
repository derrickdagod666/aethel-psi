"""A5 on REAL ACT DR6 data (fix 6c) — log-oscillatory ripple search, null-calibrated.

Mirrors aethel_psi_suite_v3.py algorithm_a5 exactly (same omega grid 2-30,
113 points; same seed 20261107; same Monte Carlo sizes for the full run),
but replaces the synthetic spectrum with the real ACT DR6 foreground-
marginalized CMB-only bandpowers (dr6_data_cmbonly.fits, LAMBDA, DR6.02)
and the synthetic diagonal/AR(1) errors with the real 135x135 covariance.

Data:
  - TT bandpowers D_l (foreground-subtracted, CMB-only), 18 bins at
    1500 <= ell <= 3000, from dr6_data_cmbonly.fits (sacc).
  - Baseline: ACT DR6 best-fit LCDM spectrum dr6_lcdm_best_fits/cmb.dat,
    binned with the SACC bandpower windows (columns sum to 1).
  - Residuals r = (D_obs - D_th)/D_th; Cov_rel = C_TT / outer(D_th, D_th).
  - Statistic: GLS chi2 improvement of a [cos(om*x), sin(om*x)] fit,
    x = ln(ell/2200), 2 dof per omega; T = max over omega.
  - Null calibration: Gaussian sims drawn with the REAL covariance
    (Cholesky), same sizes as v3 full run (n1=3000, nin=500, n_j=1000).
  - A95: power vs amplitude; v3 grid (2e-4..3e-3) PLUS an extended grid
    (2e-4..3e-2) because real errors (~1-4%) are ~10x the synthetic
    ones (0.25%), so the v3 grid cannot reach 95% power on real data.
  - Joint smooth-baseline refit: [1, x] baseline refit jointly with the
    ripple in whitened space (v3 block (c) analogue, real data).
  - Foreground: the input file IS ACT's foreground-marginalized product
    (DR6-ACT-lite; Louis et al. 2025). No extra foreground template is
    fitted here; a cross-check vs the raw 150x150 TT spectrum quantifies
    the foreground contribution that was subtracted.

Outputs: printed report + JSON summary a5_real_results.json.
Exit 0 always (this is a measurement, not a PASS/FAIL gate); the
numbers are what they are.
"""
import json
import os
import sys

import numpy as np
import sacc
from scipy.linalg import solve_triangular

SEED = 20261107
HERE = os.path.dirname(os.path.abspath(__file__))
SACC_FILE = os.path.join(HERE, "dr6_data_cmbonly.fits")
CMB_DAT = os.path.join(HERE, "dr6_lcdm_best_fits", "cmb.dat")
OUT_JSON = os.path.join(HERE, "a5_real_results.json")

rng = np.random.default_rng([SEED, 5])  # same stream index as v3 A5 (rg(5))


def load_real():
    s = sacc.Sacc.load_fits(SACC_FILE)
    th = np.loadtxt(CMB_DAT)
    ell_th, Dtt_th = th[:, 0].astype(int), th[:, 1]
    idx = [i for i, d in enumerate(s.data) if d.data_type == "cl_00"]
    ells = np.array([s.data[i].get_tag("ell") for i in idx], dtype=float)
    obs = np.array([s.data[i].value for i in idx])
    w = s.get_bandpower_windows(idx)
    W = np.asarray(w.weight)
    v = np.asarray(w.values).astype(int)
    Dgrid = Dtt_th[np.searchsorted(ell_th, v)]
    Dth = W.T @ Dgrid
    C = s.covariance.covmat[np.ix_(idx, idx)]
    m = (ells >= 1500) & (ells <= 3000)
    return ells[m], obs[m], Dth[m], C[np.ix_(np.where(m)[0], np.where(m)[0])], s, idx


def main():
    ell, obs, Dth, C, s, idx_all = load_real()
    n = len(ell)
    print(f"REAL DATA: {n} TT bins, ell {ell.min():.1f}..{ell.max():.1f}")
    print(f"  obs D_l: {np.round(obs, 1).tolist()}")
    print(f"  best-fit LCDM D_l (window-binned): {np.round(Dth, 1).tolist()}")
    r = (obs - Dth) / Dth
    Cov = C / np.outer(Dth, Dth)
    sig_rel = np.sqrt(np.diag(Cov))
    print(f"  rel residuals %: {np.round(100 * r, 2).tolist()}")
    print(f"  sigma_rel %: {np.round(100 * sig_rel, 2).tolist()}")
    print(f"  mean rel resid: {r.mean() * 100:.2f}%  (n_neg={(r < 0).sum()}/{n})")
    # baseline goodness of fit, fixed best-fit LCDM
    chi2_base = float(r @ np.linalg.solve(Cov, r))
    print(f"  chi2 of fixed best-fit LCDM (TT 1500-3000): {chi2_base:.1f} / {n} dof")
    # covariance structure actually present (v3 assumed AR(1) rho=0.3)
    corr = Cov / np.outer(sig_rel, sig_rel)
    lag1 = float(np.mean([corr[i, i + 1] for i in range(n - 1)]))
    print(f"  real TT-block mean lag-1 corr: {lag1:.3f} "
          f"(v3 synthetic assumption was AR(1) rho=0.3)")
    print(f"  mean |corr| off-diag: {float(np.mean(np.abs(corr - np.eye(n)))):.3f}")

    x = np.log(ell / 2200.0)
    omegas = np.linspace(2.0, 30.0, 113)
    c, s_ = np.cos(np.outer(omegas, x)), np.sin(np.outer(omegas, x))

    Lchol = np.linalg.cholesky(Cov)

    def whiten(R):
        return solve_triangular(Lchol, R.T, lower=True).T

    cw, sw = whiten(c), whiten(s_)
    Mcc = np.sum(cw * cw, axis=1)
    Mcs = np.sum(cw * sw, axis=1)
    Mss = np.sum(sw * sw, axis=1)
    det = Mcc * Mss - Mcs ** 2

    def stat(R):
        Rw = whiten(R)
        bc = Rw @ cw.T
        bs = Rw @ sw.T
        T = (Mss * bc ** 2 - 2 * Mcs * bc * bs + Mcc * bs ** 2) / det
        Ac = (Mss * bc - Mcs * bs) / det
        As = (Mcc * bs - Mcs * bc) / det
        return T, Ac, As

    def noise_cov(k):
        z = rng.normal(size=(k, n))
        return z @ Lchol.T

    # ---- null calibration (real covariance), v3 full-run sizes
    n1 = 3000
    T1 = stat(noise_cov(n1))[0].max(axis=1)
    T_crit = float(np.percentile(T1, 95))
    T2 = stat(noise_cov(n1))[0].max(axis=1)
    fpr = float(np.mean(T2 > T_crit))
    se = float(np.sqrt(0.05 * 0.95 / n1))
    print(f"\nNULL (real cov, {n1} sims): T_crit(95%) = {T_crit:.2f}; "
          f"independent-set FPR = {fpr:.3f} +/- {se:.3f} "
          f"(v3 synthetic diagonal: T_crit 10.19, FPR 0.050 +/- 0.004)")

    # ---- observed
    T_o, Ac, As = stat(r[None, :])
    j = int(np.argmax(T_o[0]))
    p_glob = float((1 + np.sum(T1 >= T_o[0, j])) / (n1 + 1))
    A_hat = float(np.hypot(Ac[0, j], As[0, j]))
    ph_hat = float(np.arctan2(-As[0, j], Ac[0, j]))
    sc = float(np.sqrt(Mss[j] / det[j]))
    ss_ = float(np.sqrt(Mcc[j] / det[j]))
    print(f"OBSERVED: T_max = {T_o[0, j]:.2f} at omega_hat = {omegas[j]:.2f}, "
          f"A_hat = {A_hat:.5f} (+/- ~{0.5 * (sc + ss_):.5f} per component), "
          f"phi_hat = {ph_hat:.2f}")
    print(f"  global p = {p_glob:.3g} (floor 1/(N+1) = {1 / (n1 + 1):.1e}); "
          f"T_crit = {T_crit:.2f}; detection at 5% global: {T_o[0, j] > T_crit}")
    # per-omega local significance at the max
    print(f"  local p at max (chi2_2dof): {float(np.exp(-T_o[0, j] / 2)):.3g}")

    # ---- power / A95: v3 grid + extended grid
    nin = 500
    results = {}
    for label, amps in (("v3_grid", np.geomspace(2e-4, 3e-3, 10)),
                        ("extended", np.geomspace(2e-4, 3e-2, 13))):
        power = []
        for A in amps:
            om = rng.uniform(2.0, 30.0, nin)
            ph = rng.uniform(0, 2 * np.pi, nin)
            R = noise_cov(nin) + A * np.cos(om[:, None] * x[None, :] + ph[:, None])
            power.append(float(np.mean(stat(R)[0].max(axis=1) > T_crit)))
        power = np.array(power)
        A95 = float(np.interp(0.95, power, amps)) if power.max() >= 0.95 else float("nan")
        results[label] = {"amps": amps.tolist(), "power": power.tolist(), "A95": A95}
        print(f"POWER [{label}]: " + ", ".join(f"{a:.1e}:{p:.2f}" for a, p in zip(amps, power)))
        print(f"  A95 [{label}] = {A95:.3e}" if np.isfinite(A95) else
              f"  A95 [{label}] = NOT REACHED in grid (max power {power.max():.2f})")

    # ---- joint smooth-baseline refit (v3 block (c) analogue, real data)
    Xb = np.column_stack([np.ones_like(x), x])
    # NOTE: whiten() expects rows = realisations; a design matrix (n, k)
    # must be left-multiplied by L^-1 directly, not passed through whiten().
    Xbw = solve_triangular(Lchol, Xb, lower=True)
    Pb = Xbw @ np.linalg.inv(Xbw.T @ Xbw) @ Xbw.T

    def stat_joint(R):
        Rw = whiten(R)
        Rb = Rw - Rw @ Pb.T
        chi2_b = np.sum(Rb * Rb, axis=1)
        out = np.empty((R.shape[0], len(omegas)))
        for jj in range(len(omegas)):
            Xf = np.column_stack([Xbw, cw[jj], sw[jj]])
            Pf = Xf @ np.linalg.inv(Xf.T @ Xf) @ Xf.T
            Rf = Rw - Rw @ Pf.T
            out[:, jj] = chi2_b - np.sum(Rf * Rf, axis=1)
        return out

    n_j = 1000
    T1j = stat_joint(noise_cov(n_j)).max(axis=1)
    T_crit_j = float(np.percentile(T1j, 95))
    Tj = stat_joint(r[None, :])
    j_j = int(np.argmax(Tj[0]))
    p_j = float((1 + np.sum(T1j >= Tj[0, j_j])) / (n_j + 1))
    print(f"\nJOINT [1, x] BASELINE REFIT: T_max = {Tj[0, j_j]:.2f} at omega = {omegas[j_j]:.2f} "
          f"vs threshold {T_crit_j:.2f}; global p = {p_j:.3g}")
    print(f"  baseline-only chi2 (whitened): {float(np.sum((whiten(r[None, :]) - whiten(r[None, :]) @ Pb.T) ** 2)):.1f}")

    # ---- diagonal-only version (isolates real noise level vs real correlations)
    w_diag = 1.0 / sig_rel ** 2
    Mdcc = (c * c * w_diag).sum(1)
    Mdcs = (c * s_ * w_diag).sum(1)
    Mdss = (s_ * s_ * w_diag).sum(1)
    ddet = Mdcc * Mdss - Mdcs ** 2

    def stat_diag(R):
        bc = R @ (c * w_diag).T
        bs = R @ (s_ * w_diag).T
        return (Mdss * bc ** 2 - 2 * Mdcs * bc * bs + Mdcc * bs ** 2) / ddet

    def noise_diag(k):
        return rng.normal(size=(k, n)) * sig_rel

    T1d = stat_diag(noise_diag(n1)).max(axis=1)
    T_crit_d = float(np.percentile(T1d, 95))
    Td = stat_diag(r[None, :])
    jd = int(np.argmax(Td[0]))
    p_d = float((1 + np.sum(T1d >= Td[0, jd])) / (n1 + 1))
    print(f"\nDIAGONAL-ONLY (same real data/errors, correlations dropped): "
          f"T_crit = {T_crit_d:.2f}; T_max = {Td[0, jd]:.2f} at omega = {omegas[jd]:.2f}; "
          f"global p = {p_d:.3g}")

    # ---- TE / EE consistency (same machinery, own blocks; secondary)
    for dt, col in (("TE", 2), ("EE", 6)):
        idx = [i for i, d in enumerate(s.data) if d.data_type == ("cl_0e" if dt == "TE" else "cl_ee")]
        ells2 = np.array([s.data[i].get_tag("ell") for i in idx], dtype=float)
        obs2 = np.array([s.data[i].value for i in idx])
        th_all = np.loadtxt(CMB_DAT)
        Dcol = th_all[:, col]
        w2 = s.get_bandpower_windows(idx)
        W2 = np.asarray(w2.weight)
        v2 = np.asarray(w2.values).astype(int)
        Dth2 = W2.T @ Dcol[np.searchsorted(th_all[:, 0].astype(int), v2)]
        C2 = s.covariance.covmat[np.ix_(idx, idx)]
        m2 = (ells2 >= 1500) & (ells2 <= 3000)
        ii = np.where(m2)[0]
        r2v = (obs2[m2] - Dth2[m2]) / Dth2[m2]
        Cov2 = C2[np.ix_(ii, ii)] / np.outer(Dth2[m2], Dth2[m2])
        L2 = np.linalg.cholesky(Cov2)
        x2 = np.log(ells2[m2] / 2200.0)
        c2 = np.cos(np.outer(omegas, x2))
        s2m = np.sin(np.outer(omegas, x2))
        cw2 = solve_triangular(L2, c2.T, lower=True).T
        sw2 = solve_triangular(L2, s2m.T, lower=True).T
        Mc2 = np.sum(cw2 * cw2, axis=1)
        Mx2 = np.sum(cw2 * sw2, axis=1)
        Ms2 = np.sum(sw2 * sw2, axis=1)
        dt2 = Mc2 * Ms2 - Mx2 ** 2
        Rw2 = solve_triangular(L2, r2v[None, :].T, lower=True).T
        bc2 = Rw2 @ cw2.T
        bs2 = Rw2 @ sw2.T
        T2v = (Ms2 * bc2 ** 2 - 2 * Mx2 * bc2 * bs2 + Mc2 * bs2 ** 2) / dt2
        j2 = int(np.argmax(T2v[0]))
        # quick null for the p-value (1000 sims)
        z = rng.normal(size=(1000, len(ii)))
        Rn = z @ L2.T
        Rwn = solve_triangular(L2, Rn.T, lower=True).T
        bcn = Rwn @ cw2.T
        bsn = Rwn @ sw2.T
        Tn = ((Ms2 * bcn ** 2 - 2 * Mx2 * bcn * bsn + Mc2 * bsn ** 2) / dt2).max(axis=1)
        p2 = float((1 + np.sum(Tn >= T2v[0, j2])) / 1001)
        print(f"{dt} (secondary, same 1500-3000 range): T_max = {T2v[0, j2]:.2f} at "
              f"omega = {omegas[j2]:.2f}; global p ~ {p2:.3g} (1000 null sims)")

    summary = {
        "source": "ACT DR6.02 CMB-only SACC dr6_data_cmbonly.fits (LAMBDA), foreground-marginalized; "
                  "baseline dr6_lcdm_best_fits/cmb.dat window-binned",
        "n_bins_TT_1500_3000": int(n),
        "ell": ell.tolist(),
        "rel_resid": r.tolist(),
        "sigma_rel": sig_rel.tolist(),
        "chi2_fixed_lcdm_TT": chi2_base,
        "lag1_corr_TT_block": lag1,
        "T_crit_real_cov": T_crit,
        "FPR_independent_set": fpr,
        "T_max": float(T_o[0, j]),
        "omega_hat": float(omegas[j]),
        "A_hat": A_hat,
        "phi_hat": ph_hat,
        "A_hat_sigma_per_component": [sc, ss_],
        "global_p": p_glob,
        "A95_v3_grid": results["v3_grid"]["A95"],
        "A95_extended_grid": results["extended"]["A95"],
        "power_v3_grid": results["v3_grid"]["power"],
        "power_extended_grid": results["extended"]["power"],
        "joint_baseline_T_max": float(Tj[0, j_j]),
        "joint_baseline_omega": float(omegas[j_j]),
        "joint_baseline_T_crit": T_crit_j,
        "joint_baseline_global_p": p_j,
        "diagonal_T_crit": T_crit_d,
        "diagonal_T_max": float(Td[0, jd]),
        "diagonal_omega": float(omegas[jd]),
        "diagonal_global_p": p_d,
        "seed": SEED,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
