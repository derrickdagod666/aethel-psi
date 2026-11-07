"""A5 JOINT TT+TE+EE on REAL ACT DR6 data (v6 batch, task 3 of 4) — log-oscillatory
ripple search, null-calibrated, full joint covariance.

Extends a5_real.py (TT-only) to the joint 54-bin vector (18 TT + 18 TE + 18 EE,
1500 <= ell <= 3000) from the same real ACT DR6.02 foreground-marginalized
CMB-only SACC file (dr6_data_cmbonly.fits, LAMBDA), with the FULL 54x54
covariance including TT x TE, TT x EE, TE x EE cross-blocks. Same omega grid
(2-30, 113 points), same statistic (GLS chi2 improvement of a
[cos(om*x), sin(om*x)] fit, x = ln(ell/2200), 2 dof per omega), same
null-simulation / global-p / A95 procedure as the TT-only run.

One necessary change vs a5_real.py: the TT run worked in RELATIVE units
r = (obs-th)/th with Cov_rel = C/outer(th,th). That is ill-defined for TE,
whose best-fit D_l crosses zero inside 1500-3000 (D_TE(2200) = -3.3 muK^2,
obs range -28.6..+5.4). The joint run therefore works in ABSOLUTE D_l units:
residual d = obs - th, covariance C (absolute), ripple template
t = D_th * cos(om*x) / D_th * sin(om*x) stacked over the three spectra
(i.e. the SAME fractional ripple A, omega, phi applied to TT, TE and EE —
a common multiplicative ripple on the CMB spectrum). In absolute units the
statistic is exactly equivalent to the relative-unit statistic wherever
th != 0, so the TT-only recomputation below is a direct reproducibility
check of a5_real.py (T_max = 5.88, A_hat = 0.00703, global p = 0.241).

Seed: same SEED 20261107 as a5_real.py, stream index 6 (fresh stream; the
TT recompute therefore differs from a5_real.py only by Monte Carlo noise).

Outputs: printed report + a5_joint_results.json. Exit 0 always (measurement).
"""
import json
import os

import numpy as np
import sacc
from scipy.linalg import solve_triangular

SEED = 20261107
HERE = os.path.dirname(os.path.abspath(__file__))
SACC_FILE = os.path.join(HERE, "dr6_data_cmbonly.fits")
CMB_DAT = os.path.join(HERE, "dr6_lcdm_best_fits", "cmb.dat")
OUT_JSON = os.path.join(HERE, "a5_joint_results.json")

rng = np.random.default_rng([SEED, 6])

OMEGAS = np.linspace(2.0, 30.0, 113)
N_NULL = 3000   # null sims for T_crit (same as TT-only full run)
N_FPR = 3000    # independent null sims for FPR check
N_POW = 500     # sims per amplitude for power / A95


def load_spectra():
    s = sacc.Sacc.load_fits(SACC_FILE)
    th = np.loadtxt(CMB_DAT)
    ell_th = th[:, 0].astype(int)
    specs = {}
    for dt, col in (("TT", 1), ("TE", 2), ("EE", 6)):
        sdt = {"TT": "cl_00", "TE": "cl_0e", "EE": "cl_ee"}[dt]
        idx = [i for i, d in enumerate(s.data) if d.data_type == sdt]
        ells = np.array([s.data[i].get_tag("ell") for i in idx], dtype=float)
        obs = np.array([s.data[i].value for i in idx])
        w = s.get_bandpower_windows(idx)
        W = np.asarray(w.weight)
        v = np.asarray(w.values).astype(int)
        Dth = W.T @ th[:, col][np.searchsorted(ell_th, v)]
        C = np.ascontiguousarray(
            s.covariance.covmat[np.ix_(idx, idx)], dtype=np.float64)
        m = (ells >= 1500) & (ells <= 3000)
        ii = np.where(m)[0]
        specs[dt] = {"ell": ells[m], "obs": obs[m], "Dth": Dth[m],
                     "C": C[np.ix_(ii, ii)]}
    return specs, s


def make_stat(x, Dth, C):
    """Return (stat, noise) closures for one residual vector.

    stat(R): R rows = realisations, cols = bins. Returns (T, Ac, As) with
    T shape (n_real, n_omega); Ac/As the GLS cos/sin coefficients.
    Template columns are Dth*cos(om*x), Dth*sin(om*x): a common fractional
    ripple of amplitude A = hypot(Ac, As).
    noise(k): k Gaussian realisations drawn with covariance C.
    """
    n = len(x)
    c = np.cos(np.outer(OMEGAS, x))
    s_ = np.sin(np.outer(OMEGAS, x))
    tc = c * Dth[None, :]
    ts = s_ * Dth[None, :]
    Lchol = np.linalg.cholesky(C)

    def whiten(R):
        return solve_triangular(Lchol, R.T, lower=True).T

    cw, sw = whiten(tc), whiten(ts)
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

    def noise(k):
        z = rng.normal(size=(k, n))
        return z @ Lchol.T

    sig_comp = (Mss / det, Mcc / det)  # per-omega variances; index at omega_hat
    return stat, noise, sig_comp, (Mcc, Mcs, Mss, det)


def run_search(name, R_obs, stat, noise, M=None, n_null=N_NULL):
    """Null calibration + observed T_max for one residual vector."""
    T1 = stat(noise(n_null))[0].max(axis=1)
    T_crit = float(np.percentile(T1, 95))
    T2 = stat(noise(N_FPR))[0].max(axis=1)
    fpr = float(np.mean(T2 > T_crit))
    T_o, Ac, As = stat(R_obs[None, :])
    j = int(np.argmax(T_o[0]))
    p_glob = float((1 + np.sum(T1 >= T_o[0, j])) / (n_null + 1))
    A_hat = float(np.hypot(Ac[0, j], As[0, j]))
    out = {"T_crit": T_crit, "FPR": fpr, "T_max": float(T_o[0, j]),
           "omega_hat": float(OMEGAS[j]), "A_hat": A_hat,
           "Ac": float(Ac[0, j]), "As": float(As[0, j]),
           "phi_hat": float(np.arctan2(-As[0, j], Ac[0, j])),
           "global_p": p_glob}
    if M is not None:
        Mcc, Mcs, Mss, det = M
        out["A_hat_sigma_per_component"] = [float(np.sqrt(Mss[j] / det[j])),
                                            float(np.sqrt(Mcc[j] / det[j]))]
    print(f"{name}: T_crit(95%) = {T_crit:.2f} (FPR {fpr:.3f} on indep. set); "
          f"T_max = {T_o[0, j]:.2f} at omega = {OMEGAS[j]:.2f}, "
          f"A_hat = {A_hat:.5f} (Ac={Ac[0, j]:+.5f}, As={As[0, j]:+.5f}), "
          f"global p = {p_glob:.3g}")
    return out


def power_curve(name, x, Dth, stat, noise, T_crit, amps):
    """Detection power vs amplitude; injections share one (om, phi) across
    all spectra in the vector (common ripple)."""
    power = []
    for A in amps:
        om = rng.uniform(2.0, 30.0, N_POW)
        ph = rng.uniform(0, 2 * np.pi, N_POW)
        inj = A * np.cos(om[:, None] * x[None, :] + ph[:, None]) * Dth[None, :]
        R = noise(N_POW) + inj
        power.append(float(np.mean(stat(R)[0].max(axis=1) > T_crit)))
    power = np.array(power)
    A95 = float(np.interp(0.95, power, amps)) if power.max() >= 0.95 else float("nan")
    print(f"POWER [{name}]: " + ", ".join(f"{a:.1e}:{p:.2f}" for a, p in zip(amps, power)))
    print(f"  A95 [{name}] = {A95:.3e}" if np.isfinite(A95) else
          f"  A95 [{name}] = NOT REACHED (max power {power.max():.2f})")
    return {"amps": amps.tolist(), "power": power.tolist(), "A95": A95}


def main():
    specs, s = load_spectra()
    for dt in ("TT", "TE", "EE"):
        sp = specs[dt]
        print(f"{dt}: {len(sp['ell'])} bins, ell {sp['ell'].min():.1f}..{sp['ell'].max():.1f}; "
              f"Dth range {sp['Dth'].min():.2f}..{sp['Dth'].max():.2f}; "
              f"sigma(D) range {np.sqrt(np.diag(sp['C'])).min():.2f}..{np.sqrt(np.diag(sp['C'])).max():.2f}")
    print("TE Dth values:", np.round(specs["TE"]["Dth"], 2).tolist())

    # ---- joint vector: [TT, TE, EE], full 54x54 covariance
    order = ["TT", "TE", "EE"]
    ell_j = np.concatenate([specs[dt]["ell"] for dt in order])
    obs_j = np.concatenate([specs[dt]["obs"] for dt in order])
    Dth_j = np.concatenate([specs[dt]["Dth"] for dt in order])
    # rebuild the joint covariance from the SACC file in stacked order
    idx_all = []
    for dt in order:
        sdt = {"TT": "cl_00", "TE": "cl_0e", "EE": "cl_ee"}[dt]
        idx = [i for i, d in enumerate(s.data) if d.data_type == sdt]
        ells = np.array([s.data[i].get_tag("ell") for i in idx], dtype=float)
        m = np.where((ells >= 1500) & (ells <= 3000))[0]
        idx_all += [idx[j] for j in m]
    C_joint = np.ascontiguousarray(
        s.covariance.covmat[np.ix_(idx_all, idx_all)], dtype=np.float64)
    n_j = len(ell_j)
    print(f"\nJOINT vector: {n_j} bins (18 TT + 18 TE + 18 EE)")
    ev = np.linalg.eigvalsh(C_joint)
    print(f"  joint C: symmetric, min eig {ev[0]:.4g}, max eig {ev[-1]:.4g}, "
          f"cond {ev[-1] / ev[0]:.3g}")
    d_j = np.sqrt(np.diag(C_joint))
    corr_j = C_joint / np.outer(d_j, d_j)
    for a, na in enumerate(order):
        for b, nb in enumerate(order):
            blk = corr_j[a * 18:(a + 1) * 18, b * 18:(b + 1) * 18]
            off = blk[~np.eye(18, dtype=bool)] if a == b else blk
            print(f"  corr {na}x{nb}: mean {blk.mean():+.3f}, mean|.| "
                  f"{np.abs(off).mean():.3f}, max|.| {np.abs(blk).max():.3f}")
    print(f"  lag-1 within blocks: TT "
          f"{np.mean([corr_j[i, i + 1] for i in range(17)]):.3f}, TE "
          f"{np.mean([corr_j[18 + i, 19 + i] for i in range(17)]):.3f}, EE "
          f"{np.mean([corr_j[36 + i, 37 + i] for i in range(17)]):.3f}")

    d_j_res = obs_j - Dth_j
    chi2_joint = float(d_j_res @ np.linalg.solve(C_joint, d_j_res))
    print(f"  baseline chi2 (fixed best-fit LCDM, joint): {chi2_joint:.1f} / {n_j} dof")
    for dt in order:
        sp = specs[dt]
        dd = sp["obs"] - sp["Dth"]
        print(f"  baseline chi2 {dt}: {float(dd @ np.linalg.solve(sp['C'], dd)):.1f} / 18 dof; "
              f"mean rel resid {(np.mean(dd / sp['Dth']) * 100):+.2f}% "
              f"(n_neg={(dd < 0).sum()}/18)")

    x_j = np.log(ell_j / 2200.0)
    stat_j, noise_j, _, M_j = make_stat(x_j, Dth_j, C_joint)

    # ---- joint search
    print("\n--- JOINT TT+TE+EE ripple search (common A, omega, phi) ---")
    res_joint = run_search("JOINT", d_j_res, stat_j, noise_j, M_j)
    print(f"  A_hat sigma per component: {res_joint['A_hat_sigma_per_component'][0]:.5f} (cos), "
          f"{res_joint['A_hat_sigma_per_component'][1]:.5f} (sin); "
          f"local p at max (chi2_2dof): {float(np.exp(-res_joint['T_max'] / 2)):.3g}")

    # ---- joint power / A95
    print("\n--- JOINT power ---")
    pow_joint_v3 = power_curve("joint v3 grid", x_j, Dth_j, stat_j, noise_j,
                               res_joint["T_crit"], np.geomspace(2e-4, 3e-3, 10))
    pow_joint_ext = power_curve("joint extended grid", x_j, Dth_j, stat_j, noise_j,
                                res_joint["T_crit"], np.geomspace(2e-4, 3e-2, 13))
    # direct power at the two amplitudes the verdict turns on
    pow_check = power_curve("joint check A=0.005, 0.1", x_j, Dth_j, stat_j, noise_j,
                            res_joint["T_crit"], np.array([0.005, 0.1]))

    # ---- joint smooth-baseline refit: [1, x] per spectrum (6 params) + ripple
    print("\n--- JOINT [1, x]-per-spectrum baseline refit ---")
    Lchol_j = np.linalg.cholesky(C_joint)
    Xb = np.zeros((n_j, 6))
    for k in range(3):
        Xb[k * 18:(k + 1) * 18, 2 * k] = 1.0
        Xb[k * 18:(k + 1) * 18, 2 * k + 1] = np.log(specs[order[k]]["ell"] / 2200.0)
    Xbw = solve_triangular(Lchol_j, Xb, lower=True)
    Pb = Xbw @ np.linalg.inv(Xbw.T @ Xbw) @ Xbw.T

    def whiten_j(R):
        return solve_triangular(Lchol_j, R.T, lower=True).T

    c_j = np.cos(np.outer(OMEGAS, x_j)) * Dth_j[None, :]
    s_j = np.sin(np.outer(OMEGAS, x_j)) * Dth_j[None, :]
    cw_j = whiten_j(c_j)
    sw_j = whiten_j(s_j)

    def stat_joint_base(R):
        Rw = whiten_j(R)
        Rb = Rw - Rw @ Pb.T
        chi2_b = np.sum(Rb * Rb, axis=1)
        out = np.empty((R.shape[0], len(OMEGAS)))
        for jj in range(len(OMEGAS)):
            Xf = np.column_stack([Xbw, cw_j[jj], sw_j[jj]])
            Pf = Xf @ np.linalg.inv(Xf.T @ Xf) @ Xf.T
            Rf = Rw - Rw @ Pf.T
            out[:, jj] = chi2_b - np.sum(Rf * Rf, axis=1)
        return out

    n_jb = 1000
    T1jb = stat_joint_base(noise_j(n_jb)).max(axis=1)
    T_crit_jb = float(np.percentile(T1jb, 95))
    Tjb = stat_joint_base(d_j_res[None, :])
    j_jb = int(np.argmax(Tjb[0]))
    p_jb = float((1 + np.sum(T1jb >= Tjb[0, j_jb])) / (n_jb + 1))
    print(f"JOINT baseline refit: T_max = {Tjb[0, j_jb]:.2f} at omega = {OMEGAS[j_jb]:.2f} "
          f"vs threshold {T_crit_jb:.2f}; global p = {p_jb:.3g}")

    # ---- per-spectrum searches (3000 null sims each; TT = reproducibility check)
    print("\n--- PER-SPECTRUM searches (full within-spectrum covariance) ---")
    per = {}
    for dt in order:
        sp = specs[dt]
        x = np.log(sp["ell"] / 2200.0)
        st, nz, _, M_sp = make_stat(x, sp["Dth"], sp["C"])
        r = run_search(dt, sp["obs"] - sp["Dth"], st, nz, M_sp)
        per[dt] = r
        pw = power_curve(f"{dt} extended grid", x, sp["Dth"], st, nz,
                         r["T_crit"], np.geomspace(2e-4, 3e-2, 13))
        per[dt]["power_extended"] = pw

    summary = {
        "source": "ACT DR6.02 CMB-only SACC dr6_data_cmbonly.fits (LAMBDA), "
                  "foreground-marginalized; baseline dr6_lcdm_best_fits/cmb.dat window-binned",
        "method": "Absolute D_l units; common fractional ripple A,omega,phi on TT+TE+EE; "
                  "omega grid 2-30 (113 pts), x=ln(ell/2200); GLS chi2 improvement, 2 dof/omega; "
                  "3000 null sims for T_crit, 3000 independent for FPR, 500/amp for power. "
                  "Seed 20261107 stream 6. Absolute units chosen because TE D_l crosses zero "
                  "in-range, making relative residuals ill-defined; equivalent to a5_real.py "
                  "relative-unit statistic wherever Dth != 0.",
        "n_bins": {"TT": 18, "TE": 18, "EE": 18, "joint": 54},
        "ell_range": [1500.0, 3000.0],
        "cross_block_corr": {
            "TTxTE_mean_abs": float(np.abs(corr_j[0:18, 18:36]).mean()),
            "TTxTE_max_abs": float(np.abs(corr_j[0:18, 18:36]).max()),
            "TTxEE_mean_abs": float(np.abs(corr_j[0:18, 36:54]).mean()),
            "TTxEE_max_abs": float(np.abs(corr_j[0:18, 36:54]).max()),
            "TExEE_mean_abs": float(np.abs(corr_j[18:36, 36:54]).mean()),
            "TExEE_max_abs": float(np.abs(corr_j[18:36, 36:54]).max()),
            "lag1_TT": float(np.mean([corr_j[i, i + 1] for i in range(17)])),
            "lag1_TE": float(np.mean([corr_j[18 + i, 19 + i] for i in range(17)])),
            "lag1_EE": float(np.mean([corr_j[36 + i, 37 + i] for i in range(17)])),
        },
        "chi2_fixed_lcdm": {"joint_54dof": chi2_joint,
                            "TT": per["TT"].get("chi2", None)},
        "joint": {**res_joint,
                  "power_v3_grid": pow_joint_v3, "power_extended_grid": pow_joint_ext,
                  "power_check_A005_A01": pow_check},
        "joint_baseline_refit": {"T_max": float(Tjb[0, j_jb]),
                                 "omega": float(OMEGAS[j_jb]),
                                 "T_crit": T_crit_jb, "global_p": p_jb},
        "per_spectrum": per,
        "seed": SEED,
    }
    # per-spectrum baseline chi2 values (computed above, store explicitly)
    for dt in order:
        sp = specs[dt]
        dd = sp["obs"] - sp["Dth"]
        summary["chi2_fixed_lcdm"][dt] = float(dd @ np.linalg.solve(sp["C"], dd))
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {OUT_JSON}")


if __name__ == "__main__":
    main()
