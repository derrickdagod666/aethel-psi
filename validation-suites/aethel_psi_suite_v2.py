#!/usr/bin/env python3
r"""
THE AETHEL-PSI SYNTHESIS: EXECUTION & VALIDATION SUITE  (v2, falsifiable edition)
Author: Derrick Irons / The Aethelred Institute   ORCID: 0009-0001-7064-6611

What changed relative to v1, and why
------------------------------------
v1 printed PASS unconditionally and recovered synthetic data with the same model that
generated it. v2 enforces one rule: *every test must be able to fail.*

  Gate 1  : sympy verifies (i) tensor identities on a generic static metric, (ii) the
            metric variation of  f(t) R  against a minisuperspace Euler-Lagrange
            derivation (checks the relative sign/coefficient of the (g Box - Nabla Nabla)
            term, not just the trace), and (iii) the trace equation.
  A1      : fixes the target bug (ln A_target must be ln(A_target/A_0)), the sign of the
            constant exponent, and reports the true status: one constraint, two
            parameters -> UNDERDETERMINED (status OPEN), plus chart-unit dependence.
  A2      : injection-recovery (bias/coverage), BIC model discrimination, and the
            false-positive rate on no-flow data.
  A3      : closed-form + root-solved phantom crossing, continuity check of rho_DE(a),
            and a unit-tested pre-registered decision rule. It does NOT integrate Psi
            dynamics (none are specified); that stays OPEN.
  A4      : proves the achromatic-delay / emission-time degeneracy (rank deficiency),
            fits the identifiable model, and tabulates detectability vs timing scatter.
  A5      : replaces "A_fit * 2.5" with a null-simulation global p-value, an independent
            false-alarm test, an effective trial count, and a 95%-power amplitude.
  Gate 3  : noise-weighted inner product with a PSD, calibrated noise projection,
            mismatch ~ gamma^2 scaling check, parameter-count-corrected SNR threshold,
            and a null/injection test of the linear gamma estimator.

All data here are SYNTHETIC. Swap in real arrays where marked "DATA HOOK".
Exit status is 1 if any check is FAIL. Status OPEN = scientifically unresolved.
Run:  python aethel_psi_suite_v2.py [--quick] [--seed N]
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.optimize import brentq, curve_fit

RESULTS: list[tuple[str, str, str]] = []


def header(title: str) -> None:
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)


def record(name: str, ok: bool, detail: str = "", status: str | None = None) -> bool:
    status = status or ("PASS" if ok else "FAIL")
    RESULTS.append((name, status, detail))
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def info(name: str, detail: str) -> None:
    record(name, True, detail, status="INFO")


def open_item(name: str, detail: str) -> None:
    record(name, True, detail, status="OPEN")


# ==============================================================================
# GATE 1: METRIC VARIATION (sympy)
# ==============================================================================
def gate1_metric_variation() -> None:
    header("Gate 1: tensor identities, minisuperspace variation, trace equation")
    try:
        import sympy as sp
        from sympy.calculus.euler import euler_equations
    except ImportError:
        record("sympy available", False, "pip install sympy")
        return

    def geometry(g, X):
        n = len(X)
        ginv = g.inv().applyfunc(sp.simplify)
        Gam = [[[sum(ginv[a, d] * (sp.diff(g[d, c], X[b]) + sp.diff(g[d, b], X[c])
                                   - sp.diff(g[b, c], X[d])) for d in range(n)) / 2
                 for c in range(n)] for b in range(n)] for a in range(n)]

        def riem(a, b, c, d):  # R^a_{bcd}
            return (sp.diff(Gam[a][b][d], X[c]) - sp.diff(Gam[a][b][c], X[d])
                    + sum(Gam[a][c][e] * Gam[e][b][d] - Gam[a][d][e] * Gam[e][b][c]
                          for e in range(n)))

        Ric = sp.Matrix(n, n, lambda b, d: sp.simplify(sum(riem(a, b, a, d) for a in range(n))))
        Rs = sp.simplify(sum(ginv[b, d] * Ric[b, d] for b in range(n) for d in range(n)))
        G = (Ric - Rs * g / 2).applyfunc(sp.simplify)
        return dict(g=g, ginv=ginv, Gam=Gam, R=Rs, G=G, n=n)

    def hess_box(geo, f, X):
        n = geo["n"]
        H = sp.Matrix(n, n, lambda a, b: sp.diff(f, X[a], X[b])
                      - sum(geo["Gam"][c][a][b] * sp.diff(f, X[c]) for c in range(n)))
        box = sp.simplify(sum(geo["ginv"][a, b] * H[a, b] for a in range(n) for b in range(n)))
        return H, box

    def trace(geo, M):
        n = geo["n"]
        return sum(geo["ginv"][a, b] * M[a, b] for a in range(n) for b in range(n))

    def is_zero(expr) -> bool:
        return sp.simplify(expr) == 0

    kappa, xi, Ttr = sp.symbols("kappa xi T_tr")

    # (i) generic static metric, f(r, theta): purely geometric identities in d = 4
    t, r, th, ph = sp.symbols("t r theta phi")
    X = [t, r, th, ph]
    A, B = sp.Function("A")(r), sp.Function("B")(r)
    gS = sp.diag(-A, B, r**2, r**2 * sp.sin(th) ** 2)
    geoS = geometry(gS, X)
    fS = sp.Function("f")(r, th)
    HS, boxS = hess_box(geoS, fS, X)
    record("tr(G) = -R  (generic static metric)", is_zero(trace(geoS, geoS["G"]) + geoS["R"]))
    record("g^{mn}(g_{mn} Box - Nabla_m Nabla_n) f = 3 Box f  (generic metric, f(r,theta))",
           is_zero(trace(geoS, gS * boxS - HS) - 3 * boxS))

    # (iii) trace equation on the same generic metric with Psi(r, theta)
    PsiS = sp.Function("Psi")(r, th)
    HP, boxP = hess_box(geoS, PsiS**2, X)
    EqS = (1 + 2 * kappa * xi * PsiS**2) * geoS["G"] + 2 * kappa * xi * (gS * boxP - HP)
    lhs_trace = trace(geoS, EqS) - kappa * Ttr
    claimed = -(geoS["R"] * (1 + 2 * kappa * xi * PsiS**2) - 6 * kappa * xi * boxP + kappa * Ttr)
    record("trace eq: R(1+2*kappa*xi*Psi^2) = 6*kappa*xi*Box(Psi^2) - kappa*T  (generic metric)",
           is_zero(lhs_trace - claimed))

    # (ii) minisuperspace variation of  S = int sqrt(-g) f R  on FRW with lapse N(t)
    tt = sp.symbols("t")
    N, a, f = sp.Function("N")(tt), sp.Function("a")(tt), sp.Function("f")(tt)
    Xf = [tt, *sp.symbols("x y z")]
    gF = sp.diag(-N**2, a**2, a**2, a**2)
    geoF = geometry(gF, Xf)
    HF, boxF = hess_box(geoF, f, Xf)
    E = f * geoF["G"] + gF * boxF - HF  # E_{mn} = f G_{mn} + (g Box - Nabla Nabla) f
    L = N * a**3 * f * geoF["R"]
    eqs = euler_equations(L, [N, a], tt)
    EL_N = eqs[0].lhs.subs(N, 1).doit()
    EL_a = eqs[1].lhs.subs(N, 1).doit()
    E00 = sp.simplify(E[0, 0].subs(N, 1).doit())
    E11 = sp.simplify(E[1, 1].subs(N, 1).doit())
    # delta S = int sqrt(-g) E_{mn} delta g^{mn}  =>  dS/dN = 2 a^3 E_00,  dS/da = -6 E_11 (N=1)
    record("variation w.r.t. lapse:  dS/dN = 2 a^3 E_00   (fixes G-term and (g Box - Nabla Nabla) term)",
           is_zero(EL_N - 2 * a**3 * E00))
    record("variation w.r.t. scale factor:  dS/da = -6 E_11",
           is_zero(EL_a + 6 * E11))
    info("scope", "Checks the tensor structure and trace on test metrics. It does not check the "
                  "Psi dynamics, boundary terms, or the 'FCM eigenstate compatibility' claim.")


# ==============================================================================
# A1: RG FLOW BRIDGE
# ==============================================================================
def algorithm_a1() -> None:
    header("Algorithm A1: RG running-exponent bridge (chart -> physical)")
    A0, lam = 54.0, 945.0
    omega0 = 2.0 * np.pi / lam
    A_t, omega_t = 0.1, 470.0
    ratio = omega_t / omega0
    T = np.log(ratio)
    print(f"  chart omega_0 = {omega0:.6f}; omega_phys/omega_0 = {ratio:.2f}; ln ratio = {T:.4f}")

    # Single power law A = A0 * ratio^(-gamma)
    gamma_needed = -np.log(A_t / A0) / T
    A_edition1 = A0 * ratio ** (-1.5)
    gap = abs(np.log10(A_edition1 / A_t))
    record("first-edition gamma = 1.5 refuted",
           gap > 1.0,
           f"A(gamma=1.5) = {A_edition1:.3e} vs target {A_t} ({gap:.2f} orders); "
           f"constant gamma needed = {gamma_needed:+.4f}")

    def integrate(g0: float, g1: float) -> float:
        sol = solve_ivp(lambda s, y: [-(g0 + g1 * s)], (0.0, T), [np.log(A0)],
                        rtol=1e-12, atol=1e-14)
        return sol.y[0, -1]

    def g1_from(g0: float) -> float:  # ln(A_t/A0) = -g0 T - 0.5 g1 T^2
        return 2.0 * (np.log(A0 / A_t) - g0 * T) / T**2

    worst = 0.0
    print(f"  {'gamma_0':>8} {'gamma_1':>10} {'A(mu_phys)':>12}")
    for g0 in (0.0, 0.25, 0.5, 1.0, 1.5):
        g1 = g1_from(g0)
        A_pred = np.exp(integrate(g0, g1))
        worst = max(worst, abs(integrate(g0, g1) - np.log(A_t)))
        print(f"  {g0:8.2f} {g1:10.5f} {A_pred:12.6f}")
    record("two-stage flow integrates to A_target (ODE vs closed form)", worst < 1e-8,
           f"max |ln A_pred - ln A_target| = {worst:.2e}")

    open_item("A1 is underdetermined",
              "1 amplitude constraint, 2 free exponents: every gamma_0 above 'passes'. The family "
              "is a fit, not a prediction. Derive beta_A from the action or add independent "
              "constraints.")
    open_item("epsilon target unused",
              "eps ~ 1e-29 has no flow equation in the text, so it cannot constrain (gamma_0, gamma_1).")
    for c in (1e-2, 1.0, 1e2):
        g = -np.log(A_t / A0) / np.log(ratio * c)
        info(f"unit dependence, chart->SI factor c = {c:g}", f"constant gamma needed = {g:+.4f}")
    open_item("chart-to-physical scale map",
              "ln(mu/mu_0) shifts by ln(c) under a unit redefinition, so the exponent is meaningless "
              "until the dimensionful map between chart and rad/s is derived.")


# ==============================================================================
# A2: SPECTRAL DIMENSION
# ==============================================================================
def logistic(E, k, x0):
    return 4.0 - 2.0 / (1.0 + np.exp(-k * (E - x0)))


def gauss_flow(E, s):
    return 2.0 + 2.0 * np.exp(-((E / s) ** 2))


def bic(y, yhat, sig, npar):
    chi2 = float(np.sum(((y - yhat) / sig) ** 2))
    return chi2 + npar * np.log(len(y))


def fit_spectral_dimension(E, D, sig):
    """DATA HOOK: pass real CDT/LQG (E/M_Pl, D_s, sigma) arrays here."""
    popt, pcov = curve_fit(logistic, E, D, p0=[5.0, 0.4], sigma=np.full_like(E, sig),
                           absolute_sigma=True, bounds=([0.1, -1.0], [200.0, 3.0]), maxfev=20000)
    return popt, pcov


def algorithm_a2(rng, quick: bool) -> None:
    header("Algorithm A2: spectral dimension D(E), 4 -> 2 logistic ansatz")
    E = np.linspace(0.0, 1.2, 25)
    sig, k_true, x0_true = 0.03, 8.0, 0.5
    ntr = 100 if quick else 400

    zk, zx, fail = [], [], 0
    for _ in range(ntr):
        D = logistic(E, k_true, x0_true) + rng.normal(0, sig, E.size)
        try:
            p, c = fit_spectral_dimension(E, D, sig)
        except RuntimeError:
            fail += 1
            continue
        zk.append((p[0] - k_true) / np.sqrt(c[0, 0]))
        zx.append((p[1] - x0_true) / np.sqrt(c[1, 1]))
    zk, zx = np.array(zk), np.array(zx)
    n = len(zk)
    for nm, z in (("k", zk), ("x0", zx)):
        record(f"injection-recovery {nm}: unbiased, correctly-sized errors",
               abs(z.mean()) < 4 / np.sqrt(n) + 0.1 and 0.85 < z.std() < 1.2 and 0.58 < np.mean(np.abs(z) < 1) < 0.78,
               f"mean z = {z.mean():+.3f}, std z = {z.std():.3f}, 68% coverage = {np.mean(np.abs(z) < 1):.2f} "
               f"(n = {n}, fit failures = {fail})")

    # Direction and limits of the fitted flow
    D0 = logistic(E, k_true, x0_true)
    record("flow runs 4 -> 2 (monotone decreasing, D(0) ~ 4, D(1.2) ~ 2)",
           bool(np.all(np.diff(D0) < 0)) and abs(D0[0] - 4) < 0.1 and abs(D0[-1] - 2) < 0.1,
           f"D(0) = {D0[0]:.3f}, D(1.2) = {D0[-1]:.3f}")

    # Model discrimination (BIC) and the no-flow false-positive rate
    def prefers(D, true_is_logistic: bool):
        p, _ = fit_spectral_dimension(E, D, sig)
        b_log = bic(D, logistic(E, *p), sig, 2)
        sg, _ = curve_fit(gauss_flow, E, D, p0=[0.6], sigma=np.full_like(E, sig), absolute_sigma=True,
                          bounds=([0.05], [10.0]))
        b_g = bic(D, gauss_flow(E, *sg), sig, 1)
        b_c = bic(D, np.full_like(E, 4.0), sig, 0)
        return b_log, b_g, b_c

    m = 100 if quick else 300
    right, fp = 0, 0
    for _ in range(m):
        b_log, b_g, _ = prefers(logistic(E, k_true, x0_true) + rng.normal(0, sig, E.size), True)
        right += (b_g - b_log) > 10
        b_log, _, b_c = prefers(4.0 + rng.normal(0, sig, E.size), False)
        fp += (b_c - b_log) > 10  # logistic wrongly preferred over constant D=4
    record("BIC recovers logistic truth over Gaussian-flow alternative (dBIC > 10)",
           right / m >= 0.9, f"fraction = {right / m:.2f}")
    record("false-positive rate of 'flow detected' on constant D = 4 data < 2%",
           fp / m < 0.02, f"fraction = {fp / m:.3f}")
    info("synthetic only", "No CDT/LQG dataset is loaded; this validates the pipeline, not the physics.")


# ==============================================================================
# A3: DARK ENERGY TRAJECTORY AND DECISION RULE
# ==============================================================================
def w_cpl(a, w0, wa):
    return w0 + wa * (1.0 - a)


def rho_cpl(a, w0, wa):
    return a ** (-3.0 * (1.0 + w0 + wa)) * np.exp(-3.0 * wa * (1.0 - a))


def decision_rule(x_hat, cov, model=(-0.752, -0.860), lcdm=(-1.0, 0.0), nsigma=3.0):
    """Pre-registered rule on (w0, wa). Thresholds are fixed here, before data (DATA HOOK).
    d2 = Mahalanobis distance^2, compared with the 2-dof chi2 quantile for nsigma (two-sided)."""
    crit = stats.chi2.isf(2 * stats.norm.sf(nsigma), df=2)
    ci = np.linalg.inv(cov)

    def d2(mu):
        d = np.asarray(x_hat) - np.asarray(mu)
        return float(d @ ci @ d)

    dm, dl = d2(model), d2(lcdm)
    if dm > crit and dl <= crit:
        return "MODEL_DISFAVORED", dm, dl, crit
    if dl > crit and dm <= crit:
        return "LCDM_DISFAVORED", dm, dl, crit
    if dl > crit and dm > crit:
        return "BOTH_DISFAVORED", dm, dl, crit
    return "INCONCLUSIVE", dm, dl, crit


def algorithm_a3() -> None:
    header("Algorithm A3: w(a) trajectory and pre-registered decision rule")
    w0, wa = -0.752, -0.860  # targets taken from the manuscript; verify against the DESI DR2 release
    a_closed = 1.0 + (1.0 + w0) / wa
    a_root = brentq(lambda a: w_cpl(a, w0, wa) + 1.0, 0.3, 1.0, xtol=1e-14)
    record("phantom crossing: closed form = root solve",
           abs(a_closed - a_root) < 1e-10,
           f"a_c = {a_root:.6f}, z_c = {1 / a_root - 1:.4f}")

    # continuity: d ln rho / d ln a = -3 (1 + w) must hold for the closed form
    a = np.linspace(0.3, 1.0, 400)
    dlnrho = np.gradient(np.log(rho_cpl(a, w0, wa)), np.log(a))
    err = np.max(np.abs(dlnrho + 3.0 * (1.0 + w_cpl(a, w0, wa)))[2:-2])
    record("rho_DE(a) satisfies the continuity equation", err < 1e-3, f"max residual = {err:.1e}")

    # Decision-rule unit tests: the rule must be able to say each thing
    # NOTE: covariances below are placeholders for the test, NOT the DESI covariance.
    cov = np.diag([0.057**2, 0.22**2])
    cases = {
        "data at model target": ((-0.752, -0.860), "LCDM_DISFAVORED"),
        "data at LCDM": ((-1.0, 0.0), "MODEL_DISFAVORED"),
        "data midway": ((-0.88, -0.43), "INCONCLUSIVE"),
    }
    for nm, (x, expect) in cases.items():
        got, dm, dl, crit = decision_rule(x, cov)
        record(f"decision rule: {nm} -> {expect}", got == expect,
               f"got {got} (d2_model = {dm:.1f}, d2_LCDM = {dl:.1f}, crit = {crit:.1f})")
    open_item("Psi dark-energy dynamics not integrated",
              "w(a) above is the CPL form of the target, not a prediction. Crossing w = -1 is "
              "impossible for a single minimally coupled canonical scalar, so it requires the "
              "non-minimal xi R Psi^2 coupling; integrate the Gate-1 field equations with a "
              "specified V(Psi) to turn this into a test.")


# ==============================================================================
# A4: NEUTRINO DELAY
# ==============================================================================
KPC_M, C_MS = 3.0856775814913673e19, 299792458.0


def mass_delay_coeff(m_eV: float, D_kpc: float = 10.0) -> float:
    """dt = (D / 2c) (m c^2 / E)^2  ->  coefficient in microsecond * MeV^2."""
    return (D_kpc * KPC_M / (2.0 * C_MS)) * (m_eV * 1e-6) ** 2 * 1e6


def a4_design(E, E0=5.0, Emax=50.0, include_t0=False):
    L = np.log(E / E0) / np.log(Emax / E0)
    cols = [1.0 / E**2, np.ones_like(E), L]  # mass, achromatic offset, Psi-log
    if include_t0:  # Shapiro (achromatic) and unknown emission time t0 are the same column
        cols.insert(2, np.ones_like(E))
    return np.column_stack(cols)


def algorithm_a4(rng, quick: bool) -> None:
    header("Algorithm A4: neutrino delay decomposition (mass / achromatic / Psi-log)")
    Am = mass_delay_coeff(0.1)
    info("mass-term normalization", f"m = 0.1 eV at 10 kpc -> A_mass = {Am:.0f} us*MeV^2 "
                                    f"({Am / 100:.1f} us at 10 MeV). v1 used 50, i.e. 0.5 us at 10 MeV.")
    E = np.linspace(5.0, 50.0, 40)

    Jfull = a4_design(E, include_t0=True)
    rank = np.linalg.matrix_rank(Jfull)
    record("Shapiro constant and emission time t0 are degenerate (rank < columns)",
           rank < Jfull.shape[1], f"rank = {rank} for {Jfull.shape[1]} parameters. Only the "
                                  "offset (t0 + Shapiro) is identifiable from timing alone.")

    J = a4_design(E)
    sig, A_psi, offset = 1.5, 8.5, 3000.0
    ntr = 200 if quick else 600
    cov = sig**2 * np.linalg.inv(J.T @ J)
    zm, zp, est = [], [], []
    for _ in range(ntr):
        y = offset + Am / E**2 + A_psi * J[:, 2] + rng.normal(0, sig, E.size)
        beta = np.linalg.lstsq(J, y, rcond=None)[0]
        zm.append((beta[0] - Am) / np.sqrt(cov[0, 0]))
        zp.append((beta[2] - A_psi) / np.sqrt(cov[2, 2]))
        est.append(beta[[0, 2]])
    zm, zp = np.array(zm), np.array(zp)
    for nm, z in (("A_mass", zm), ("A_psi", zp)):
        record(f"injection-recovery {nm}", abs(z.mean()) < 4 / np.sqrt(ntr) and 0.9 < z.std() < 1.1,
               f"mean z = {z.mean():+.3f}, std z = {z.std():.3f}")
    corr = np.corrcoef(np.array(est).T)[0, 1]
    info("estimator correlation (A_mass, A_psi)", f"{corr:+.2f}")

    print("\n  Detectability of A_psi = 8.5 us (marginalized over mass term and offset).")
    print("  ASSUMPTIONS: events uniform in 5-50 MeV; sigma_t is the effective per-event timing")
    print("  scatter relative to the reference feature. Set sigma_t from your detector/emission model.")
    print(f"  {'sigma_t [us]':>14} {'N events':>9} {'sigma(A_psi) [us]':>18} {'3-sigma floor [us]':>19} {'8.5 us visible?':>16}")
    for st in (1.5, 100.0, 1000.0, 10000.0):
        for Nev in (40, 10000):
            Jn = a4_design(np.linspace(5.0, 50.0, Nev))
            s_psi = st * np.sqrt(np.linalg.inv(Jn.T @ Jn)[2, 2])
            print(f"  {st:14.1f} {Nev:9d} {s_psi:18.2f} {3 * s_psi:19.2f} {'yes' if A_psi > 3 * s_psi else 'no':>16}")
    open_item("realistic timing scatter",
              "Whether an 8.5 us log-energy delay is observable depends entirely on sigma_t. The "
              "table shows where it vanishes; justify sigma_t from the emission physics.")


# ==============================================================================
# GATE 3: GW250114 PIPELINE (noise-weighted)
# ==============================================================================
MSUN_S = 4.925491e-6  # G Msun / c^3


def aligo_like_psd(f):
    """Illustrative analytic aLIGO-like PSD (approximate fit). DATA HOOK: use the GWOSC PSD."""
    x = f / 245.4
    return 1e-48 * (0.0152 * x**-4 + 0.2935 * x**2.25 + 2.7951 * x**1.5 - 6.5080 * x**0.75 + 17.7622)


def uaf_blunted_waveform(frequency_array, mass_1, mass_2, luminosity_distance, theta_jn, phase,
                         gamma_i=0.4, width_frac=0.15, floor_val=0.5, f_merger=None,
                         f_cut=250.0, **kwargs):
    """Toy frequency-domain model: f^(-7/6) inspiral amplitude, leading-order TaylorF2 phase,
    smooth high-f taper, times the blunting modifier. Amplitude is arbitrary (normalize to a
    target SNR). In production call bilby.gw.source.lal_binary_black_hole instead."""
    if f_merger is None:
        raise ValueError("Must explicitly pass f_merger to uaf_blunted_waveform.")
    f = np.asarray(frequency_array, dtype=float)
    f_safe = np.where(f <= 0, 1e-10, f)
    Mc = (mass_1 * mass_2) ** 0.6 / (mass_1 + mass_2) ** 0.2 * MSUN_S
    amp = Mc ** (5.0 / 6.0) * f_safe ** (-7.0 / 6.0) / luminosity_distance / np.sqrt(1.0 + (f / f_cut) ** 4)
    psi = 2.0 * np.pi * f * 0.0 - phase - np.pi / 4 + (3.0 / 128.0) * (np.pi * Mc * f_safe) ** (-5.0 / 3.0)
    ph = np.exp(-1j * psi)
    pol = {"plus": amp * (1.0 + np.cos(theta_jn) ** 2) / 2.0 * ph,
           "cross": amp * np.cos(theta_jn) * 1j * ph}
    width = width_frac * f_merger
    mod = np.maximum(1.0 - gamma_i * np.exp(-((f - f_merger) ** 2) / (2.0 * width**2)), floor_val)
    return {k: v * mod for k, v in pol.items()}


def inner(a, b, psd, df):
    return 4.0 * df * np.real(np.sum(np.conj(a) * b / psd))


def gate3_gw250114(rng, quick: bool) -> None:
    header("Gate 3: GW250114 blunting test (noise-weighted, calibrated)")
    df = 0.25
    f = np.arange(20.0, 500.0 + df, df)
    psd = aligo_like_psd(f)
    record("PSD positive on 20-500 Hz", bool(np.all(psd > 0)))

    fm, kw = 150.0, dict(mass_1=33.6, mass_2=32.2, luminosity_distance=500.0, theta_jn=0.0, phase=0.0)
    h0 = uaf_blunted_waveform(f, gamma_i=0.0, f_merger=fm, **kw)["plus"]
    h0 = h0 * (80.0 / np.sqrt(inner(h0, h0, psd, df)))  # normalize GR signal to network SNR ~ 80
    rho = np.sqrt(inner(h0, h0, psd, df))
    width = 0.15 * fm
    g = np.exp(-((f - fm) ** 2) / (2.0 * width**2))  # shape of the dip
    u = h0 * g  # d h / d(-gamma)

    # Calibration 1: noise projected on a unit template must be N(0, 1)
    nsim = 800 if quick else 3000
    amp_n = np.sqrt(psd / (4.0 * df))
    hat = h0 / rho
    proj = np.empty(nsim)
    for i in range(nsim):
        n = amp_n * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size))
        proj[i] = inner(n, hat, psd, df)
    record("noise calibration: <n|h_hat> ~ N(0,1)",
           abs(proj.mean()) < 4 / np.sqrt(nsim) and 0.93 < proj.std() < 1.07,
           f"mean = {proj.mean():+.3f}, std = {proj.std():.3f}")

    # Mismatch scaling and exact-null checks
    def mismatch(gam):
        h1 = h0 * np.maximum(1.0 - gam * g, 0.5)
        ov = np.abs(inner(h0, h1, psd, df) + 1j * 0.0) / np.sqrt(inner(h0, h0, psd, df) * inner(h1, h1, psd, df))
        return 1.0 - ov

    record("mismatch(gamma = 0) = 0", abs(mismatch(0.0)) < 1e-12, f"{mismatch(0.0):.2e}")
    gs = np.array([0.02, 0.04, 0.08, 0.16])
    slope = np.polyfit(np.log(gs), np.log([mismatch(x) for x in gs]), 1)[0]
    record("mismatch ~ gamma^2 for small gamma", 1.8 < slope < 2.2, f"log-log slope = {slope:.3f}")

    mm = mismatch(0.4)
    info("mismatch at gamma = 0.4", f"{mm:.3e}")
    for D in (1, 2, 4):
        thr = np.sqrt(D / (2.0 * mm))
        info(f"distinguishability threshold, D = {D} model parameters",
             f"rho_thr = sqrt(D/(2*mismatch)) = {thr:.1f}  (rho = {rho:.0f}: {'distinguishable' if rho > thr else 'NOT'})")
    open_item("parameter projection",
              "A rigorous threshold projects out correlated GR parameters (chirp mass, q, spins). "
              "This uses the amplitude-only dip, which is nearly orthogonal to them, but it is not proven here.")

    # Calibration 2: linear gamma estimator, null and injection
    uu = inner(u, u, psd, df)
    sig_g = 1.0 / np.sqrt(uu)
    info("sigma_gamma at rho = 80", f"{sig_g:.4f}  (linear, fixed f_merger and width)")

    def gamma_hat(gam_true):
        out = np.empty(nsim)
        for i in range(nsim):
            n = amp_n * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size))
            d_minus_h0 = -gam_true * u + n
            out[i] = -inner(d_minus_h0, u, psd, df) / uu
        return out

    gh0 = gamma_hat(0.0)
    z = gh0 / sig_g
    nfp = int(np.sum(np.abs(z) > 3.0))
    p_fp = stats.binom.sf(nfp - 1, nsim, 2 * stats.norm.sf(3.0)) if nfp > 0 else 1.0
    record("null: gamma_hat / sigma ~ N(0,1)",
           abs(z.mean()) < 4 / np.sqrt(nsim) and 0.93 < z.std() < 1.07,
           f"mean = {z.mean():+.3f}, std = {z.std():.3f}")
    record("null: false-positive rate at 3 sigma matches 0.27%", p_fp > 1e-3,
           f"{nfp}/{nsim} (expected {nsim * 2 * stats.norm.sf(3.0):.1f}), binomial p = {p_fp:.3f}")
    gh1 = gamma_hat(0.4)
    record("injection gamma = 0.4 recovered", abs(gh1.mean() - 0.4) < 4 * sig_g / np.sqrt(nsim),
           f"mean gamma_hat = {gh1.mean():.4f} +/- {sig_g:.4f}; detection (|z|>3) = {np.mean(np.abs(gh1 / sig_g) > 3):.3f}")
    open_item("free f_merger and width",
              "Fitting those jointly makes the problem nonlinear: apply the A5 null-simulation "
              "look-elsewhere procedure over (f_merger, width).")


# ==============================================================================
# A5: CMB LOG-OSCILLATORY RIPPLE (null-calibrated)
# ==============================================================================
def algorithm_a5(rng, quick: bool) -> None:
    header("Algorithm A5: log-oscillatory ripple search (null-calibrated)")
    l = np.linspace(1500, 3000, 150)  # DATA HOOK: real bandpowers, with covariance
    C0 = 1000.0 * (l / 2200.0) ** (-1.2)
    sig_rel = 2.5 / C0
    w = 1.0 / sig_rel**2
    x = np.log(l / 2200.0)
    omegas = np.linspace(2.0, 30.0, 113)
    c, s = np.cos(np.outer(omegas, x)), np.sin(np.outer(omegas, x))
    Mcc, Mcs, Mss = (c * c * w).sum(1), (c * s * w).sum(1), (s * s * w).sum(1)
    det = Mcc * Mss - Mcs**2

    def stat(R):
        """Phase is fitted linearly (cos, sin amplitudes): T(omega) is chi2 with 2 dof under the null."""
        bc, bs = R @ (c * w).T, R @ (s * w).T
        T = (Mss * bc**2 - 2 * Mcs * bc * bs + Mcc * bs**2) / det
        return T, (Mss * bc - Mcs * bs) / det, (Mcc * bs - Mcs * bc) / det

    def noise(n):
        return rng.normal(size=(n, l.size)) * sig_rel

    def ripple(A, om, ph):
        return A * np.cos(om * x + ph)

    n1 = 600 if quick else 3000
    T1 = stat(noise(n1))[0].max(axis=1)
    T_crit = np.percentile(T1, 95)
    T2 = stat(noise(n1))[0].max(axis=1)
    fpr = np.mean(T2 > T_crit)
    se = np.sqrt(0.05 * 0.95 / n1)
    record("independent null: false-alarm rate = 5% at the calibrated threshold",
           abs(fpr - 0.05) < 3 * se, f"{fpr:.3f} +/- {se:.3f} (T_crit = {T_crit:.2f})")

    p_loc = np.exp(-T_crit / 2.0)
    n_eff = np.log(0.95) / np.log1p(-p_loc)
    info("effective independent trials", f"N_eff ~ {n_eff:.1f} (omega grid points = {omegas.size}; "
                                         f"v1 counted 1000 including the phase grid)")

    # Observed (synthetic) data: the v1 injection
    A_true, om_true, ph_true = 0.005, 12.5, 0.8
    r_obs = ripple(A_true, om_true, ph_true) + noise(1)[0]
    T_o, Ac, As = stat(r_obs[None, :])
    j = int(np.argmax(T_o[0]))
    p_glob = (1 + np.sum(T1 >= T_o[0, j])) / (n1 + 1)
    A_hat, ph_hat = float(np.hypot(Ac[0, j], As[0, j])), float(np.arctan2(-As[0, j], Ac[0, j]))
    sc, ss = np.sqrt(Mss[j] / det[j]), np.sqrt(Mcc[j] / det[j])
    Ac_t, As_t = A_true * np.cos(ph_true), -A_true * np.sin(ph_true)
    record("injection recovered (amplitude components within 4 sigma, omega near truth)",
           abs(Ac[0, j] - Ac_t) < 4 * sc and abs(As[0, j] - As_t) < 4 * ss and abs(omegas[j] - om_true) < 1.5,
           f"A_hat = {A_hat:.5f} (true {A_true}), omega_hat = {omegas[j]:.2f} (true {om_true}), phi_hat = {ph_hat:.2f}")
    info("global p-value of the observed maximum", f"p < {p_glob:.1e} (floor 1/(N+1) = {1 / (n1 + 1):.1e}; T_max = {T_o[0, j]:.0f})")

    # Detection power: smallest amplitude detected 95% of the time at the 5% global threshold
    amps = np.geomspace(2e-4, 3e-3, 10)
    nin = 200 if quick else 500
    power = []
    for A in amps:
        om = rng.uniform(2.0, 30.0, nin)
        ph = rng.uniform(0, 2 * np.pi, nin)
        R = noise(nin) + A * np.cos(om[:, None] * x[None, :] + ph[:, None])
        power.append(np.mean(stat(R)[0].max(axis=1) > T_crit))
    power = np.array(power)
    A95 = float(np.interp(0.95, power, amps)) if power.max() >= 0.95 else float("nan")
    record("power rises monotonically with amplitude", bool(np.all(np.diff(power) > -0.05)),
           ", ".join(f"{a_:.1e}:{p_:.2f}" for a_, p_ in zip(amps, power)))
    info("95%-power amplitude (replaces 'A_fit * 2.5')", f"A_95 ~ {A95:.2e} at 5% global false-alarm rate")
    open_item("real-data requirements",
              "Real ACT DR6 needs bandpower covariance, foreground marginalization, and the "
              "Lambda-CDM best fit refit jointly. The null simulations must use the real covariance.")


# ==============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer Monte Carlo trials")
    ap.add_argument("--seed", type=int, default=20261107)
    args = ap.parse_args()
    rg = lambda k: np.random.default_rng([args.seed, k])  # noqa: E731

    print("=" * 80)
    print(" THE AETHEL-PSI SYNTHESIS: v2 FALSIFIABLE VALIDATION SUITE")
    print("=" * 80)
    gate1_metric_variation()
    algorithm_a1()
    algorithm_a2(rg(2), args.quick)
    algorithm_a3()
    algorithm_a4(rg(4), args.quick)
    gate3_gw250114(rg(3), args.quick)
    algorithm_a5(rg(5), args.quick)

    header("Summary")
    counts = {k: sum(1 for _, s_, _ in RESULTS if s_ == k) for k in ("PASS", "FAIL", "OPEN", "INFO")}
    for nm, st, _ in RESULTS:
        if st in ("FAIL", "OPEN"):
            print(f"  [{st}] {nm}")
    print("\n  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print("  PASS means a check that could fail did not. OPEN items need new theory or real data.")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
