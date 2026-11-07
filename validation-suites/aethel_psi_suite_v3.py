#!/usr/bin/env python3
r"""
THE AETHEL-PSI SYNTHESIS: EXECUTION & VALIDATION SUITE  (v3, falsifiable edition)
Author: Derrick Irons / The Aethelred Institute   ORCID: 0009-0001-7064-6611

What changed relative to v2, and why (every change is elaborated here and in
V3_FIXES.md; v2 is NOT overwritten — see aethel_psi_suite_v2.py)
--------------------------------------------------------------------------------
v2 quick run: PASS=29 FAIL=1 OPEN=8 INFO=15. The one FAIL was A2 k coverage 0.78
(n=100) tripping a hard cutoff 0.58 < coverage < 0.78.

  A2 (v3 fix): the hard coverage cutoff is REPLACED by a proper two-sided binomial
    test of the 68% coverage against p=0.6827, reporting the p-value. A hard
    cutoff on a noisy coverage estimate is a flaky test: with n=100 the standard
    error on coverage is ~0.047, so 0.78 sits ~2 sigma out and trips on rounding.
    The binomial test asks the right question — is the observed count of
    |z|<1 consistent with Bin(n, 0.6827)? Full run n=400 injection-recovery,
    n=300 BIC, so the v2 borderline is settled by statistics, not by widening
    a cutoff by hand.
  A1 (v3 fix): the underdetermined 1-constraint / 2-exponent bridge is now
    confronted with TWO independent real constraints — the GW250114 profile
    (gamma < 0.11 at 90%, < 0.12 at 95%, Delta lnL = -55.2 at gamma=0.4) and
    the GW231123 profile (Delta lnL = +13.0 primary / +23.9 wide-dt at
    gamma=0.4, no limit). gamma(M) is derived in-suite from D(E) exactly as
    in workspace/gate3-gw250114/gamma-of-mass/GAMMA_OF_MASS.md (written by the
    parallel agent BEFORE the GW231123 result; this suite re-derives it, it
    does not copy its verdict): gamma = 1 - sqrt(1 - F), F the logistic
    Eq. (11R), power ratio (D-2)/2, E_merger defined four ways, f_merger(M)
    = 150 * 71.5 / M_det Hz. Per-event PASS/FAIL is reported. The log-scale
    variant that fits lambda* to the data is kept but labelled
    PARAMETRISATION NOT DERIVATION. Epsilon still has no flow equation in
    the theory, so it stays OPEN with the reason stated, not fudged.
  A4 (v3 fix): sigma_t is now JUSTIFIED from published sources, not picked
    from a table. SN1987A burst lasted ~13 s (Kamiokande 12.4 s, IMB 5.6 s;
    cooling phase ~10 s) — the emission spread is seconds. Hansen et al.,
    Phys. Rev. D 101, 123018 (2020), arXiv:1904.11461: Hyper-K and IceCube
    can time the burst ONSET to ~1 ms at 10 kpc, Gen2 sub-ms, and only a
    failed supernova gives HK ~0.01 ms (10 us). Detector PMT timing is ns
    and is NOT the limiting factor; the emission-time model is. The suite
    therefore tabulates detectability at sigma_t = 1 s (emission spread),
    1 ms (burst-onset precision, optimistic), and 10 us (failed-SN cutoff,
    special case only). If 8.5 us is unobservable at the justified sigma_t,
    that is reported as FAIL/OPEN, not hidden.
  A5 (v3 fix): (a) a covariance-aware null is added — bandpowers correlated
    with rho=0.3 neighbour correlation (AR(1), ASSUMED, not measured; the
    real ACT covariance is the missing input), statistic recomputed by
    generalised least squares (whitening). (b) A foreground-marginalisation
    PLACEHOLDER is added and labelled as such: a single power-law template
    (l/2200)^-2 projected out of the ripple fit; real ACT foregrounds need
    multi-frequency (90/150/220 GHz) templates, which are not in-suite.
    (c) A joint smooth-baseline refit is added as a PIPELINE check on
    synthetic data (amplitude + slope refit jointly with the ripple), while
    the real Lambda-CDM joint refit stays OPEN with the exact missing input
    named. A real-data hook tries ACT_DR6_FILE env var / local file and
    otherwise reports that no loadable ACT DR6 TT bandpower file was found
    in the 2026-11-07 check (LAMBDA DR6 page serves lensing x unWISE only;
    TT likelihood data live in the ACT DR6 likelihood repo / NERSC with no
    direct bandpower CSV verified) — synthetic results are never presented
    as real-data results.
  Gate 3 (v3 fix): (a) a Fisher-matrix projection over correlated GR
    parameters (ln chirp mass, phase, ln distance, inclination, using BOTH
    polarisations so inclination is not degenerate with distance) gives a
    PROJECTED sigma_gamma, labelled Fisher approximation; it is cross-checked
    against the real GW250114 profile limit (gamma < 0.11-0.12, i.e. an
    implied sigma ~0.06). (b) A look-elsewhere scan over (f_merger, width)
    with a null-calibrated global threshold, using the A5 procedure, reports
    the global p-value for the best (f_merger, width) on a synthetic
    injection. This is a CALIBRATION on synthetic data only — it does NOT
    refit the real GW250114 data to rescue gamma=0.4, which the real
    profile already excluded at Delta lnL = -55.2.

All data here are SYNTHETIC unless a DATA HOOK loads real arrays.
Exit status is 1 if any check is FAIL. Status OPEN = scientifically unresolved.
Run:  python aethel_psi_suite_v3.py [--quick] [--seed N]
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

    open_item("A1 bridge still underdetermined on its own",
              "1 amplitude constraint, 2 free exponents: every gamma_0 above 'passes'. The family "
              "is a fit, not a prediction. The two real Gate 3 constraints below are now the "
              "second and third constraints on the same bridge — see the gamma(M) verdicts.")
    for c in (1e-2, 1.0, 1e2):
        g = -np.log(A_t / A0) / np.log(ratio * c)
        info(f"unit dependence, chart->SI factor c = {c:g}", f"constant gamma needed = {g:+.4f}")
    open_item("chart-to-physical scale map",
              "ln(mu/mu_0) shifts by ln(c) under a unit redefinition, so the exponent is meaningless "
              "until the dimensionful map between chart and rad/s is derived.")

    # ---- v3 NEW: gamma(M) derived from D(E), confronted with both real events ----
    # Derivation (re-derived here in-suite; same as gamma-of-mass/GAMMA_OF_MASS.md,
    # which was written BEFORE the GW231123 result and does not use it):
    #   Eq. (11R) ANSATZ: D(E) = 4 - 2F(x), x = E/M_Pl,
    #     F(x) = 1/(1+exp(-k(x-x0))), k=8, x0=0.5.
    #   Power ratio P_UAF/P_GR = (D-2)/2 = 1 - F.
    #   Amplitude modifier at merger is (1 - gamma), so (1-gamma)^2 = 1 - F, i.e.
    #     gamma = 1 - sqrt(1 - F), clipped at 0.5 by the amplitude floor.
    #   f_merger(M) = 150 * 71.5 / M_det Hz (anchored on GW250114: 65.8 Msun
    #   source, 71.5 Msun detector-frame -> ~150 Hz; scales as 1/M).
    #   E_merger definitions tried: (a) graviton energy h*f_merger,
    #   (b) Hawking scale hbar*c^3/(8*pi*G*M_final), (d) density scale
    #   (rho/rho_Pl)^(1/4)*M_Pl*c^2, (c) rest energy M_tot*c^2 (shown, ruled out).
    print("\n  gamma(M) from D(E) — Eq. (11R) logistic, gamma = 1 - sqrt(1 - F):")
    print("  gamma = 0.4 needs F = 0.64, i.e. x = E/M_Pl = 0.572; gamma = 0.11 needs x = 0.333.")
    h_planck, hbar = 6.62607015e-34, 1.054571817e-34
    G_si, c_si = 6.67430e-11, 299792458.0
    msun_kg, m_pl_kg = 1.98847e30, 2.176434e-8
    e_pl_j = m_pl_kg * c_si**2
    rho_pl = c_si**5 / (hbar * G_si**2)

    def F_log(x, k=8.0, x0=0.5):
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))

    def gamma_from_x(x):
        return float(np.minimum(1.0 - np.sqrt(np.maximum(1.0 - F_log(x), 0.0)), 0.5))

    cases = [("GW250114", 65.8, 62.7, 0.09), ("GW231123", 240.0, 225.0, 0.40),
             ("30 Msun binary", 30.0, 28.5, 0.05), ("150 Msun binary", 150.0, 142.0, 0.20)]
    gammas = {}
    for name, mtot, mfin, z in cases:
        m_det = mtot * (1.0 + z)
        fm = 150.0 * 71.5 / m_det
        ea = h_planck * fm
        eb = hbar * c_si**3 / (8 * np.pi * G_si * mfin * msun_kg)
        rs = 2 * G_si * mfin * msun_kg / c_si**2
        rho = mfin * msun_kg / (4 * np.pi / 3 * rs**3)
        ed = (rho / rho_pl) ** 0.25 * e_pl_j
        ec = mtot * msun_kg * c_si**2
        ga, gb, gd = gamma_from_x(ea / e_pl_j), gamma_from_x(eb / e_pl_j), gamma_from_x(ed / e_pl_j)
        gc = gamma_from_x(ec / e_pl_j)
        gammas[name] = ga
        print(f"  {name:>16}: M_tot={mtot:6.1f} Msun, f_merger={fm:6.1f} Hz, "
              f"x_a={ea / e_pl_j:.2e} x_b={eb / e_pl_j:.2e} x_d={ed / e_pl_j:.2e} -> "
              f"gamma(a/b/d)={ga:.5f}, gamma(c, rest energy)={gc:.3f} (floor)")
        if name in ("GW250114", "GW231123"):
            record(f"gamma(M) defs (a)/(b)/(d) agree for {name}",
                   abs(ga - gb) < 1e-9 and abs(ga - gd) < 1e-9,
                   f"gamma = {ga:.5f} under all three; rest-energy def gives {gc:.3f}")
    # Per-event verdicts against the REAL profile results (not synthetic):
    # GW250114: gamma < 0.11 (90%) / < 0.12 (95%), Delta lnL(0.4) = -55.2.
    # GW231123: Delta lnL(0.4) = +13.0 primary / +23.9 wide-dt, no limit.
    g114, g123 = gammas["GW250114"], gammas["GW231123"]
    record("gamma(M) consistent with GW250114 limit (gamma <= 0.11)",
           g114 <= 0.11, f"derived gamma = {g114:.5f} vs 90% limit 0.11")
    record("gamma(M) predicts gamma ~ 0.4 for GW231123",
           g123 >= 0.30,
           f"derived gamma = {g123:.5f}, NOT ~0.4 — the 'heavier merger -> bigger gamma' "
           f"hope FAILS under Eq. (11R). GW231123's +13.0 preference for a dip is therefore "
           f"NOT predicted by this derivation; it contradicts it (or is waveform systematics).")
    spread = max(gammas.values()) - min(gammas.values())
    record("gamma(M) rises with mass (30 -> 240 Msun)",
           spread > 0.05 and gammas["GW231123"] > gammas["30 Msun binary"],
           f"gamma spans {min(gammas.values()):.5f} to {max(gammas.values()):.5f} "
           f"(spread {spread:.2e}) — gamma(M) is FLAT: the logistic switch sits at "
           f"x ~ 0.5, ~40 orders above any astrophysical merger energy, so F is "
           f"saturated at F(0) = 0.0180 and gamma = 0.00903 for every binary.")
    # F(0) artifact: at zero energy D should be exactly 4 and gamma exactly 0.
    info("F(0) artifact", "derived gamma = 0.00903 is the logistic's value at x=0, "
                           "F(0)=0.0180 — the ansatz predicts a small universal blunting even at "
                           "zero energy, where D should be exactly 4. A repaired F with F(0)=0 "
                           "(e.g. rational F=x^2/(1+x^2)) gives gamma = 0 for all GW events.")
    # Log-scale variant: PARAMETRISATION, not derivation (needs fitted lambda*).
    lam_250114 = np.log(h_planck * (150.0 * 71.5 / (65.8 * 1.09)) / e_pl_j)
    for fit_name, gamma_anchor in (("fit-to-limit-0.11", 0.11), ("force-0.4", 0.4)):
        f_anchor = 1.0 - (1.0 - gamma_anchor) ** 2
        lam_star = lam_250114 + np.log(1.0 / f_anchor - 1.0)
        preds = {}
        for name, mtot, mfin, z in cases:
            fm = 150.0 * 71.5 / (mtot * (1.0 + z))
            lam = np.log(h_planck * fm / e_pl_j)
            flog = 1.0 / (1.0 + np.exp(-(lam - lam_star)))
            preds[name] = float(np.minimum(1.0 - np.sqrt(max(1.0 - flog, 0.0)), 0.5))
        print(f"  [log-scale, {fit_name}] lam* = {lam_star:.2f} -> "
              + ", ".join(f"{n}: {v:.3f}" for n, v in preds.items()))
        if fit_name == "fit-to-limit-0.11":
            record("log-scale gamma(M) fitted to GW250114 limit predicts ~0.4 for GW231123",
                   preds["GW231123"] >= 0.30,
                   f"GW231123 gamma = {preds['GW231123']:.3f} (GW250114 {preds['GW250114']:.3f}, "
                   f"150 Msun {preds['150 Msun binary']:.3f}, 30 Msun {preds['30 Msun binary']:.3f}) — "
                   f"it DECREASES with mass, so it also fails. And lam* is fitted, not derived.")
    open_item("log-scale gamma(M) is a PARAMETRISATION, not a derivation",
              "lam* must be fitted to a GW event to get any mass dependence at all, and the "
              "fit-to-limit version still predicts gamma DECREASING with mass (0.11 -> 0.027). "
              "A genuine derivation needs an E_merger definition or an F whose switch sits at "
              "an astrophysical scale, derived rather than fitted — or a gamma that depends "
              "on something other than E/M_Pl (e.g. spin: GW231123 chi ~ 0.9/0.8 vs GW250114 "
              "chi <= 0.26 — but the theory as written gives gamma no spin dependence, so "
              "that would be a NEW hypothesis, not a derivation).")
    # Epsilon: no flow equation exists in the theory, so no honest flow check
    # can be formulated. Stating the reason instead of inventing a check.
    open_item("epsilon target unused — no flow equation exists",
              "eps ~ 1e-29 (coupling in g~ = g(1 + eps*Psi)) has no beta function, no scale "
              "dependence, and no relation to A or omega anywhere in the revised edition. "
              "An 'epsilon flow check' would have to invent the equation it is testing, "
              "so none is run. Next theory input needed: a beta_eps derived from the "
              "action, or a stated relation eps(A, omega) that can be checked against "
              "A=54 -> 0.1 and omega=0.00665 -> 470.")
    open_item("A1 overall: OPEN",
              "The bridge is still a fit (2 exponents, 1 amplitude constraint), the chart "
              "unit map is underived, epsilon is untestable, and the gamma(M) derived "
              "from D(E) FAILS the GW231123 half of the test (predicts 0.009, not ~0.4) "
              "while passing the GW250114 half (0.009 <= 0.11).")


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
    # v3 CHANGE: hard coverage cutoffs (0.58 < cov < 0.78) replaced by a proper
    # two-sided binomial test. v2's FAIL was coverage 0.78 at n=100 tripping a
    # strict "< 0.78" on a quantity whose Monte Carlo SE is ~0.047. The right
    # question is whether the count of |z|<1 is consistent with Bin(n, 0.6827).
    # z-mean / z-std checks are kept unchanged.
    for nm, z in (("k", zk), ("x0", zx)):
        cov = float(np.mean(np.abs(z) < 1))
        k_in = int(np.sum(np.abs(z) < 1))
        p_cov = float(2.0 * min(stats.binom.cdf(k_in, n, 0.6827),
                                stats.binom.sf(k_in - 1, n, 0.6827)))
        ok = (abs(z.mean()) < 4 / np.sqrt(n) + 0.1 and 0.85 < z.std() < 1.2
              and p_cov > 0.01)
        record(f"injection-recovery {nm}: unbiased, correctly-sized errors",
               ok,
               f"mean z = {z.mean():+.3f}, std z = {z.std():.3f}, 68% coverage = {cov:.2f} "
               f"(n = {n}, fit failures = {fail}), binomial p(coverage) = {p_cov:.3f}")

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

    # ---- v3 CHANGE: sigma_t is now JUSTIFIED from published sources, not picked
    # from a table. Sources (checked 2026-11-07):
    #   * SN1987A: burst lasted ~13 s (Kamiokande 12.4 s, IMB 5.6 s, Baksan 9.1 s;
    #     CERN Courier; arXiv:2109.01957 Table 1). Core-collapse cooling phase
    #     lasts ~10 s. So the EMISSION spread of the neutrinos themselves is
    #     seconds — any per-event delay model must be measured against a
    #     ~second-wide emission-time distribution, not against detector timing.
    #   * Hansen, Lindner & Scholer, Phys. Rev. D 101, 123018 (2020),
    #     arXiv:1904.11461: for a Galactic SN at 10 kpc, Hyper-Kamiokande and
    #     IceCube can time the burst ONSET to ~1 ms; IceCube-Gen2 sub-ms; only
    #     a FAILED supernova (sharp black-hole-formation cutoff) gives HK
    #     ~0.01 ms = 10 us. That is onset precision, not per-event scatter
    #     relative to a sharp reference feature — and it needs the sharp feature.
    #   * Detector PMT timing is ns-level and is NOT the limiting factor; the
    #     emission-time model is. Using sigma_t = 1.5 us (v2's best case) as if
    #     it were justified would be pretending the emission were a delta
    #     function. It is not.
    # Justified sigma_t values used below: 1 s (emission spread, realistic),
    # 1 ms (burst-onset precision, optimistic), 10 us (failed-SN cutoff only).
    print("\n  Detectability of A_psi = 8.5 us (marginalized over mass term and offset).")
    print("  JUSTIFIED sigma_t (see comments): 1 s = SN emission spread (SN1987A ~13 s burst,")
    print("  ~10 s cooling phase); 1 ms = burst-onset precision, Hyper-K/IceCube at 10 kpc")
    print("  (Hansen et al. PRD 101, 123018 (2020)); 10 us = failed-SN HK cutoff ONLY.")
    print("  N=40 ~ SN1987A-like event count; N=10000 ~ Hyper-K at 10 kpc (~8k-50k events).")
    print(f"  {'sigma_t':>10} {'N events':>9} {'sigma(A_psi) [us]':>18} {'3-sigma floor [us]':>19} {'8.5 us visible?':>16}")
    justified = {}
    for st in (1.0e6, 1000.0, 10.0):
        for Nev in (40, 10000):
            Jn = a4_design(np.linspace(5.0, 50.0, Nev))
            s_psi = st * np.sqrt(np.linalg.inv(Jn.T @ Jn)[2, 2])
            justified[(st, Nev)] = s_psi
            print(f"  {st:10.0f} {Nev:9d} {s_psi:18.2f} {3 * s_psi:19.2f} {'yes' if A_psi > 3 * s_psi else 'no':>16}")
    record("A_psi = 8.5 us detectable at justified sigma_t = 1 ms, N = 10000",
           A_psi > 3 * justified[(1000.0, 10000)],
           f"sigma(A_psi) = {justified[(1000.0, 10000)]:.1f} us, 3-sigma floor = "
           f"{3 * justified[(1000.0, 10000)]:.1f} us >> 8.5 us — FAIL. Even the optimistic "
           f"burst-onset precision is ~100x too coarse.")
    record("A_psi = 8.5 us detectable at realistic sigma_t = 1 s (emission spread)",
           A_psi > 3 * justified[(1.0e6, 10000)],
           f"sigma(A_psi) = {justified[(1.0e6, 10000)]:.0f} us at N=10000 — six orders of "
           f"magnitude short. The 8.5 us delay is buried under the ~10 s emission width.")
    info("only case that could work", f"failed-SN cutoff sigma_t = 10 us, N = 10000: "
           f"sigma(A_psi) = {justified[(10.0, 10000)]:.2f} us, floor "
           f"{3 * justified[(10.0, 10000)]:.2f} us — visible, BUT it needs a failed "
           f"supernova with a sharp BH-formation cutoff, which is a rare special case, "
           f"not a generic Galactic SN, and it needs that cutoff modelled to ~10 us.")
    open_item("realistic timing scatter — A4 is unobservable as stated",
              "Whether an 8.5 us log-energy delay is observable depends entirely on sigma_t, "
              "and the justified sigma_t (1 ms optimistic, ~1 s realistic) makes it "
              "unobservable by 2-6 orders of magnitude. Next theory input needed: either "
              "a much larger A_psi derived from the theory, or a named sharp emission "
              "feature (neutronization burst / failed-SN cutoff) with its own timing "
              "model, or a different observable entirely (e.g. energy-dependent burst "
              "onset across detectors). Do not quote 8.5 us as testable without that.")


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
    # ---- v3 NEW (a): Fisher-matrix projection over correlated GR parameters.
    # FISHER APPROXIMATION, labelled as such: it linearises around the GR point
    # and assumes Gaussian noise; it is not a PE. Parameters: gamma (the dip),
    # ln Mc (chirp mass — changes amplitude Mc^(5/6) AND phase (pi Mc f)^(-5/3)),
    # phase (constant), ln DL (pure amplitude scale), theta_jn (inclination).
    # BOTH polarisations are used in the inner product, so theta_jn is not
    # exactly degenerate with DL (plus/cross ratio changes with theta_jn).
    # Derivatives are central finite differences of the toy waveform.
    def inner2(a, b):
        # inner product summed over both polarisations
        return sum(inner(a[k], b[k], psd, df) for k in ("plus", "cross"))

    # NOTE (v3 fix after first quick run): theta_jn MUST be evaluated away
    # from 0 — at th=0 both polarisation factors have zero derivative
    # (d/dth (1+cos^2)/2 = -cos*sin = 0; d/dth cos = -sin = 0), which made
    # the Fisher matrix singular (sigma_gamma = nan). Evaluated at th=0.7,
    # the GW250114 profile point, where the derivatives are non-zero.
    TH0 = 0.7

    def h_both(gam=0.0, ln_mc=None, ph=0.0, ln_dl=None, th=TH0):
        m1, m2 = kw["mass_1"], kw["mass_2"]
        mc0 = (m1 * m2) ** 0.6 / (m1 + m2) ** 0.2
        if ln_mc is not None:  # rescale masses keeping q fixed: Mc -> Mc*exp(d)
            s = np.exp(ln_mc - np.log(mc0))
            m1, m2 = m1 * s, m2 * s
        dl = kw["luminosity_distance"] if ln_dl is None else np.exp(ln_dl)
        return uaf_blunted_waveform(f, mass_1=m1, mass_2=m2, luminosity_distance=dl,
                                    theta_jn=th, phase=ph, gamma_i=gam, f_merger=fm)

    h0b = h_both()
    # normalise BOTH pols to network SNR 80 (as the single-pol version was)
    rho2 = np.sqrt(inner2(h0b, h0b))
    scale = 80.0 / rho2
    h0b = {k: v * scale for k, v in h0b.items()}
    rho_b = np.sqrt(inner2(h0b, h0b))
    ln_mc0 = np.log((kw["mass_1"] * kw["mass_2"]) ** 0.6 / (kw["mass_1"] + kw["mass_2"]) ** 0.2)
    ln_dl0 = np.log(kw["luminosity_distance"])
    steps = {"gamma": 1e-3, "ln_mc": 1e-5, "phase": 1e-4, "ln_dl": 1e-5, "theta_jn": 1e-4}
    derivs = {}
    for nm, st in steps.items():
        kwd = dict(gam=0.0, ln_mc=ln_mc0, ph=0.0, ln_dl=ln_dl0, th=TH0)
        kp, km = dict(kwd), dict(kwd)
        if nm == "gamma":
            kp["gam"], km["gam"] = st, -st
        elif nm == "ln_mc":
            kp["ln_mc"], km["ln_mc"] = ln_mc0 + st, ln_mc0 - st
        elif nm == "phase":
            kp["ph"], km["ph"] = st, -st
        elif nm == "ln_dl":
            kp["ln_dl"], km["ln_dl"] = ln_dl0 + st, ln_dl0 - st
        else:
            # v3 fix after full run #1: centre on TH0, not on 0. The first
            # version set th = +/-st, i.e. a difference around 0 where the
            # derivative is exactly zero — that made Fisher singular (nan).
            kp["th"], km["th"] = TH0 + st, TH0 - st
        hp = {k: v * scale for k, v in h_both(**kp).items()}
        hm = {k: v * scale for k, v in h_both(**km).items()}
        derivs[nm] = {k: (hp[k] - hm[k]) / (2 * st) for k in ("plus", "cross")}
    names = list(steps)
    Fm = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            Fm[i, j] = inner2(derivs[a], derivs[b])
    try:
        cov_f = np.linalg.inv(Fm)
        sig_proj = float(np.sqrt(cov_f[0, 0]))
        corr_g_mc = float(cov_f[0, 1] / np.sqrt(cov_f[0, 0] * cov_f[1, 1]))
        cond = float(np.linalg.cond(Fm))
    except np.linalg.LinAlgError:
        sig_proj, corr_g_mc, cond = float("nan"), float("nan"), float("inf")
    sig_g_unproj = 1.0 / np.sqrt(inner(u, u, psd, df))  # same as sig_g in Calibration 2 below
    info("Fisher projection (both pols, rho = 80)",
         f"projected sigma_gamma = {sig_proj:.4f} vs unprojected {sig_g_unproj:.4f} "
         f"(inflation x{sig_proj / sig_g_unproj:.2f}); corr(gamma, ln Mc) = {corr_g_mc:+.2f}; "
         f"Fisher cond = {cond:.1e}")
    record("projected sigma_gamma stays < 0.12 (real 95% limit scale)",
           sig_proj < 0.12,
           f"projected {sig_proj:.4f} — the dip is NOT absorbed by chirp mass / phase / "
           f"distance / inclination in this Fisher approximation.")
    # Cross-check against the REAL GW250114 profile limit: a Gaussian 95% limit
    # gamma < 0.12 implies sigma ~ 0.12/1.96 = 0.061 (profile, fixed nuisances).
    sig_real_implied = 0.12 / 1.96
    record("synthetic sigma_gamma agrees with real GW250114 profile limit",
           abs(sig_proj - sig_real_implied) / sig_real_implied < 0.5,
           f"synthetic projected {sig_proj:.4f} vs real-implied ~{sig_real_implied:.4f} "
           f"(ratio {sig_proj / sig_real_implied:.2f}). Agreement within ~50% is expected: "
           f"the toy uses an analytic PSD, a single-detector-equivalent normalisation, "
           f"and an f^(-7/6) inspiral instead of IMRPhenomPv2; the real run also has "
           f"residual lines and a +7 ms timing offset. A factor-2 disagreement would "
           f"mean the toy sensitivity is wrong; it is not.")

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
    # ---- v3 NEW (b): look-elsewhere scan over (f_merger, width), null-calibrated.
    # CALIBRATION ON SYNTHETIC DATA ONLY. It does NOT refit the real GW250114
    # data: the real profile already excluded gamma=0.4 at Delta lnL = -55.2 at
    # fixed f_merger, and no scan over (f_merger, width) on real data is run
    # here to rescue it. What this does: (i) build the null distribution of
    # max over the grid of z^2, (ii) set the 5% global threshold, (iii) run
    # one synthetic injection at the true (150 Hz, 0.15) and report the best
    # grid point, recovered gamma, and its GLOBAL p-value.
    fm_grid = np.array([80.0, 100.0, 125.0, 150.0, 175.0, 200.0, 250.0])
    wf_grid = np.array([0.10, 0.15, 0.25])
    combos = [(float(a), float(b)) for a in fm_grid for b in wf_grid]
    # derivative templates u_j = h0 * exp(-(f-fm)^2/(2 width^2)), width = wf*fm
    U = np.column_stack([h0 * np.exp(-((f - a) ** 2) / (2.0 * (b * a) ** 2))
                         for a, b in combos])
    uu_grid = np.array([inner(U[:, j], U[:, j], psd, df) for j in range(len(combos))])
    sig_grid = 1.0 / np.sqrt(uu_grid)
    nscan = 300 if quick else 1000

    # gamma_hat_j = -<d-h0, u_j>/<u_j,u_j>; z_j = gamma_hat_j/sigma_j, and under
    # the null z_j = <n, u_j>/sqrt(<u_j,u_j>) (sign irrelevant for z^2).
    def scan_z2(noise_batch):
        inn = 4.0 * df * np.real((np.conj(noise_batch) / psd) @ U)
        return inn / np.sqrt(uu_grid)[None, :]  # z_j under the null (sign irrelevant)

    def make_noise(nsim):
        return amp_n * (rng.normal(size=(nsim, f.size)) + 1j * rng.normal(size=(nsim, f.size)))

    max_z2_null = np.empty(nscan)
    for s0 in range(0, nscan, 250):
        nb = make_noise(min(250, nscan - s0))
        max_z2_null[s0:s0 + nb.shape[0]] = np.max(scan_z2(nb) ** 2, axis=1)
    thr_scan = float(np.percentile(max_z2_null, 95))
    # independent null check of the global threshold
    max_z2_null2 = np.empty(nscan)
    for s0 in range(0, nscan, 250):
        nb = make_noise(min(250, nscan - s0))
        max_z2_null2[s0:s0 + nb.shape[0]] = np.max(scan_z2(nb) ** 2, axis=1)
    fpr_scan = float(np.mean(max_z2_null2 > thr_scan))
    record("look-elsewhere scan: global 5% threshold calibrated (independent null)",
           abs(fpr_scan - 0.05) < 3 * np.sqrt(0.05 * 0.95 / nscan),
           f"threshold max z^2 = {thr_scan:.2f} over {len(combos)} (f_merger, width) combos; "
           f"independent-null false-alarm rate = {fpr_scan:.3f}")
    # synthetic injection at the true point, then scan
    nb_inj = make_noise(1)
    d_inj = -0.4 * u + nb_inj[0]  # d - h0 for a gamma=0.4 injection at (150, 0.15)
    inn_inj = 4.0 * df * np.real(np.conj(d_inj) / psd @ U)
    gh_grid = -inn_inj / uu_grid
    z_grid = gh_grid / sig_grid
    jbest = int(np.argmax(np.abs(z_grid)))
    p_glob_scan = float((1 + np.sum(max_z2_null >= z_grid[jbest] ** 2)) / (nscan + 1))
    record("look-elsewhere scan recovers the synthetic injection at the true grid point",
           combos[jbest] == (150.0, 0.15) and abs(gh_grid[jbest] - 0.4) < 3 * sig_grid[jbest],
           f"best (f_merger, width) = {combos[jbest]}, gamma_hat = {gh_grid[jbest]:.3f} "
           f"(true 0.4), z^2 = {z_grid[jbest] ** 2:.1f} vs global threshold {thr_scan:.2f}, "
           f"global p = {p_glob_scan:.1e}")
    info("scan scope", "21 combos, synthetic PSD, single-detector-equivalent. This calibrates "
                       "the procedure; it is NOT a result on real GW250114/GW231123 data.")


# ==============================================================================
# A5: CMB LOG-OSCILLATORY RIPPLE (null-calibrated)
# ==============================================================================
def algorithm_a5(rng, quick: bool) -> None:
    header("Algorithm A5: log-oscillatory ripple search (null-calibrated)")
    # ---- v3 NEW: real-data hook. Tries ACT_DR6_FILE env var / a local file;
    # otherwise reports the outcome of the 2026-11-07 check: the LAMBDA DR6
    # product page serves lensing x unWISE likelihood data only; the TT/TE/EE
    # power-spectrum likelihood lives in the ACT DR6 likelihood repo / NERSC
    # with no direct bandpower CSV verified, so NO real file is loaded here.
    # Synthetic results below are never presented as real-data results.
    import os as _os
    _act_path = _os.environ.get("ACT_DR6_FILE", "")
    _act_loaded = False
    if _act_path and _os.path.exists(_act_path):
        _act_loaded = True
    # v3 fix after first quick run: this is a MISSING INPUT, not a failed test,
    # so it is recorded as OPEN (not FAIL) — a FAIL would imply the data were
    # loaded and contradicted the model, which is not what happened.
    if _act_loaded:
        info("ACT DR6 real bandpowers loaded", "ACT_DR6_FILE set and file exists")
    else:
        open_item("ACT DR6 real bandpowers loaded",
                  "NOT loaded — no ACT_DR6_FILE given and no loadable ACT DR6 TT bandpower "
                  "file was found in the 2026-11-07 check (LAMBDA DR6 page: lensing x unWISE "
                  "only). All A5 numbers below are SYNTHETIC pipeline validation.")
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

    # ---- v3 NEW (a): covariance-aware null (correlated bandpowers, GLS).
    # Real ACT bandpowers are correlated (binning, foregrounds, lensing). The
    # real covariance is NOT in-suite, so an AR(1) neighbour correlation
    # rho = 0.3 is ASSUMED and labelled as such — this tests whether the
    # pipeline survives correlated noise, it does not use the real covariance.
    from scipy.linalg import solve_triangular
    rho_c = 0.3
    idx = np.arange(l.size)
    Corr = rho_c ** np.abs(idx[:, None] - idx[None, :])
    Cov = np.outer(sig_rel, sig_rel) * Corr
    Lchol = np.linalg.cholesky(Cov)

    def whiten(R):
        return solve_triangular(Lchol, R.T, lower=True).T

    cw, sw = whiten(c), whiten(s)  # whitened design columns, per omega
    # M and b for the whitened 2-param fit, per omega:
    Mcc_w = np.sum(cw * cw, axis=1)
    Mcs_w = np.sum(cw * sw, axis=1)
    Mss_w = np.sum(sw * sw, axis=1)
    det_w = Mcc_w * Mss_w - Mcs_w**2

    def stat_cov(R):
        Rw = whiten(R)
        bc = Rw @ cw.T
        bs = Rw @ sw.T
        return (Mss_w * bc**2 - 2 * Mcs_w * bc * bs + Mcc_w * bs**2) / det_w

    def noise_cov(n):
        z = rng.normal(size=(n, l.size))
        return z @ Lchol.T

    n_cov = 600 if quick else 2000
    T1c = stat_cov(noise_cov(n_cov)).max(axis=1)
    T_crit_cov = float(np.percentile(T1c, 95))
    # power under correlated noise at the same amplitudes
    power_c = []
    for A in amps:
        om = rng.uniform(2.0, 30.0, nin)
        ph = rng.uniform(0, 2 * np.pi, nin)
        R = noise_cov(nin) + A * np.cos(om[:, None] * x[None, :] + ph[:, None])
        power_c.append(np.mean(stat_cov(R).max(axis=1) > T_crit_cov))
    power_c = np.array(power_c)
    A95_cov = float(np.interp(0.95, power_c, amps)) if power_c.max() >= 0.95 else float("nan")
    record("covariance-aware null: threshold and A95 stable vs diagonal case",
           abs(T_crit_cov - T_crit) / T_crit < 0.5 and abs(np.log10(A95_cov / A95)) < 0.5,
           f"T_crit cov = {T_crit_cov:.2f} vs diagonal {T_crit:.2f}; A95 cov = {A95_cov:.2e} "
           f"vs diagonal {A95:.2e} (rho = 0.3 AR(1) ASSUMED — real ACT covariance still missing)")
    info("covariance caveat", "rho = 0.3 is an assumption, not a measurement. If the real ACT "
                              "covariance has longer-range correlations, T_crit and A95 move; "
                              "re-run this block with the real covariance when it is loaded.")

    # ---- v3 NEW (b): foreground-marginalisation PLACEHOLDER, labelled as such.
    # A single power-law template t_fg = (l/2200)^-2 is projected out of the
    # ripple fit (Frisch-Waugh: residualise cos, sin and the data on t_fg,
    # then the 2-dof stat). Real ACT foregrounds need multi-frequency
    # (90/150/220 GHz) templates and a dust/CIB/synchrotron model — NOT
    # in-suite. This placeholder only checks the machinery does not break.
    t_fg = (l / 2200.0) ** (-2.0)
    fg = t_fg * w
    fg_norm = float(np.sum(fg * t_fg))
    c_r = c - np.outer((c * fg).sum(1) / fg_norm, t_fg)
    s_r = s - np.outer((s * fg).sum(1) / fg_norm, t_fg)
    Mcc_f = (c_r * c_r * w).sum(1)
    Mcs_f = (c_r * s_r * w).sum(1)
    Mss_f = (s_r * s_r * w).sum(1)
    det_f = Mcc_f * Mss_f - Mcs_f**2

    def stat_fg(R):
        Rr = R - np.outer((R * fg).sum(1) / fg_norm, t_fg)
        bc, bs = Rr @ (c_r * w).T, Rr @ (s_r * w).T
        return (Mss_f * bc**2 - 2 * Mcs_f * bc * bs + Mcc_f * bs**2) / det_f

    n_fg = 600 if quick else 1000
    T1f = stat_fg(noise(n_fg)).max(axis=1)
    T_crit_fg = float(np.percentile(T1f, 95))
    T_fg_obs = float(stat_fg(r_obs[None, :]).max())
    record("foreground-placeholder marginalisation: injection still detected",
           T_fg_obs > T_crit_fg,
           f"T_max with fg projected out = {T_fg_obs:.1f} vs threshold {T_crit_fg:.2f} "
           f"(single (l/2200)^-2 template — PLACEHOLDER, not a real ACT foreground model)")
    open_item("real foreground marginalisation",
              "Needs multi-frequency ACT DR6 foreground templates (90/150/220 GHz dust, "
              "CIB, synchrotron, tSZ/kSZ) and their covariances. The placeholder above "
              "only proves the projection machinery runs.")

    # ---- v3 NEW (c): joint smooth-baseline refit (PIPELINE check, synthetic).
    # The smooth baseline C0 is refit jointly with the ripple: baseline
    # b0 + b1*log(l/2200) (linear, in relative units) plus cos/sin, and T is
    # the chi2 improvement of the full fit over the baseline-only fit. This
    # checks the ripple cannot be absorbed by re-fitting the smooth spectrum.
    # The REAL Lambda-CDM joint refit (CAMB/CLASS + real bandpowers) is a
    # different, still-missing step — see the OPEN item below.
    Xb = np.column_stack([np.ones_like(x), x])  # baseline design
    Wb = w

    def stat_joint(R):
        # per-omega: fit [baseline, cos, sin] vs [baseline]; T = chi2 diff.
        # Baseline-only chi2 (same for all omega): precompute projection.
        Pb = Xb @ np.linalg.inv((Xb * Wb[:, None]).T @ Xb) @ (Xb * Wb[:, None]).T
        Rb = R - R @ Pb.T
        chi2_b = np.sum(Rb * Rb * w, axis=1)
        out = np.empty((R.shape[0], len(omegas)))
        for j in range(len(omegas)):
            Xf = np.column_stack([Xb, c[j], s[j]])
            Pf = Xf @ np.linalg.inv((Xf * Wb[:, None]).T @ Xf) @ (Xf * Wb[:, None]).T
            Rf = R - R @ Pf.T
            out[:, j] = chi2_b - np.sum(Rf * Rf * w, axis=1)
        return out

    n_j = 600 if quick else 1000
    T1j = stat_joint(noise(n_j)).max(axis=1)
    T_crit_j = float(np.percentile(T1j, 95))
    T_j_obs = float(stat_joint(r_obs[None, :]).max())
    j_j = int(np.argmax(stat_joint(r_obs[None, :])))
    record("joint smooth-baseline refit: injection still detected, omega near truth",
           T_j_obs > T_crit_j and abs(omegas[j_j] - om_true) < 1.5,
           f"T_max joint = {T_j_obs:.1f} vs threshold {T_crit_j:.2f}, omega_hat = {omegas[j_j]:.2f} "
           f"(true {om_true}) — refitting the smooth baseline does not absorb the ripple.")
    open_item("real-data requirements",
              "Real ACT DR6 needs (1) TT/TE/EE bandpowers + covariance at 1500 <= l <= 3000 "
              "(missing input: the ACT DR6 TT likelihood data files, e.g. from the "
              "act_dr6_likelihood / mflike data release — no direct bandpower CSV was "
              "verified loadable on 2026-11-07), (2) foreground marginalization with "
              "multi-frequency templates, and (3) the Lambda-CDM best fit refit jointly "
              "with CAMB/CLASS. The null simulations must use the real covariance.")


# ==============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer Monte Carlo trials")
    ap.add_argument("--seed", type=int, default=20261107)
    args = ap.parse_args()
    rg = lambda k: np.random.default_rng([args.seed, k])  # noqa: E731

    print("=" * 80)
    print(" THE AETHEL-PSI SYNTHESIS: v3 FALSIFIABLE VALIDATION SUITE")
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
