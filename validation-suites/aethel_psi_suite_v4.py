#!/usr/bin/env python3
r"""
THE AETHEL-PSI SYNTHESIS: EXECUTION & VALIDATION SUITE  (v4, claims-test edition)
Author: Derrick Irons / The Aethelred Institute   ORCID: 0009-0001-7064-6611

Purpose: test the NEW LaTeX claims pasted 2026-11-07 BEFORE they go in the paper.
Do NOT edit the manuscript. v2 and v3 are NOT overwritten.

Claims under test (B1-B6), each with a pre-registered PASS threshold written
in the code before the result is printed, and each able to FAIL:
  B1 Planck anchor: lambda=945 anchored to l_Pl gives ln(mu_chart/mu_phys)=88.46,
     beta_A = 1/14 ~= 0.0714, beta_eps ~= 0.755, A 54 -> ~0.1.
     Number vs derivation are scored SEPARATELY: a number can PASS while its
     derivation FAILs.
  B2 Phantom crossing: eta G ln(Psi/Psi0) in the action, crossing at a_c=0.476,
     z_c~=1.10, claimed as a topological phase transition, superseding the DESI
     CPL crossing a_c=0.7116, z_c=0.4052 (w0=-0.752, wa=-0.86).
  B3 Fisher: degradation factor 1.001 over [Mc, q, gamma, w_f] for an SNR~80
     GW250114-like event; gamma orthogonal to Mc/q.
  B4 "Five gates successfully closed" and "parameter-free".
  B5 ACT DR6 non-diagonal covariance ingested; upper limits on CMB ripples.
  B6 NCG/TQFT: Connes spectral action derives xi R Psi^2; Gilkey-de Witt heat
     kernel cancels Lambda as D flows 4->2; AdS2 x S2 / SYK at D=2 preserves
     unitarity / resolves the information paradox.

Baseline carried over from v3 (full run, default seed 20261107):
  v3: PASS=40 FAIL=5 OPEN=10. gamma(M)=0.00903 FLAT for 30-240 Msun;
  log-scale GW231123 gamma=0.027; A_psi=8.5 us unobservable at sigma_t=1 ms
  (3-sigma floor 218.5 us) and 1 s (218,504 us); Fisher (v3 set
  [gamma, ln Mc, phase, ln DL, theta_JN]) projected sigma_gamma=0.0533 vs
  unprojected 0.0494, inflation x1.079; ACT DR6 TT bandpowers NOT loaded.
Real GW data used as constraints (not synthetic):
  GW250114: gamma=0.4 excluded, Delta lnL=-55.2, gamma<0.11 (90%)/<0.12 (95%).
  GW231123: Delta lnL=+13.0 at gamma=0.4, no limit.
All other data here are SYNTHETIC unless a DATA HOOK loads real arrays.
Exit status is 1 if any check is FAIL. Status OPEN = scientifically unresolved.
Run:  python aethel_psi_suite_v4.py [--quick] [--seed N]
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy import stats
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

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
# Shared toy waveform / PSD (same as v3 Gate 3, copied so v4 is standalone)
# ==============================================================================
MSUN_S = 4.925491e-6


def aligo_like_psd(f):
    x = f / 245.4
    return 1e-48 * (0.0152 * x**-4 + 0.2935 * x**2.25 + 2.7951 * x**1.5 - 6.5080 * x**0.75 + 17.7622)


def uaf_blunted_waveform(frequency_array, mass_1, mass_2, luminosity_distance, theta_jn, phase,
                         gamma_i=0.4, width_frac=0.15, floor_val=0.5, f_merger=None,
                         f_cut=250.0, **kwargs):
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


def inner2(a, b, psd, df):
    return sum(inner(a[k], b[k], psd, df) for k in ("plus", "cross"))


# ==============================================================================
# B1: PLANCK ANCHOR + beta_A / beta_eps
# ==============================================================================
def test_b1() -> None:
    header("B1: Planck anchor, beta_A = 1/14, beta_eps ~= 0.755")
    # --- B1a: the NUMBERS. Pre-registered thresholds, set before computing:
    #   |ln - 88.46| < 0.01 ; |gamma_needed - 0.0711| < 0.001 ; |A_pred - 0.1|/0.1 < 5%.
    c_si, l_pl = 299792458.0, 1.616255e-35  # CODATA 2018 Planck length (m)
    lam_chart = 945.0
    omega_chart = 2.0 * np.pi * c_si / (lam_chart * l_pl)
    ln_ratio = float(np.log(omega_chart / 470.0))
    print(f"  lambda_phys = 945 * l_Pl = {lam_chart * l_pl:.3e} m; "
          f"omega_chart = 2*pi*c/(945*l_Pl) = {omega_chart:.3e} rad/s")
    record("B1a number: ln(mu_chart/mu_phys) = 88.46",
           abs(ln_ratio - 88.46) < 0.01,
           f"ln(omega_chart/470) = {ln_ratio:.2f} (claimed 88.46)")
    gamma_needed = -np.log(0.1 / 54.0) / ln_ratio
    record("B1a number: gamma needed over ln=88.46 is 0.0711",
           abs(gamma_needed - 0.0711) < 0.001,
           f"gamma_needed = {gamma_needed:.4f} (claimed 0.0711; over the old ln=11.166 "
           f"it was 0.5635 — the anchor, not the flow, is what changed the exponent)")
    beta_a = 1.0 / 14.0
    a_pred = 54.0 * np.exp(-beta_a * ln_ratio)
    err_pct = abs(a_pred - 0.1) / 0.1 * 100.0
    record("B1a number: beta_A=1/14 over ln=88.46 lands A ~ 0.1",
           err_pct < 5.0,
           f"A_pred = {a_pred:.4f} vs target 0.1 ({err_pct:.1f}% error); "
           f"1/14 = {beta_a:.6f} vs needed {gamma_needed:.4f}. "
           f"Note: with beta_A=1/14 over the OLD ln=11.166, A_pred = "
           f"{54.0 * np.exp(-beta_a * 11.166):.2f} — the number only works WITH the anchor.")
    info("B1a caveat", "The anchor lambda=945 -> 945*l_Pl is a CHOICE, not a derivation. "
                       "It redefines mu_chart; nothing in the theory picks 945*l_Pl over "
                       "945 in any other unit. The number passes; the choice stays an assumption.")

    # --- B1b: the DERIVATION of beta_A. A 4-form cannot be integrated over a 3-manifold.
    # Pontryagin class (standard convention, e.g. Nakahara / Gilkey):
    #   p1 = -1/(8 pi^2) Tr(F ^ F),  and  int_M p1  is an INTEGER (Pontryagin number).
    # Hence (1/4 pi^2) int Tr(F^F) = -2 * int p1  in  2*Z  — it can only be
    # ..., -4, -2, 0, 2, 4, ... ; it can NEVER be 1/14 ~= 0.0714.
    p1_int_values = [int(k) for k in (-2, -1, 0, 1, 2)]
    beta_from_p1 = [-2.0 * k for k in p1_int_values]
    print(f"  (1/4pi^2) int Tr(F^F) = -2 * int(p1); int(p1) in Z, so possible values "
          f"include {beta_from_p1}; 1/14 = {beta_a:.4f} is not among them.")
    record("B1b derivation: beta_A = (1/4pi^2) int_{S3} Tr(F^F) = 1/14",
           False,
           "FAIL on two independent grounds: (i) Tr(F^F) is a 4-form, S3 is "
           "3-dimensional, so int_{S3} of it is 0 / undefined — the integral as "
           "written does not exist; (ii) over a closed 4-manifold, "
           "(1/4pi^2) int Tr(F^F) = -2 * (Pontryagin number) in 2Z, never 1/14. "
           "A Chern-Simons 3-form on S3 could be non-zero, but its level is an "
           "integer and no bundle/connection/level is specified that gives 1/14.")
    open_item("B1b what would fix it",
              "Specify the bundle and connection on S3 x S1 (or the Chern-Simons level "
              "on S3), compute the integral explicitly, and show it equals 1/14 — or "
              "state beta_A = 1/14 as a fitted/assumed value. The numerical coincidence "
              "with gamma_needed (0.0714 vs 0.0711) is not a derivation.")

    # --- B1c: beta_eps. chi(S3 x S1) = chi(S3)*chi(S1) = 0*0 = 0. And c1 lives
    # in H^2(S3 x S1), which vanishes by Kuenneth (H^2(S3)=0, H^2(S1)=0,
    # H^1(S3)=0), so c1 = 0 exactly and c1^2 = 0, not 2.265. Even ignoring that,
    # c1^2 for a line bundle over a closed oriented 4-manifold is the
    # intersection pairing of an integral class with itself: an INTEGER.
    # (chi + c1^2)/3 = 0.755 would need c1^2 = 2.265, non-integer. FAIL either way.
    chi_s3, chi_s1 = 0, 0  # chi(S^n) = 1 + (-1)^n: S3 -> 0, S1 -> 0
    chi_prod = chi_s3 * chi_s1
    c1_squared_implied = 3.0 * 0.755 - chi_prod
    print(f"  chi(S3 x S1) = {chi_s3}*{chi_s1} = {chi_prod}; "
          f"beta_eps=0.755 needs c1^2 = {c1_squared_implied:.3f}")
    record("B1c derivation: beta_eps = (chi(S3xS1) + c1^2)/3 ~= 0.755",
           False,
           f"FAIL: chi(S3xS1) = {chi_prod}, and H^2(S3xS1) = 0 by Kuenneth, so "
           f"c1 = 0 and c1^2 = 0, giving beta_eps = 0, not 0.755. Independently, "
           f"the formula as written needs c1^2 = {c1_squared_implied:.3f}, which "
           f"is not an integer — but c1^2 is an intersection number of an integral "
           f"class, hence an integer. No bundle with c1^2 = 2.265 exists.")
    open_item("B1c beta_eps status",
              "beta_eps ~= 0.755 is FITTED, not derived, until a beta_eps is derived "
              "from the action or a stated eps(A, omega) relation is given — the same "
              "OPEN item as v3. The Planck anchor does not supply it: the anchor "
              "fixes the A-flow scale, it says nothing about eps.")


# ==============================================================================
# B2: PHANTOM CROSSING AT a_c = 0.476
# ==============================================================================
def w_cpl(a, w0, wa):
    return w0 + wa * (1.0 - a)


def test_b2() -> None:
    header("B2: phantom crossing at a_c = 0.476, z_c ~= 1.10 (Gauss-Bonnet)")
    # --- B2a: CPL consistency. DESI target in v2/v3: w0=-0.752, wa=-0.86.
    w0, wa = -0.752, -0.860
    a_cpl = brentq(lambda a: w_cpl(a, w0, wa) + 1.0, 0.3, 1.0, xtol=1e-14)
    wa_needed = -(1.0 + w0) / (1.0 - 0.476)
    record("B2a CPL crossing is a_c = 0.7116, z_c = 0.4052 (DESI target)",
           abs(a_cpl - 0.711628) < 1e-6,
           f"a_c = {a_cpl:.6f}, z_c = {1 / a_cpl - 1:.4f}")
    record("B2a claimed a_c = 0.476 is consistent with the DESI target (w0, wa)",
           abs(wa_needed - wa) < 0.05,
           f"FAIL: a_c=0.476 with w0=-0.752 needs wa = {wa_needed:.4f}, not the DESI "
           f"wa = {wa:.3f}; with DESI wa, w(0.476) = {w_cpl(0.476, w0, wa):.4f}, "
           f"already phantom — the crossing was earlier, at 0.7116. The 0.476 claim "
           f"either abandons the DESI target or comes from a different w(a) trajectory, "
           f"which is not stated.")
    # --- B2b: the integration as claimed. The LaTeX text gives NO V(Psi), NO xi,
    # NO eta, NO Psi0, NO initial conditions, NO equations of motion — only the
    # action and the claimed result. An integration cannot be reproduced from that.
    open_item("B2b GB integration is UNTESTABLE as stated",
              "Missing inputs, all required to reproduce a_c=0.476: V(Psi) (functional "
              "form + parameters), xi, eta, Psi0, initial Psi(t0) and Psi-dot(t0), and "
              "the equations of motion used. The text cites 'Algorithm A3, execution "
              "suite v3' — v3 A3 does NOT integrate Psi dynamics; it is OPEN in v3 for "
              "exactly that reason. No V that happens to give 0.476 was invented here.")
    # Illustrative scan (labelled as such, NOT a proof): fixed LambdaCDM background,
    # V = (1/2) m^2 Psi^2, m = 1 (H0 units), Psi(a=0.1)=0.5, Psi-dot=0, eta=0 first.
    # Canonical w = (Psi-dot^2/2 - V)/(Psi-dot^2/2 + V) in [-1, 1]; it touches -1
    # only instantaneously at turning points (Psi-dot=0). Report where that happens
    # as xi varies — if the "crossing" moves with xi/m/Psi0, it is tuned, not protected.
    Om, Ol = 0.3, 0.7

    def scan(xi, eta, m=1.0):
        def rhs(tau, y):
            a, Psi, Pd = y
            H = np.sqrt(Om / a**3 + Ol)
            Hd = -1.5 * Om / a**3
            R = 6.0 * (Hd + 2 * H**2)
            G = 24.0 * H**2 * (Hd + H**2)
            gb = eta * G / Psi if abs(Psi) > 1e-9 else eta * G * np.sign(Psi) * 1e9
            return [a * H, Pd, -3 * H * Pd - m * m * Psi - xi * R * Psi - gb]
        try:
            sol = solve_ivp(rhs, (0.0, 3.0), [0.1, 0.5, 0.0], rtol=1e-8, atol=1e-10,
                            max_step=0.05)
            a, Psi, Pd = sol.y
            w = (0.5 * Pd**2 - 0.5 * m * m * Psi**2) / (0.5 * Pd**2 + 0.5 * m * m * Psi**2 + 1e-300)
            imin = int(np.argmin(w))
            return dict(ok=bool(sol.success), nfev=int(sol.nfev), a_final=float(a[-1]),
                        min_psi=float(np.min(np.abs(Psi))), max_psi=float(np.max(np.abs(Psi))),
                        min_w=float(w[imin]), a_at_min_w=float(a[imin]))
        except Exception as e:  # noqa: BLE001 - integration failure IS the result
            return dict(ok=False, nfev=-1, a_final=float("nan"), min_psi=float("nan"),
                        max_psi=float("nan"), min_w=float("nan"), a_at_min_w=float("nan"),
                        err=str(e)[:80])

    print("  Illustrative scan (NOT the claimed proof): V=(1/2)m^2 Psi^2, m=1, Psi(0.1)=0.5, "
          "fixed LambdaCDM background, canonical w only:")
    for xi, eta in [(0.0, 0.0), (0.1, 0.0), (0.5, 0.0)]:
        r = scan(xi, eta)
        print(f"    xi={xi:.1f} eta={eta:.2f}: ok={r['ok']} nfev={r['nfev']} a_final={r['a_final']:.2f} "
              f"min|Psi|={r['min_psi']:.2e} min_w={r['min_w']:.3f} at a={r['a_at_min_w']:.3f}")
    r_eta = scan(0.1, 0.01)
    print(f"    xi=0.1 eta=0.01: ok={r_eta['ok']} nfev={r_eta['nfev']} "
          f"min|Psi|={r_eta['min_psi']:.2e} max|Psi|={r_eta['max_psi']:.2e}")
    info("B2b scan reading",
         "With eta=0 the canonical w touches -1 only at turning points, and the "
         "a of that turning point moves with xi (see table) — tuned, not protected. "
         "With eta != 0 the GB force is eta*G/Psi, which diverges as Psi -> 0; the "
         "xi=0.5, eta=0 run already drives min|Psi| to ~1e-5, so any eta != 0 term is "
         "~1e5 x eta x G there. No regulator is stated in the text.")
    # --- B2c: log singularity. Psi = A cos(Theta) crosses zero every half-cycle.
    th = np.linspace(0.0, 2 * np.pi, 2001)
    psi = np.cos(th)
    near_zero = np.min(np.abs(psi))
    dln = -np.tan(th)  # d/dTheta ln|cos Theta| = -tan(Theta), diverges at pi/2, 3pi/2
    print(f"  Psi = cos(Theta): min|Psi| on grid = {near_zero:.2e}; "
          f"max|d ln|Psi|/dTheta| on grid = {np.max(np.abs(dln)):.1e} (diverges at zeros)")
    record("B2c ln(Psi) is well-defined as written for Psi = A cos(Theta)",
           False,
           "FAIL as written: cos(Theta) crosses zero every half-cycle, ln(Psi) -> -inf "
           "there and ln of a negative Psi is not real. d ln|Psi|/dTheta = -tan(Theta) "
           "diverges at the zeros, so the GB force eta*G/Psi is singular at every "
           "zero-crossing. No regulator (|Psi|, Psi^2, a stated domain) is given in "
           "the text. ln(Psi^2) has the same 1/Psi force. This must be regulated "
           "before any 'protected phase transition' can be claimed.")
    open_item("B2c what would fix it",
              "State the regulator (e.g. ln((Psi^2 + delta^2)/Psi0^2) with a derived "
              "delta, or a Psi that never crosses zero with proof), give V, xi, eta, "
              "Psi0 and initial conditions, and publish the integrated w(a) with the "
              "crossing. Until then a_c=0.476 is an asserted number, not a result.")


# ==============================================================================
# B3: FISHER DEGRADATION 1.001 OVER [Mc, q, gamma, w_f]
# ==============================================================================
def test_b3() -> None:
    header("B3: Fisher degradation 1.001 over [Mc, q, gamma, w_f]")
    f = np.arange(20.0, 500.0 + 0.25, 0.25)
    psd = aligo_like_psd(f)
    df = 0.25

    def masses(Mc, q):
        M = Mc * q**(-0.6) * (1.0 + q) ** 1.2
        return M / (1.0 + q), M * q / (1.0 + q)

    Mc0, q0 = 28.6, 32.2 / 33.6  # GW250114 source chirp mass, mass ratio

    def h_1pn(gam=0.0, Mc=Mc0, q=q0, wf=0.15, th=0.7):
        # 1PN TaylorF2 phase, so q is identifiable at fixed Mc via eta in the
        # 1PN term (3715/756 + 55 eta/9) v^2. The v3 Newtonian toy has NO q
        # dependence at fixed Mc (amplitude and phase depend only on Mc), so
        # its Fisher matrix is singular in q — see the gamma=0 check below.
        m1, m2 = masses(Mc, q)
        M = m1 + m2
        eta = m1 * m2 / M**2
        Mc_s, M_s = Mc * MSUN_S, M * MSUN_S
        amp = Mc_s ** (5.0 / 6.0) * np.where(f <= 0, 1e-10, f) ** (-7.0 / 6.0) / 500.0 \
            / np.sqrt(1.0 + (f / 250.0) ** 4)
        v = (np.pi * M_s * f) ** (1.0 / 3.0)
        psi = -np.pi / 4 + (3.0 / 128.0) * (np.pi * Mc_s * f) ** (-5.0 / 3.0) \
            * (1.0 + (3715.0 / 756.0 + 55.0 * eta / 9.0) * v**2)
        ph = np.exp(-1j * psi)
        pol = {"plus": amp * (1.0 + np.cos(th) ** 2) / 2.0 * ph,
               "cross": amp * np.cos(th) * 1j * ph}
        mod = np.maximum(1.0 - gam * np.exp(-((f - 150.0) ** 2) / (2.0 * (wf * 150.0) ** 2)), 0.5)
        return {k: val * mod for k, val in pol.items()}

    # (i) At the GR point gamma=0, d h/d w_f = 0 identically (the dip vanishes
    # when gamma=0), and in the Newtonian toy d h/d q = 0 at fixed Mc. A Fisher
    # matrix over [Mc, q, gamma, w_f] evaluated at gamma=0 is therefore SINGULAR
    # — a degradation factor of 1.001 cannot have been computed that way.
    h0 = h_1pn(gam=0.0)
    scale0 = 80.0 / np.sqrt(inner2(h0, h0, psd, df))
    d_wf_0 = (h_1pn(gam=0.0, wf=0.151)["plus"] - h_1pn(gam=0.0, wf=0.149)["plus"]) / 0.002
    record("B3 Fisher over [Mc, q, gamma, w_f] is non-singular at the GR point gamma=0",
           False,
           f"FAIL as a computation: at gamma=0, |d h/d w_f| = {np.max(np.abs(d_wf_0)):.1e} "
           f"(exactly 0 — the dip vanishes with gamma), and in the Newtonian toy "
           f"d h/d q at fixed Mc is also 0. The claimed 1.001 therefore cannot be a "
           f"Fisher factor evaluated at the GR point; the evaluation point (and the "
           f"waveform's q dependence) are undisclosed in the text.")
    # (ii) Evaluate at the UAF point gamma=0.4, where w_f is identifiable.
    # Pre-registered PASS: inflation within 1.001 +/- 0.01 (0.991-1.011).
    h0 = h_1pn(gam=0.4)
    scale = 80.0 / np.sqrt(inner2(h0, h0, psd, df))
    steps = {"gamma": 1e-3, "Mc": 1e-3, "q": 1e-4, "wf": 1e-3}
    base = dict(gam=0.4, Mc=Mc0, q=q0, wf=0.15)
    keymap = {"gamma": "gam", "Mc": "Mc", "q": "q", "wf": "wf"}
    derivs = {}
    for nm, st in steps.items():
        kp, km = dict(base), dict(base)
        kp[keymap[nm]] = base[keymap[nm]] + st
        km[keymap[nm]] = base[keymap[nm]] - st
        hp = {k: v * scale for k, v in h_1pn(**kp).items()}
        hm = {k: v * scale for k, v in h_1pn(**km).items()}
        derivs[nm] = {k: (hp[k] - hm[k]) / (2 * st) for k in ("plus", "cross")}
    names = list(steps)
    Fm = np.zeros((4, 4))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            Fm[i, j] = inner2(derivs[a], derivs[b], psd, df)
    cond = float(np.linalg.cond(Fm))
    try:
        cov = np.linalg.inv(Fm)
        sig_proj = float(np.sqrt(cov[0, 0]))
        sig_un = 1.0 / float(np.sqrt(Fm[0, 0]))
        inflation = sig_proj / sig_un
        corr_mc = float(cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1]))
        corr_q = float(cov[0, 2] / np.sqrt(cov[0, 0] * cov[2, 2]))
        corr_wf = float(cov[0, 3] / np.sqrt(cov[0, 0] * cov[3, 3]))
    except np.linalg.LinAlgError:
        sig_proj = sig_un = inflation = corr_mc = corr_q = corr_wf = float("nan")
    info("B3 Fisher at gamma=0.4 (1PN toy, both pols, rho=80)",
         f"projected sigma_gamma = {sig_proj:.4f}, unprojected {sig_un:.4f}, "
         f"inflation = {inflation:.3f}, corr(gamma, Mc) = {corr_mc:+.3f}, "
         f"corr(gamma, q) = {corr_q:+.3f}, corr(gamma, w_f) = {corr_wf:+.3f}, cond = {cond:.1e}")
    record("B3 claimed degradation factor is 1.001",
           abs(inflation - 1.001) <= 0.01,
           f"FAIL: recomputed inflation = {inflation:.3f} at gamma=0.4, not 1.001. "
           f"The driver is corr(gamma, w_f) = {corr_wf:+.2f} — the dip depth and "
           f"width trade off — while corr(gamma, Mc) = {corr_mc:+.3f} and "
           f"corr(gamma, q) = {corr_q:+.3f} are indeed ~0. Orthogonality to Mc/q "
           f"in substance is TRUE; the number 1.001 is not. (v3, without w_f in "
           f"the set, got inflation x1.079 — same story: small, but not 1.001.)")
    record("B3 gamma is orthogonal to Mc and q in substance (|corr| < 0.05)",
           abs(corr_mc) < 0.05 and abs(corr_q) < 0.05,
           f"corr(gamma, Mc) = {corr_mc:+.3f}, corr(gamma, q) = {corr_q:+.3f}")


# ==============================================================================
# B4: "FIVE GATES CLOSED" AND "PARAMETER-FREE"
# ==============================================================================
def test_b4() -> None:
    header("B4: 'five gates closed' and 'parameter-free'")
    # v3 counts, read from the v3 output file on disk (not copied from memory):
    # if the file is absent, fall back to the recorded v3 numbers and say so.
    v3_counts = None
    try:
        with open("workspace/new-theory-tests/aethel_psi_suite_v3_output_full.txt") as fh:
            for line in fh:
                if line.strip().startswith("PASS="):
                    parts = dict(p.split("=") for p in line.strip().split())
                    v3_counts = {k: int(v) for k, v in parts.items()}
    except OSError:
        v3_counts = {"PASS": 40, "FAIL": 5, "OPEN": 10, "INFO": 20}
    print(f"  v3 full-run counts (from file): {v3_counts}")
    record("B4 'five gates successfully closed' — v3 has no FAIL/OPEN left",
           v3_counts["FAIL"] == 0 and v3_counts["OPEN"] == 0,
           f"FAIL: v3 full run is PASS={v3_counts['PASS']} FAIL={v3_counts['FAIL']} "
           f"OPEN={v3_counts['OPEN']}. The 5 FAILs are gamma(M) flat 0.00903 (not ~0.4 "
           f"for GW231123), gamma(M) not rising with mass, log-scale gamma(M) 0.027 for "
           f"GW231123, and A_psi=8.5 us unobservable at sigma_t=1 ms and 1 s.")
    # gamma(M) recomputed in-suite: the Planck anchor must NOT change it, because
    # gamma(M) is derived from D(E), not from the RG flow. Confirm flat 0.00903.
    h_planck = 6.62607015e-34
    e_pl_j = 2.176434e-8 * 299792458.0 ** 2

    def F_log(x, k=8.0, x0=0.5):
        return 1.0 / (1.0 + np.exp(-k * (x - x0)))

    def gamma_from_x(x):
        return float(np.minimum(1.0 - np.sqrt(np.maximum(1.0 - F_log(x), 0.0)), 0.5))

    gammas = {}
    for name, mtot, z in [("GW250114", 65.8, 0.09), ("GW231123", 240.0, 0.40),
                          ("30 Msun", 30.0, 0.05), ("150 Msun", 150.0, 0.20)]:
        fm = 150.0 * 71.5 / (mtot * (1.0 + z))
        gammas[name] = gamma_from_x(h_planck * fm / e_pl_j)
    print("  gamma(M) recomputed in v4: " + ", ".join(f"{k}={v:.5f}" for k, v in gammas.items()))
    record("B4 Planck anchor changes gamma(M) (it should not — different derivation)",
           len(set(round(v, 5) for v in gammas.values())) > 1 or gammas["GW231123"] >= 0.30,
           f"gamma(M) is still FLAT at {gammas['GW250114']:.5f} for 30-240 Msun — the "
           f"anchor does not rescue gamma(M), as expected (D(E), not RG flow). "
           f"The 'gates closed' claim therefore still fails on gamma(M) alone.")
    # Count free/unspecified parameters in the NEW action + spectral action.
    # Each entry is a parameter/function the text does not give a value or a
    # derivation for. (D(E) k, x0 are already ansatz parameters in v3.)
    free_params = [
        "xi (non-minimal coupling)", "eta (Gauss-Bonnet coupling)", "Psi0 (log reference)",
        "V(Psi) functional form + its parameters (>=2: mass, self-coupling)",
        "f in S=Tr(f(D_Psi/mu)) (spectral-action cutoff function)",
        "mu (spectral-action scale)", "D_Psi (Dirac operator incl. Psi coupling)",
        "D(E) logistic k, x0 (ansatz, v3)", "gamma(M) log-scale lambda* (fitted, v3)",
        "eps flow: beta_eps / eps(A, omega) (no equation, v3)",
        "chart unit anchor 945*l_Pl (a choice, B1a)",
    ]
    print(f"  Unspecified parameters/functions in the new framework ({len(free_params)}):")
    for p in free_params:
        print(f"    - {p}")
    record("B4 framework is parameter-free (0 unspecified parameters)",
           len(free_params) == 0,
           f"FAIL: {len(free_params)} parameters/functions remain unspecified (listed "
           f"above). Replacing a fitted exponent by a fitted topological integer, or "
           f"by an anchor choice, does not make the framework parameter-free while "
           f"xi, eta, Psi0, V, f, mu and D_Psi are all still free.")


# ==============================================================================
# B5: ACT DR6 COVARIANCE "INGESTED"
# ==============================================================================
def test_b5() -> None:
    header("B5: ACT DR6 non-diagonal covariance ingested")
    # Attempted sources, checked by the agent on 2026-11-07 (browser, not exec):
    #  * LAMBDA ACT DR6 derived-data pages:
    #    https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_lh_info.html
    #    https://lambda.gsfc.nasa.gov/product/act/actadv_dr6_lensing_xunwise_info.html
    #    -> lensing likelihood data and lensing x unWISE data ONLY (bandpowers,
    #       covariance, binning, fiducial spectra for LENSING, not TT/TE/EE).
    #  * ACT DR6 data products page: https://act.princeton.edu/act-dr6-data-products
    #    -> full release on LAMBDA / Globus / NERSC
    #      (/global/cfs/cdirs/cmb/data/act_dr6/dr6.02), chains on NERSC portal.
    #  * Full TT/TE/EE likelihood: https://github.com/ACTCollaboration/act_dr6_mflike
    #    (README read 2026-11-07): code is pip-installable, but the DATA are
    #    installed by `cobaya-install .../act_dr6_example.yaml`, which downloads
    #    CAMB etc. into a packages directory; the official data file is
    #    dr6_data.fits, run with cobaya + CosmoRec-aware CAMB. There is NO direct
    #    loadable TT bandpower CSV / covariance file served by LAMBDA or GitHub.
    #  * Env-var hook: if ACT_DR6_FILE points at a real file, load it.
    act_path = os.environ.get("ACT_DR6_FILE", "")
    loaded = bool(act_path) and os.path.exists(act_path)
    print("  Attempted: LAMBDA DR6 lensing pages (lensing + lensing x unWISE only); "
          "act.princeton.edu data products (LAMBDA/Globus/NERSC, no direct CSV); "
          "github.com/ACTCollaboration/act_dr6_mflike (data = dr6_data.fits via "
          "cobaya-install, needs cobaya + CAMB + CosmoRec).")
    if loaded:
        info("ACT DR6 real bandpowers loaded", f"ACT_DR6_FILE={act_path}")
    else:
        open_item("B5 ACT DR6 TT bandpowers + covariance loaded in-suite",
                  "NOT loaded — no ACT_DR6_FILE given and no loadable TT bandpower CSV / "
                  "covariance file was found in the 2026-11-07 check. The mflike route "
                  "needs cobaya-install (dr6_data.fits) + CAMB + CosmoRec, which is a "
                  "separate installation, not a file that can be ingested here. All A5 "
                  "numbers therefore remain SYNTHETIC pipeline validation, as in v3.")
    record("B5 abstract claim 'ingestion of the full non-diagonal ACT DR6 covariance "
           "matrix establishes upper limits' is supported by files checked",
           loaded,
           "FAIL as a manuscript claim: no ACT DR6 TT covariance was ingested in this "
           "suite or in v3, and no upper limit on CMB ripples was computed from real "
           "data. The v3 A5 numbers (A95 ~ 1.31e-3 diagonal, 1.77e-3 with assumed "
           "rho=0.3 AR(1) covariance) are synthetic. Do not print the abstract claim "
           "until dr6_data.fits is installed via cobaya and A5 is re-run on it.")


# ==============================================================================
# B6: NCG / TQFT — SPECTRAL ACTION, HEAT KERNEL, AdS2/SYK
# ==============================================================================
def test_b6() -> None:
    header("B6: NCG/TQFT — spectral action, heat-kernel Lambda cancellation, AdS2/SYK")
    try:
        import sympy as sp
    except ImportError:
        record("sympy available", False, "pip install sympy")
        return
    # --- B6a: Gilkey-de Witt coefficients for a Laplace-type operator
    # P = -Delta + E on a closed 4-manifold, acting on a bundle of rank r.
    # Standard (Gilkey 1995; Vassilevich review):
    #   a0 = (4pi)^-2 int tr(1)
    #   a2 = (4pi)^-2 int tr(R/6 - E)
    #   a4 = (4pi)^-2 (1/360) int tr(60 R E + 180 E^2 + 30 Omega^2
    #        + 12 Delta R + 5 R^2 - 2 R_mn R^mn + 2 R_mnrs R^mnrs)
    #   (sign conventions vary; the STRUCTURE — which monomials occur — is fixed.)
    # The spectral action is S = Tr f(D/mu) ~ sum_k f_k * a_k(D^2), with f_k the
    # moments of f. Verify the structure symbolically and read off what it can
    # and cannot fix.
    R, E, Om2, dR, R2, Ric2, Riem2, r = sp.symbols("R E Omega2 dR R2 Ric2 Riem2 r")
    a0 = r
    a2 = R / 6 - E
    a4 = (60 * R * E + 180 * E**2 + 30 * Om2 + 12 * dR + 5 * R2 - 2 * Ric2 + 2 * Riem2) / 360
    # If the scalar enters as E = -Psi^2 (the usual Yukawa/mass-type endomorphism,
    # the ONLY Psi dependence a Laplace-type D^2 can carry at this order), then:
    Psi = sp.symbols("Psi")
    a4_psi = sp.expand(a4.subs(E, -Psi**2))
    coeff_RPsi2 = sp.expand(a4_psi).coeff(R * Psi**2)
    a2_psi = sp.expand(a2.subs(E, -Psi**2))
    print(f"  a2 with E=-Psi^2: {a2_psi}  (Psi^2 mass term, coefficient fixed)")
    print(f"  a4 with E=-Psi^2: R*Psi^2 coefficient = {coeff_RPsi2} "
          f"(60*R*(-Psi^2)/360 = -R*Psi^2/6)")
    record("B6a heat-kernel coefficients a0/a2/a4 verified for P = -Delta + E",
           bool(a0 == r and sp.simplify(a2 - (R / 6 - E)) == 0
                and sp.simplify(a4 - (60 * R * E + 180 * E**2 + 30 * Om2 + 12 * dR
                                      + 5 * R2 - 2 * Ric2 + 2 * Riem2) / 360) == 0),
           "a0 ~ vol, a2 ~ R/6 - E, a4 as above (Gilkey). With E = -Psi^2, the "
           "a4 cross term gives R*Psi^2 with coefficient -1/6 — FIXED by the heat "
           "kernel, not a free xi. That is a real constraint IF D_Psi is specified "
           "so that E = -Psi^2.")
    # But D_Psi is NOT specified in the LaTeX text: no Dirac operator, no f, no mu,
    # no statement of how Psi enters D. Without D_Psi, E is unknown, so xi is not derived.
    open_item("B6a xi derived from the spectral action",
              "OPEN: D_Psi, f and mu are not specified in the text, so E is unknown and "
              "the coefficient of R*Psi^2 cannot be read off. With the illustrative "
              "choice E = -Psi^2 it would be -1/6, not a free xi — but that choice is "
              "ours, not the paper's. To derive xi, state D_Psi explicitly (how Psi "
              "enters: mass term? Yukawa? fluctuated Dirac operator?) and compute E.")
    # --- B6b: Lambda cancellation as D flows 4 -> 2.
    # a0 is (4pi)^-2 int tr(1) — a pure volume term, i.e. the cosmological constant
    # in the spectral action. It contains NO spectral-dimension factor, NO Psi
    # winding number, and NO D-dependence: the Gilkey coefficients are computed
    # at FIXED dimension d (here d=4). A "cancellation as D flows 4->2" would need
    # a d-dependent a0(d) or an extra topological term added by hand.
    has_winding = any(s in str(a0) + str(a2) + str(a4) for s in ("wind", "Psi"))
    # (a0/a2/a4 contain Psi only through E; there is no winding-number monomial.)
    record("B6b Lambda is cancelled by a term present in a0/a2/a4 as D flows 4->2",
           False,
           "FAIL as a derivation from the heat kernel: a0 ~ int tr(1) is the Lambda "
           "term and contains no D-dependence and no Psi winding number; a2/a4 are "
           "computed at fixed d=4 and contain Psi only through the endomorphism E. "
           "No monomial in a0/a2/a4 can cancel Lambda 'as D flows', because D does "
           "not appear in them. The cancellation would need an extra, separately "
           "specified winding term — which is not in the coefficients and not stated.")
    open_item("B6b what would fix it",
              "Give a d-dependent heat-kernel / spectral-action computation (a_k(d) "
              "for non-integer d, or a stated spectral-dimension regulator), state the "
              "winding-number term and its coefficient, and show the Lambda term "
              "cancelling term-by-term. As written it is an asserted cancellation.")
    # --- B6c: AdS2 x S2 / SYK at D=2.
    open_item("B6c AdS2/SYK preserves unitarity / resolves the information paradox",
              "OPEN: nothing is computed. Required, none present: the SYK Hamiltonian "
              "(N Majorana fermions, coupling J, q-body interaction), the large-N "
              "entropy S ~ N log 2 / 2 and its match to the horizon area, the Lyapunov "
              "exponent / chaos bound, and the map from the D=2 geometry to the SYK "
              "boundary (which AdS2 radius, which S2 radius, which dictionary). "
              "'Compactifies to AdS2 x S2' is a claim about the geometry, not a proof "
              "about unitarity. No entropy, no SYK coupling, no chaos exponent is given.")
    info("B6 scope",
         "This is a CHECK of what the heat kernel does and does not contain, not a "
         "proof attempt. The coefficients above are standard and verified; the "
         "cancellation and the SYK map are asserted without computed terms.")


# ==============================================================================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer Monte Carlo trials (v4 has none heavy)")
    ap.add_argument("--seed", type=int, default=20261107)
    args = ap.parse_args()
    print("=" * 80)
    print(" THE AETHEL-PSI SYNTHESIS: v4 CLAIMS-TEST VALIDATION SUITE")
    print(" Testing the 2026-11-07 LaTeX claims BEFORE they go in the paper")
    print("=" * 80)
    test_b1()
    test_b2()
    test_b3()
    test_b4()
    test_b5()
    test_b6()

    header("Summary")
    counts = {k: sum(1 for _, s_, _ in RESULTS if s_ == k) for k in ("PASS", "FAIL", "OPEN", "INFO")}
    for nm, st, _ in RESULTS:
        if st in ("FAIL", "OPEN"):
            print(f"  [{st}] {nm}")
    print("\n  " + "  ".join(f"{k}={v}" for k, v in counts.items()))
    print("  PASS means a check that could fail did not. OPEN items need new theory or real data.")
    print("  A claim can PASS on its number and FAIL on its derivation — both are reported.")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
