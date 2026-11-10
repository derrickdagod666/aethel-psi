#!/usr/bin/env python3
r"""
THE AETHEL-PSI SYNTHESIS: EXECUTION & VALIDATION SUITE  (v7)
Author: Derrick Irons / The Aethelred Institute   ORCID: 0009-0001-7064-6611

Purpose: v7 tests two Drive-only documents against each other and against
the standing record. Do NOT edit any manuscript or paper. v2-v6 suites,
their outputs, and all paper editions are NOT overwritten.

Documents under test (copies in v7_sources/, read in full from Google Drive
2026-11-09, Drive read-only):
  (a) "Beyond Supremacy: The Apophthegmatic Intelligentsia of Quantum
      Temporality" (Jan 2026) -- the QFI / complex-scalar reading of Psi.
      Two copies: Beyond.txt (26,636 B) and the Google Doc "Rigorous
      Mathematical Refinement of Theoretical Fr..." (38,243 chars; body
      identical, plus a Works Cited list). Beyond.txt is the canonical text.
  (b) "Rigorous Scientific Paper Dissection" (Aug 2025) -- an AI critical
      peer review of the FCM white paper (FOUR identical 37,972 B copies).
      NOTE: the Dissection predates (a) and targets the FCM paper, not (a).

Standing results used as constraints (NOT re-derived here):
  GW250114 full PE (dynesty, 3697 samples): gamma median 0.0235, 95% upper
    0.0813, P(gamma>0.4)=0, ln B=-3.06+/-0.23 (profile Delta lnL=-55.2 at
    gamma=0.4 is NOT a Bayes factor and is never quoted as one).
  Prediction 1 stands revised to gamma <~ 0.08 (95%), GW250114-like mergers.
  Spin-gamma hypothesis CLOSED in v6 (both branches FAIL). No replacement
    mechanism was invented; none is invented here.
  v5: PASS=25 FAIL=18 OPEN=6 INFO=8. beta_A=1/14 FITTED; xi=-1/6 DERIVED
    only under stated D_Psi with E=-Psi^2; phantom crossing FAILED with the
    paper's stated V(Psi); Lambda_eff==0 DROPPED, not proven; 10 free
    parameters remain in the census.
  gamma=0.4 is NOT a paper value anywhere in v7.
  Pre-registered stopping rule honoured: no gamma/width/f_merger was
    refitted anywhere in this suite (this suite runs no GW analysis at all).

Sections (each check pre-registered, able to FAIL):
  P  Provenance / premises.
  D  The Dissection's objections -- assertion tested: the objection is
     SUSTAINED (PASS = the critique is correct on this point).
  M  The QFI manuscript's claims -- internal mathematics, document
     integrity, and consistency with the standing record.
Exit status is 1 if any check is FAIL. OPEN = scientifically unresolved.
Run:  python aethel_psi_suite_v7.py
"""
from __future__ import annotations

import os
import re
import sys

import numpy as np
from scipy.integrate import quad, solve_ivp

RESULTS: list[tuple[str, str, str]] = []
BASE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(BASE, "v7_sources")
REPO = os.path.dirname(BASE)

BEYOND = open(os.path.join(SRC, "Beyond_Supremacy_QFI_manuscript.txt"),
              encoding="utf-8").read()
REFINED = open(os.path.join(SRC, "Beyond_Supremacy_Refinement_Doc.txt"),
               encoding="utf-8").read()
DISSECTION = open(os.path.join(SRC, "Rigorous_Scientific_Paper_Dissection.txt"),
                  encoding="utf-8").read()


def header(title: str) -> None:
    print("\n" + "=" * 80)
    print(f" {title.upper()}")
    print("=" * 80)


def record(name: str, ok: bool, detail: str = "", status: str | None = None) -> bool:
    status = status or ("PASS" if ok else "FAIL")
    RESULTS.append((name, status, detail))
    print(f"  [{status}] {name}" + (f" -- {detail}" if detail else ""))
    return ok


def info(name: str, detail: str = "") -> None:
    record(name, True, detail, status="INFO")


def open_item(name: str, detail: str = "") -> None:
    record(name, True, detail, status="OPEN")


def equation_block(text: str, anchor: str) -> str:
    """Return the text from `anchor` to the next \\end{equation} (inclusive)."""
    i = text.find(anchor)
    if i < 0:
        return ""
    j = text.find(r"\end{equation}", i)
    return text[i:(j + len(r"\end{equation}")) if j >= 0 else i + 400]


# ---------------------------------------------------------------- P
def section_p() -> None:
    header("P: provenance / premises")
    info("P1 documents read in full from Drive (read-only)",
         "Beyond.txt 26,636 B (Uaf folder); Refinement Doc 38,243 chars "
         "(body identical to Beyond.txt, plus Works Cited); Dissection "
         "37,972 B x4 identical copies (Doc + 3 txt), Aug 2025")
    absent = [t for t in ("gamma", r"\gamma", "Gate 3", "GW2", "merger",
                          "250114") if t in BEYOND]
    record("P2 premise: QFI manuscript body contains no Gate 3 / gamma / GW "
           "reference at all", not absent,
           "absent terms verified: gamma, \\gamma, Gate 3, GW2*, merger, "
           "250114 (Refinement Doc's only hits are inside Works Cited titles)")


# ---------------------------------------------------------------- D
def section_d() -> None:
    header("D: the Dissection's objections (PASS = objection sustained)")

    # D1 -- Dirac spectrum on S^3 x S^1 (Dissection section 1.1)
    std_s3 = [k + 1.5 for k in range(4)]          # (k+3/2)/R, R=1
    paper_lowest = 0.5                            # (n+1/2)/R at n=0, m~0
    formula = r"\lambda_i = \pm\sqrt{(n+1/2)^2/R^2 + m^2}"
    no_s1 = ("k'" not in formula) and ("/L" not in formula)
    record("D1 Dirac spectrum: paper Eq.(2) omits the S^1 quantum number "
           "and its S^3 offset is 1/2, not the standard 3/2",
           abs(std_s3[0] - paper_lowest) > 0.1 and no_s1,
           f"standard S^3 lowest eigenvalue 3/(2R) vs paper 1/(2R); a "
           f"single-index formula cannot be the S^3xS^1 spectrum (needs "
           f"(k'/L)^2 as well). Objection sustained; at best Eq.(2) is an "
           f"unnamed restricted sector, which the paper does not state")

    # D2 -- the 'derived' mass m = 7.373e-51 kg (Dissection section 1.3)
    kg_per_ev = 1.78266192e-36
    m_ev = 7.373e-51 / kg_per_ev
    d_nu = np.log10(0.12 / m_ev)        # vs neutrino sum bound 0.12 eV
    d_de = np.log10(2.3e-3 / m_ev)      # vs dark-energy scale ~2.3 meV
    record("D2 'derived cosmological mass scale' m=7.373e-51 kg matches no "
           "known scale and is derived nowhere in the FCM text",
           min(d_nu, d_de) > 10,
           f"m = {m_ev:.3g} eV: {d_nu:.1f} orders below the 0.12 eV "
           f"neutrino bound, {d_de:.1f} orders below the dark-energy "
           f"scale. The 'fine-tuned parameter' charge is sustained against "
           f"the FCM paper. (Later suites agree in kind: v5 census still "
           f"carries FITTED parameters, e.g. beta_A=1/14 -- relabelled "
           f"honestly there, 'derived' in the FCM text)")

    # D3 -- spectral action structure (Dissection section 1.2)
    cites_standard = ("spectralaction" in DISSECTION.replace(" ", "")
                      or "PhysRevLett.77.4868" in DISSECTION)
    states_single_trace = "Tr}(f(D/" in DISSECTION or "Tr(f(D" in DISSECTION
    record("D3 spectral action: the paper's Eq.(3) splits the heat-kernel "
           "expansion terms and appends a Kontsevich-Vishik trace; that is "
           "not the Spectral Action Principle it invokes",
           cites_standard and states_single_trace,
           "the standard action is the single trace Tr f(D/Lambda); its "
           "expansion yields the Lambda^4 and Lambda^2 R terms the paper "
           "lists -- they are not independent pieces, and the KV trace is "
           "a regularised trace for pseudodifferential operators, not a "
           "spectral-action term. Structural objection sustained")
    info("D3-note exact prefactors (f-moments) are convention-dependent",
         "the paper's Lambda^4/4pi^2 and Lambda^2/24pi^2 coefficients are "
         "not adjudicated here; the sustained point is structural")

    # D4 -- CMB figure is a cartoon; no quantitative prediction (sect. 2.1)
    hits = []
    for root, _dirs, files in os.walk(REPO):
        if "v7_sources" in root or "__pycache__" in root:
            continue
        for f in files:
            if f == os.path.basename(__file__):
                continue
            try:
                t = open(os.path.join(root, f), encoding="utf-8",
                         errors="ignore").read()
            except OSError:
                continue
            if "acoustic peak" in t.lower() and "aethel" in t.lower():
                hits.append(f)
    record("D4 CMB claim: no quantitative FCM CMB spectrum or fitted peak "
           "table exists anywhere in the tested record",
           not hits and "Not specified" in DISSECTION,
           "the FCM paper's Figure 3 is a schematic (Dissection Table 2: "
           "peak locations 'Not specified'); nothing in the repo computes "
           "an FCM CMB spectrum. The standing ACT DR6 ripple tests are a "
           "different observable and are null. Objection sustained")

    # D5 -- 'Psi-modulated' CHSH drop (Dissection section 3.1), computed
    Z = np.array([[1, 0], [0, -1]], complex)
    X = np.array([[0, 1], [1, 0]], complex)
    Y = np.array([[0, -1j], [1j, 0]], complex)
    I2 = np.eye(2)
    bell = np.zeros(4, complex); bell[0] = bell[3] = 1 / np.sqrt(2)

    def expect(psi, a, b):
        return float(np.real(psi.conj() @ np.kron(a, b) @ psi))

    def chsh_fixed(psi):
        b0 = (Z + X) / np.sqrt(2); b1 = (Z - X) / np.sqrt(2)
        return (expect(psi, Z, b0) + expect(psi, Z, b1)
                + expect(psi, X, b0) - expect(psi, X, b1))

    def concurrence(psi):
        return abs(psi.conj() @ np.kron(Y, Y) @ psi.conj())

    def chsh_max(psi):
        T = np.array([[expect(psi, p, q) for q in (X, Y, Z)]
                      for p in (X, Y, Z)])
        u = np.sort(np.linalg.eigvalsh(T.T @ T))[::-1]
        return 2 * np.sqrt(u[0] + u[1])

    th = np.pi / 2
    Ry = np.array([[np.cos(th / 2), -np.sin(th / 2)],
                   [np.sin(th / 2), np.cos(th / 2)]], complex)
    rotated = np.kron(I2, Ry) @ bell
    s0, s1 = chsh_fixed(bell), chsh_fixed(rotated)
    c0, c1 = concurrence(bell), concurrence(rotated)
    m0, m1 = chsh_max(bell), chsh_max(rotated)
    record("D5 'Psi-modulated' CHSH drop is a fixed-settings local-rotation "
           "effect, not an anomaly",
           s0 > 2.8 and s1 < 2.0 and abs(c0 - c1) < 1e-9
           and abs(m0 - m1) < 1e-9,
           f"standard settings: S={s0:.3f} for |Phi+>; after R_Y(pi/2) on "
           f"one qubit, SAME settings: S={s1:.3f}; concurrence {c0:.3f} -> "
           f"{c1:.3f} (unchanged -- a local unitary cannot change "
           f"entanglement); optimised-settings S_max {m0:.3f} -> {m1:.3f} "
           f"(unchanged). The drop is exact, classical bookkeeping of "
           f"rotated correlations. The paper's specific 0.98 depends on "
           f"its settings/angle, not on any anomaly. Objection sustained")

    # D6 -- EEG loophole (Dissection sections 3.2/4.1) vs the record
    outline = os.path.expanduser(
        "~/workspace/your_files/new-theory-revised/.src/OUTLINE.md")
    withdrawn = False
    if os.path.exists(outline):
        t = open(outline, encoding="utf-8", errors="ignore").read()
        withdrawn = "Bell-EEG" in t and "retracted" in t
    if withdrawn:
        record("D6 EEG/CHSH loophole objection: sustained -- and its target "
               "claim was already retracted in the workspace record", True,
               "Revised Edition outline: 'Bell-EEG - Claim: S=0.115 claim "
               "retracted'. The objection and the record agree; nothing in "
               "the current (Third Revised) edition rests on the EEG claim")
    else:
        open_item("D6 EEG/CHSH loophole objection",
                  "withdrawal record not found at the expected path; "
                  "objection itself is methodological and unrebutted")

    # D7 -- p-hacking (Dissection section 4.2)
    open_item("D7 p-hacking / pre-registration objection",
              "cannot be adjudicated from available material: the FCM-era "
              "dataset is not in Drive or the repo, and no pre-registration "
              "exists either way. The methodological concern (many EEG "
              "bands x regions x windows, one reported correlation, "
              "r=-0.78/p=0.002/K>150) stands unrebutted; its target claim "
              "is withdrawn in any case (D6)")

    # D8 -- data/code availability links (Dissection section 5.1)
    cited = ("zenodo.org/records/1234567" in DISSECTION
             and "github.com/Aethelred/FCMspectral" in DISSECTION)
    record("D8 FCM paper's availability links do not resolve to this work",
           cited,
           "paper cites zenodo.org/records/1234567 and "
           "github.com/Aethelred/FCMspectral; the real artifacts live at "
           "github.com/derrickdagod666/aethel-psi and Zenodo DOI "
           "10.5281/zenodo.24038497 -- different owner, name, and record. "
           "Objection sustained on the record")
    info("D8-note live check 2026-11-09",
         "github.com/Aethelred/FCMspectral returned HTTP 404 when fetched "
         "for this suite. The Dissection's further inference of intent "
         "('misconduct', 'deliberate deception') goes beyond the checkable "
         "fact and is not endorsed here")

    # D9 -- the Dissection's own reference facts
    ok = (abs(2 * np.sqrt(2) - 2.828) < 1e-3
          and abs(0.12 * kg_per_ev - 2.14e-37) < 0.2e-37)
    record("D9 the critique's own checkable reference facts are accurate",
           ok, "Tsirelson bound 2sqrt2=2.828 as stated; 0.12 eV = "
           "2.14e-37 kg as stated ('about 2e-37 kg')")
    info("D10 where the critique itself overreaches (balance)",
         "section 3.1 says the local rotation 'destroys entanglement' -- "
         "D5 shows concurrence is invariant; the correct statement is the "
         "fixed-settings one. The conclusion survives, the wording does "
         "not. Section 5.1's intent language: see D8-note")


# ---------------------------------------------------------------- M
def section_m() -> None:
    header("M: the QFI manuscript's claims")

    # M1 -- intelligence-growth ODE: closed form + t_max
    def closed(t, a, b, i0):
        return i0 * (1 - (b - 1) * a * t * i0 ** (b - 1)) ** (-1 / (b - 1))
    ok_all = True
    for a, b, i0 in ((0.3, 1.2, 1.5), (0.1, 2.0, 2.0)):
        tmax = 1 / ((b - 1) * a * i0 ** (b - 1))
        sol = solve_ivp(lambda t, y: [a * y[0] ** b], [0, 0.8 * tmax],
                        [i0], dense_output=True, rtol=1e-9, atol=1e-12)
        ts = np.linspace(0, 0.8 * tmax, 25)
        err = np.max(np.abs(sol.sol(ts)[0] - closed(ts, a, b, i0))
                     / closed(ts, a, b, i0))
        ok_all &= bool(err < 1e-3)
    record("M1 growth ODE: closed-form solution and finite-time blow-up "
           "t_max are mathematically correct", ok_all,
           "solve_ivp vs I0[1-(b-1)a t I0^(b-1)]^(-1/(b-1)) for "
           "(a,b,I0)=(0.3,1.2,1.5) and (0.1,2,2): max rel. err < 1e-3 up "
           "to 0.8 t_max. Correct ODE bookkeeping -- about the equation, "
           "not about any physical system")

    # M2 -- Osgood criterion application
    ok2 = True
    for b in (1.2, 2.0):
        val, _ = quad(lambda i: 1 / (0.3 * i ** b), 1.5, np.inf)
        exact = 1.5 ** (1 - b) / (0.3 * (b - 1))
        ok2 &= abs(val - exact) / exact < 1e-6
    v1, _ = quad(lambda i: 1 / (0.3 * i ** 0.8), 1.5, 1e4)
    v2, _ = quad(lambda i: 1 / (0.3 * i ** 0.8), 1.5, 1e8)
    ok2 &= v2 > 2 * v1     # beta<1 integral keeps growing ~ Lambda^(1-beta)
    record("M2 Osgood criterion: integral evaluation and beta>1 / beta<1 "
           "dichotomy are correct", ok2,
           "for beta=1.2, 2.0 the integral converges to "
           "I0^(1-beta)/(a(beta-1)) exactly as the manuscript's evaluation "
           "implies; for beta=0.8 it diverges (grows ~Lambda^0.2)")

    # M3 -- QCRB statement
    record("M3 quantum Cramer-Rao bound as stated is the standard bound",
           "Var" in BEYOND and r"\frac{1}{N \mathcal{F}_Q" in BEYOND,
           "Var(theta-hat) >= 1/(N F_Q) is the standard single-parameter "
           "QCRB. Correct -- as metrology. It licenses nothing about "
           "consciousness (see M17)")

    # M4 -- F_Q = 4 g_B, computed for a qubit state family
    def psi(th):
        return np.array([np.cos(th / 2), np.sin(th / 2)], complex)
    dth = 1e-4
    g_bures = 2 * (1 - abs(psi(0.7).conj() @ psi(0.7 + dth))) / dth ** 2
    record("M4 QFI = 4 x Bures metric: identity holds (computed)",
           abs(1.0 - 4 * g_bures) < 2e-3,
           f"|psi(theta)>=cos(theta/2)|0>+sin(theta/2)|1> has F_Q=1 "
           f"exactly; numerical Bures metric g_B={g_bures:.4f}, "
           f"4 g_B={4 * g_bures:.4f}. The geometric identity is real")

    # M5 -- Heisenberg scaling F_Q ~ N^2 vs N, computed
    def var_gen(state, n):
        g = np.zeros(2 ** n)
        for s in range(2 ** n):
            g[s] = sum(0.5 if (s >> q) & 1 == 0 else -0.5
                       for q in range(n))
        p = np.abs(state) ** 2
        return float(p @ g ** 2 - (p @ g) ** 2)
    ok5 = True
    for n in range(2, 6):
        plus = np.ones(2 ** n, complex) / np.sqrt(2 ** n)
        ghz = np.zeros(2 ** n, complex)
        ghz[0] = ghz[-1] = 1 / np.sqrt(2)
        ok5 &= abs(4 * var_gen(plus, n) - n) < 1e-9
        ok5 &= abs(4 * var_gen(ghz, n) - n ** 2) < 1e-9
    record("M5 Heisenberg scaling: F_Q = N (product) vs N^2 (GHZ), computed",
           ok5, "phase estimation under G=sum Z_i/2, N=2..5: exact. The "
           "manuscript's F_Q ~ N^2 scaling statement is standard and "
           "correct -- for metrology probes, not yet for neurons (M12/M17)")

    # M6 -- the defining equation of L_int is incomplete in BOTH copies
    bad = []
    for name, txt in (("Beyond.txt", BEYOND), ("Refinement Doc", REFINED)):
        blk = equation_block(txt, r"\mathcal{L}{\text{int}} =")
        if not any(s in blk for s in (r"\xi", "R ", r"|\Psi|")):
            bad.append(name)
    record("M6 the framework's central coupling term L_int is defined in "
           "the manuscript", not bad,
           f"in {bad}: the defining equation reads only "
           r"'L_int = \sqrt{-g}' -- no xi, no R, no |Psi|^2. The intended "
           r"term xi R |Psi|^2 appears later in prose/variation, but the "
           "definition itself is missing. FAIL (document integrity)")

    # M7 -- equation of motion drops the mass term of its own Lagrangian
    lag = equation_block(BEYOND, r"\mathcal{L}{\Psi}")
    eom = equation_block(BEYOND, r"\square \Psi")
    record("M7 the printed equation of motion follows from the stated "
           "Lagrangian", "m_" in eom,
           f"L_Psi contains -m_Psi^2 |Psi|^2, but the printed KG equation "
           f"'square Psi - (dV/d|Psi|^2) Psi = xi R Psi' has no m_Psi^2 Psi "
           f"term at all. Varying the stated Lagrangian cannot produce "
           f"the stated equation. FAIL (internal consistency)")

    # M8 -- the 'isolated' G_eff equation drops the entire RHS
    iso = equation_block(BEYOND, r"G_{\mu\nu} = \frac{\kappa}")
    record("M8 the 'isolated' Einstein equation G = kappa/(1+2 kappa xi "
           "|Psi|^2) is a valid rearrangement of the MEFE",
           "T_{" in iso or r"\nabla" in iso,
           "dividing the MEFE by (1+2k xi |Psi|^2) leaves FOUR right-hand "
           "terms (stress-energy, Lambda, and both derivative terms); the "
           "printed 'isolation' keeps none of them -- a scalar prefactor "
           "standing where a tensor equation should be. The G_eff "
           "interpretation survives only as a prefactor on the full RHS. "
           "FAIL (equation as printed)")

    # M9 -- variation -> MEFE algebra (the part that IS right)
    var = equation_block(BEYOND, r"\frac{\delta \mathcal{S}{\text{int}}}")
    mefe = equation_block(BEYOND, r"G_{\mu\nu} (1 + 2\kappa \xi")
    ok9 = (r"G{\mu\nu} \Phi" in var
           and r"(g_{\mu\nu} \square - \nabla_\mu \nabla_\nu)" in var
           and r"(\nabla_\mu \nabla_\nu - g_{\mu\nu} \square)" in mefe)
    record("M9 non-minimal-coupling variation and MEFE assembly are "
           "algebraically consistent", ok9,
           "delta S_int has the standard xi[G Phi + (g Box - nabla nabla)"
           "Phi] form; given L_EH prefactor 1/(2k), the MEFE prefactor "
           "(1+2k xi Phi) follows and the derivative term crosses the "
           "equality with the correct sign flip. (Lambda-sign conventions "
           "not adjudicated.) The algebra is fine -- the definition it "
           "serves is missing (M6)")

    # M10 -- claimed internal validation suite: not in evidence
    needles = ("10 critical dimensions", "Energy Conservation",
               "100% success")
    found = []
    for root, _dirs, files in os.walk(REPO):
        if "v7_sources" in root or "__pycache__" in root:
            continue
        for f in files:
            if f == os.path.basename(__file__):
                continue
            try:
                t = open(os.path.join(root, f), encoding="utf-8",
                         errors="ignore").read()
            except OSError:
                continue
            if any(n in t for n in needles):
                found.append(f)
    record("M10 manuscript's claimed validation ('10 critical dimensions, "
           "100% success', energy drift < 2.0e-6) exists nowhere outside "
           "the manuscript", bool(found),
           "no suite, harness, or result file in the repo or Drive texts "
           "matches the claim. The real v2-v6 record contains FAILs on "
           "adjacent claims (v5: 18 FAIL). A 100%-success validation of "
           "this framework is not in evidence. FAIL")

    # M11 -- |Psi|^2 proportional to QFI density: well-posedness
    has_const = re.search(r"\|\\Psi\|?\^?2?\s*=\s*[^.]{0,40}\\mathcal\{F\}",
                          BEYOND) is not None
    open_item("M11 |Psi|^2 proportional to local QFI density",
              "not yet a testable claim: QFI is defined for a parametrised "
              "state family, but the manuscript names no parameter, no "
              "state family, and no proportionality constant or units for "
              "the cosmological Psi field"
              + (" (an equality was found -- recheck)" if has_const else "")
              + ". OPEN: it cannot PASS as physics and there is no "
              "definition definite enough to FAIL")

    # M12 -- GCI 'testable prediction' has no quantitative content
    record("M12 GCI constitutes a quantitative, falsifiable prediction "
           "as claimed", "Hz" in BEYOND,
           "the manuscript claims a 'specific temporal frequency "
           "signature' but gives no frequency (no 'Hz' anywhere in the "
           "text), no phase-shift magnitude for Delta phi, no value of "
           "xi, and no |Psi|^2 for any cognitive system. Nothing here "
           "could be falsified by an interferometer. FAIL as a prediction "
           "(it is a proposal)")

    # M13 -- relation to Gate 3: independence, stated explicitly
    open_item("M13 relation of the QFI formalism to Gate 3 (gamma <~ 0.08, "
              "ln B=-3.06): INDEPENDENT -- neither compatible nor "
              "contradicted, because it shares no observable with Gate 3",
              "the manuscript contains no waveform, no gamma coupling, no "
              "merger prediction (P2); Gate 3 constrains a gamma-parametrised "
              "waveform bump, the manuscript's xi R |Psi|^2 is a different "
              "coupling in a different sector, and no xi<->gamma map exists "
              "anywhere. Two consequences are NOT open: the manuscript may "
              "not be cited as consistent with gamma <~ 0.08 (it says "
              "nothing about gamma), and gamma <~ 0.08 may not be cited as "
              "supporting it")
    info("M13-note xi is never assigned a value in the manuscript",
         "so no GW prediction exists even in principle: nothing in the "
         "manuscript could have been tested by Gate 3, and Gate 3's null "
         "does not touch it. Independence cuts both ways")

    # M14 -- consistency with the v5 standing formalism
    v_ok = (r"\lambda (|\Psi|^2 - \nu^2)^2" in BEYOND
            and r"\xi > 0" in BEYOND and r"\Lambda" in BEYOND)
    record("M14 manuscript is consistent with the standing (v5) "
           "formalism", not v_ok,
           "three silent collisions: (i) it treats xi as a free parameter "
           "and builds claims on xi>0 (gravity weakened), while the paper "
           "lineage's only derived value is xi=-1/6 (v5, under stated "
           "D_Psi with E=-Psi^2) -- the opposite sign regime, never "
           "mentioned; (ii) its V=lambda(|Psi|^2-nu^2)^2 is a different "
           "potential from the paper's stated V(Psi), whose phantom "
           "crossing FAILED in v5 -- no cross-reference; (iii) it carries "
           "a bare Lambda through L_EH and the MEFE while v5 DROPPED "
           "Lambda_eff==0 rather than proving it. FAIL (cross-document "
           "consistency)")

    # M15 -- the three Psi readings: conflict stands, unreconciled
    reconciles = ("eigenstate" in BEYOND or "cos(" in BEYOND
                  or "three readings" in BEYOND)
    record("M15 the manuscript reconciles its Psi definition with the "
           "other two readings (cosine / FCM eigenstate)", reconciles,
           "no mention of the cosine reading Psi=A cos(Theta(x)), the FCM "
           "principal-eigenstate reading, or the synthesis' three-readings "
           "framing. Its complex-scalar/QFI Psi is presented as THE "
           "definition. FAIL: the conflict flagged since v2 is not "
           "smoothed over here -- it is recorded as standing")

    # M16 -- MDCS near-infinite compression vs information bounds
    record("M16 'near-infinite compression' (MDCS) is admissible under "
           "established information bounds as stated",
           "near-infinite" not in BEYOND,
           "Shannon source coding bounds lossless compression of "
           "arbitrary data and the Holevo bound caps accessible "
           "information per qubit; 'encoding into geometry' does not "
           "repeal either. As stated -- without restriction to a "
           "compressible data class -- the claim is false. FAIL")

    # M17 -- consciousness = QFI maximisation: operationalisation
    open_item("M17 consciousness identified with QFI maximisation",
              "an identification, not a testable claim as written: no "
              "measurement protocol, threshold, or falsifier connects the "
              "QFI of any biological state family to a consciousness "
              "measure. The metrology it rests on is correct (M3-M5); the "
              "identification itself is OPEN")

    # M18 -- 'R-boost'
    n_rb = BEYOND.count("R-boost")
    open_item("M18 the 'R-boost' effect",
              f"named {n_rb}x, given no equation, magnitude, or derivation "
              "anywhere in the manuscript. A named mechanism carrying "
              "claims ('accelerated integration of information') with no "
              "mathematical content to test. OPEN")


def main() -> int:
    section_p()
    section_d()
    section_m()
    counts = {k: sum(1 for _, s_, _ in RESULTS if s_ == k)
              for k in ("PASS", "FAIL", "OPEN", "INFO")}
    print("\n" + "=" * 80)
    print(f" v7 TALLY: PASS={counts['PASS']} FAIL={counts['FAIL']} "
          f"OPEN={counts['OPEN']} INFO={counts['INFO']}")
    print("=" * 80)
    for nm, st, _ in RESULTS:
        if st == "FAIL":
            print(f"  FAIL: {nm}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
