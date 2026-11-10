# v7 test batch — QFI manuscript vs the Dissection, against the standing record

**Date:** 2026-11-09. Putt's instruction: "Yes do all your suggestions" (Drive-audit follow-ups).
**Result: PASS=15 FAIL=8 OPEN=5 INFO=5.** Suite: `aethel_psi_suite_v7.py` (exit 1, correct behaviour — FAILs present); full output: `aethel_psi_suite_v7_output.txt`. Sources in `v7_sources/`.
**No paper edited. No GW run started. v2–v6 suites and outputs untouched.**

Documents tested (read in full from Google Drive, read-only):
- **(a)** "Beyond Supremacy: The Apophthegmatic Intelligentsia of Quantum Temporality" (Jan 2026) — Ψ as a complex scalar whose |Ψ|² tracks Quantum Fisher Information. Two copies (Beyond.txt, 26,636 B; Google Doc "Rigorous Mathematical Refinement…", 38,243 chars) — bodies identical; the Doc adds only a Works Cited list.
- **(b)** "Rigorous Scientific Paper Dissection" (Aug 2025) — an AI critical review of the **FCM white paper**. It predates (a) and does not target it; v7 tests its objections on their merits against established theory and the standing record.

Standing constraints (not re-derived): GW250114 PE γ median 0.0235, **95% upper 0.0813**, ln B = −3.06 ± 0.23 (the −55.2 profile value is never quoted as a Bayes factor); spin-γ CLOSED (v6); v5 census (β_A = 1/14 FITTED, ξ = −1/6 derived only under stated D_Ψ with E = −Ψ², phantom crossing FAILED with the paper's stated V(Ψ), Λ_eff ≡ 0 DROPPED). γ = 0.4 appears nowhere as a paper value. No γ/width/f_merger was refitted; this batch ran no GW analysis at all.

## Summary

| # | Test | Verdict |
|---|---|---|
| D1–D9 | The Dissection's eight adjudicable objections (Dirac spectrum, fine-tuned mass, spectral action, CMB figure, CHSH "anomaly", EEG loophole, availability links, its own reference facts) | **Sustained — 8 PASS, 1 OPEN (p-hacking not adjudicable)** |
| M1–M5, M9 | Manuscript's borrowed mathematics (growth ODE, Osgood, QCRB, F_Q = 4g_B, Heisenberg scaling, MEFE variation algebra) | **PASS — all correct** |
| M6–M8, M10, M12 | Manuscript's own formalism and validation claims | **FAIL — central term undefined, EOM inconsistent, G_eff equation false as printed, claimed 100% validation not in evidence, GCI not quantitative** |
| M13 | Relation to Gate 3 | **OPEN — INDEPENDENT: no shared observable (see §3)** |
| M14–M16 | Consistency with v5 formalism / reconciliation of Ψ readings / MDCS compression | **FAIL** |
| M11, M17, M18 | \|Ψ\|²∝QFI well-posedness / consciousness identification / "R-boost" | **OPEN — not testable as written** |

## 1. The Dissection's objections — sustained almost in full

Full details in the suite output; the load-bearing ones:

- **D1 (Dirac spectrum).** The FCM paper's Eq. (2), λ = ±√((n+½)²/R² + m²), cannot be the Dirac spectrum on S³×S¹: it has one quantum number where the product spectrum needs two (the (k′/L)² circle term is absent), and even read as S³ alone its offset is ½ against the standard (k+3/2)/R. Sustained. At best Eq. (2) is an unnamed restricted sector — the paper does not say so.
- **D2 (the "fatal" fine-tuned parameter).** m = 7.373×10⁻⁵¹ kg = 4.14×10⁻¹⁵ eV — 13.5 orders of magnitude below the neutrino bound scale, 11.7 below the dark-energy scale, derived nowhere in the text. **The Dissection's central charge is sustained against the FCM paper.** Note the direction of travel since: v5's own census relabelled the surviving fitted parameters honestly (β_A = 1/14 FITTED; ξ = −1/6 derived only under a stated D_Ψ). The charge was fatal to the FCM paper's "first-principles / parameter-free" framing, and that framing is exactly what the later suites dropped.
- **D5 (CHSH "anomaly"), computed.** For |Φ⁺⟩ with standard settings S = 2.828. Apply the paper's "Ψ-modulation" R_Y(π/2) to one qubit and keep the *same* settings: **S = 0.000**, while concurrence stays 1.000 and the optimised-settings S_max stays 2.828. A local unitary cannot change entanglement; the entire "anomaly" is fixed-settings bookkeeping of rotated correlations. The paper's specific 0.98 is its settings/angle, not physics. Sustained — by computation, not rhetoric.
- **D6 (EEG loophole).** Sustained, and moot: the target claim was already retracted in the workspace record (Revised Edition outline: "Bell-EEG — Claim: S=0.115 claim retracted"). Nothing in the Third Revised Edition rests on it.
- **D4 (CMB).** No quantitative FCM CMB spectrum or fitted peak table exists anywhere in the tested record; Figure 3 is a schematic. Sustained. (The standing ACT DR6 ripple tests are a different observable, and null.)
- **D3 (spectral action).** Structural point sustained: the action is the single trace Tr f(D/Λ); its heat-kernel expansion *produces* the Λ⁴ and Λ²R terms — they are not independent pieces to which a Kontsevich–Vishik trace can be appended as a "quantum term." Exact prefactors (f-moments) are convention-dependent and were not adjudicated (INFO).
- **D8 (availability links).** The FCM paper cites zenodo.org/records/1234567 and github.com/Aethelred/FCMspectral; the real artifacts are github.com/derrickdagod666/aethel-psi and DOI 10.5281/zenodo.24038497. Live check for this suite: the cited GitHub URL returned **404**. Sustained on the record. The Dissection's further inference of intent ("misconduct", "deliberate deception") goes beyond the checkable fact and is **not** endorsed here.
- **D7 (p-hacking).** OPEN — the FCM-era dataset exists nowhere accessible, and no pre-registration exists either way. The concern stands unrebutted; its target is withdrawn regardless.
- **Balance (D9/D10).** The critique's own checkable facts are accurate (Tsirelson 2√2, the eV↔kg conversion). It overreaches in one place: §3.1 says the rotation "destroys entanglement" — D5 shows concurrence is invariant; the correct statement is the fixed-settings one. Its conclusion survives its wording.

## 2. The QFI manuscript — sound borrowed mathematics, broken own formalism

**What passes is real.** The intelligence-growth ODE's closed form and finite-time blow-up t_max are exactly right (verified against solve_ivp); the Osgood-criterion application is right, including the β<1 divergence side; the QCRB is stated correctly; **F_Q = 4 g_B is a true identity** (computed: qubit family, g_B = 0.2500, 4g_B = 1.0000 = F_Q); Heisenberg scaling F_Q = N vs N² is exact (computed N=2..5); and the non-minimal-coupling variation → MEFE assembly is algebraically consistent, prefactor (1+2κξΦ) and sign flip included. None of this is the manuscript's invention — it is standard metrology and scalar-tensor algebra, correctly transcribed.

**What fails is the manuscript's own contribution:**

- **M6 — the central term is never defined.** In *both* copies, the defining equation for the interaction Lagrangian reads, in full: L_int = √−g. No ξ, no R, no |Ψ|². The intended ξR|Ψ|² appears later in prose and in the variation — but the framework's core term exists only by inference from its own variation. FAIL (document integrity).
- **M7 — the equation of motion drops its own mass term.** L_Ψ contains −m_Ψ²|Ψ|²; the printed Klein–Gordon equation has no m_Ψ²Ψ term at all. Varying the stated Lagrangian cannot produce the stated equation. FAIL.
- **M8 — the "isolated" G_eff equation is false as printed.** Dividing the MEFE by (1+2κξ|Ψ|²) leaves four right-hand terms; the printed isolation G_μν = κ/(1+2κξ|Ψ|²) keeps none of them. The "gravity is weakened where |Ψ|² is high" reading survives only as a prefactor on the full right-hand side, not as the printed equation. FAIL.
- **M10 — the claimed validation does not exist.** The manuscript asserts "a comprehensive validation suite… across 10 critical dimensions, achieving a 100% success rate" with energy drift < 2.0×10⁻⁶. No such suite, harness, or result exists in Drive, the repo, or the workspace; the real v2–v6 record contains FAILs on adjacent claims (v5 alone: 18). FAIL — the same failure mode as the FCM paper's CMB figure (D4): a validation *narrated*, not performed.
- **M12 — GCI is not a prediction.** "Gravitational Consciousness Interferometry" claims a "specific temporal frequency signature," but the text contains no frequency (the string "Hz" appears nowhere), no Δφ magnitude, no ξ value, no |Ψ|² for any cognitive system. There is nothing an interferometer could falsify. FAIL as a prediction; it is a proposal.
- **M14 — silent collisions with the standing formalism.** (i) ξ is treated as free, with physical claims built on ξ>0 — while the lineage's only derived value is ξ=−1/6 (v5, under stated assumptions): the opposite sign regime, never mentioned. (ii) Its V = λ(|Ψ|²−ν²)² is a different potential from the paper's stated V(Ψ) whose phantom crossing FAILED in v5 — no cross-reference. (iii) It carries a bare Λ through the action and field equations while v5 DROPPED Λ_eff≡0 rather than proving it. FAIL.
- **M15 — the Ψ conflict stands.** The manuscript never mentions the cosine reading Ψ = A cos(Θ(x)), the FCM eigenstate reading, or the synthesis' three-readings framing; its complex-scalar/QFI Ψ is presented as *the* definition. This is a silent replacement, not a reconciliation. The three incompatible definitions remain three.
- **M16 — MDCS "near-infinite compression"** contradicts the Shannon source-coding and Holevo bounds as stated (no restriction to a compressible data class). FAIL.
- **OPEN, honestly:** M11 (|Ψ|² ∝ QFI density names no parameter, state family, constant, or units — not yet a claim definite enough to pass *or* fail), M17 (consciousness = QFI maximisation is an identification with no protocol, threshold, or falsifier), M18 ("R-boost": named twice, given no equation or magnitude).

## 3. The Gate 3 question — INDEPENDENT, and what that forbids

The manuscript's body contains **no** reference to γ, Gate 3, gravitational waves, mergers, or GW250114 at all (verified term-by-term; the Doc version's only hits are inside Works Cited titles). Gate 3 constrains a γ-parametrised waveform modification; the manuscript's ξR|Ψ|² is a different coupling in a different sector, ξ is never assigned a value, and no ξ↔γ map exists anywhere. **Verdict: the formalism is independent of the standing Gate 3 results — neither compatible with them nor contradicted by them, because it makes no prediction in their observable space.**

Two consequences are not open and should govern any future citation:
1. The manuscript may **not** be cited as "consistent with γ ≲ 0.08" — it says nothing about γ.
2. γ ≲ 0.08 (and ln B = −3.06) may **not** be cited as supporting the manuscript — Gate 3's null does not touch it, and its claimed gravitational signatures of cognition (M12) remain untested and, as written, untestable.

## v7 check counts

**PASS=15 FAIL=8 OPEN=5 INFO=5** (28 verdict checks + 5 INFO). D-section PASS means "objection sustained"; M-section verdicts are on the manuscript's claims as printed. Counts are per check and not directly comparable to v5/v6 (different check sets, different documents).

## What remains OPEN / next steps

- M11 could become testable only by specifying the QFI parameter, state family, and the constant (with units) linking |Ψ|² to F_Q. Until then the QFI reading has no quantitative content beyond standard metrology.
- M6/M7/M8 are repairable document defects (write L_int, restore m_Ψ² in the EOM, un-isolate the G_eff equation) — but repairing them is editing the manuscript, which this batch was instructed not to do. They are recorded here for any future edition decision, which is Putt's.
- Nothing in v7 changes any standing result: γ ≲ 0.08 (95%) and ln B = −3.06 stand; spin-γ stays closed; no new mechanism was invented to connect the QFI manuscript to them.

## Files changed / created (v7 only)

- `validation-suites/aethel_psi_suite_v7.py` — this suite
- `validation-suites/aethel_psi_suite_v7_output.txt` — full run output (exit 1)
- `validation-suites/v7_sources/` — the three source texts as read from Drive
- `validation-suites/V7_TESTS.md` — this report
- `validation-suites/README.md` — v7 row added
- Summary copy: `~/workspace/user/files/drive-science-audit/V7_VALIDATION_REPORT.md`

Nothing else was modified: v2–v6 suites and outputs, all paper editions, and all Drive files are untouched. No dynesty/Bilby run was started by this batch.
